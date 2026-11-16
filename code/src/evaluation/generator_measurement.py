"""Generator-side measurements; no reward, control, or termination changes.

Policy and physics frequencies must match: contact timestamps are physics ticks.
THW tracking is conditional on valid following; exposure denominators are kept.
"""
from __future__ import annotations

from dataclasses import asdict
import hashlib
import json
import math

import numpy as np

from src.environments.physical_defender_env import body_on_straight_road
from src.evaluation.counterfactual_probe import ContactRecorder, confirmed_contacts
from src.evaluation.safety_metrics import VehicleState, constant_velocity_ttc, following_headway

VERSION = "generator-measurement-v1"


def json_value(value):
    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    raise TypeError(type(value).__name__)


def digest(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(",", ":"),
                                    default=json_value, allow_nan=False).encode()).hexdigest()


def world_snapshot(env):
    """Exact JSON state for all actors, road geometry, config and simulator RNG.

Exclude object links (road/lane), logs and methods. Lane geometry and class
implementations are recorded separately. The command and target curve are
deliberately outside the physical world hash, in the run configuration.
"""
    actors = []
    for vehicle in env.inner.road.vehicles:
        state = {}
        for key, value in vars(vehicle).items():
            if key in {"road", "lane", "history"} or callable(value):
                continue
            # Unknown state must fail closed rather than silently disappear.
            json.dumps(value, default=json_value, allow_nan=False)
            state[key] = value
        actors.append({"class": type(vehicle).__module__ + "." + type(vehicle).__qualname__, "state": state})
    road = env.inner.road
    rng = road.np_random
    rng_state = rng.bit_generator.state if hasattr(rng, "bit_generator") else rng.get_state()
    snapshot = {"actors": actors, "generator_id": road.vehicles.index(env.inner.vehicle),
            "target_id": road.vehicles.index(env.target), "rng": rng_state,
            "road": road.network.to_config(), "config": env.inner.config,
            "time": env.inner.time, "steps": env.inner.steps}
    return json.loads(json.dumps(snapshot, default=json_value, allow_nan=False))


def following_drac(follower, leader):
    """Longitudinal closing-speed squared / (2 bumper gap).

    Defined only for aligned, laterally overlapping, approaching following
    with positive gap. Non-approach, overlap and other geometry return None.
    """
    if abs(follower.heading) > math.radians(15) or abs(leader.heading) > math.radians(15):
        return None
    headway = following_headway(follower, leader)
    axis = (math.cos(follower.heading), math.sin(follower.heading))
    speed = follower.vx * axis[0] + follower.vy * axis[1]
    closing = (follower.vx - leader.vx) * axis[0] + (follower.vy - leader.vy) * axis[1]
    if headway is None or headway <= 0 or closing <= 0:
        return None
    return closing ** 2 / (2 * headway * speed)


def contact_outcome(events, last_tick, pair_ids, crashed_ids):
    confirmed = confirmed_contacts(events, last_tick)
    pair_ids = set(pair_ids)
    involved = [event for event in confirmed if pair_ids.intersection(event["actors"])]
    attributed = {actor for event in confirmed for actor in event["actors"]}
    return {
        "pair_contact": any(set(event["actors"]) == pair_ids for event in involved),
        "background_contact": any(set(event["actors"]) != pair_ids for event in involved),
        "any_traffic_contact": bool(confirmed),
        "unattributed_crashed_ids": sorted(set(crashed_ids) - attributed),
        "confirmed_contacts": confirmed,
        "pending_impacts": [event for event in events if event["kind"] == "scheduled_impact"
                            and event["step"] == last_tick],
    }


class GeneratorMeasurement:
    def __init__(self, env, reset_info):
        if env.driving._sim_freq != env.driving._policy_freq:
            raise ValueError("One physics tick per policy step required for contact attribution")
        if env.scenario != "cut_in":
            raise ValueError("This adapter currently freezes cut-in roles only")
        self.env = env
        self.initial_world = world_snapshot(env)
        self.initial_hash = digest(self.initial_world)
        self.reset_info = reset_info
        self.vehicles = list(env.inner.road.vehicles)
        self.pair_ids = [self.vehicles.index(env.inner.vehicle), self.vehicles.index(env.target)]
        self.recorder = ContactRecorder(self.vehicles)
        self.dt = 1 / env.driving._policy_freq
        self.trace = []
        self.previous_speeds = [float(env.inner.vehicle.speed), float(env.target.speed)]
        self.previous_acceleration = None
        self.initial_x = [float(env.inner.vehicle.position[0]), float(env.target.position[0])]
        self.reset_invalid = self._invalid_reset()

    def _invalid_reset(self):
        states = [VehicleState.from_vehicle(v) for v in self.vehicles]
        overlaps = [[i, j] for i in range(len(states)) for j in range(i + 1, len(states))
                    if constant_velocity_ttc(states[i], states[j]) == 0]
        invalid = [i for i, v in enumerate(self.vehicles) if v.crashed
                   or not body_on_straight_road(v, self.env.driving._lanes_count) or float(v.velocity[0]) < -.1]
        return {"overlap_pairs": overlaps, "invalid_actor_ids": invalid}

    def before_step(self):
        self.recorder.step = len(self.trace)

    def after_step(self, info):
        env = self.env
        if list(env.inner.road.vehicles) != self.vehicles:
            raise RuntimeError("Actor registry changed during measured episode")
        g, t = VehicleState.from_vehicle(env.inner.vehicle), VehicleState.from_vehicle(env.target)
        speeds = [float(env.inner.vehicle.speed), float(env.target.speed)]
        acceleration = [(new - old) / self.dt for new, old in zip(speeds, self.previous_speeds)]
        jerk = (None if self.previous_acceleration is None else
                [(new - old) / self.dt for new, old in zip(acceleration, self.previous_acceleration)])
        self.previous_speeds, self.previous_acceleration = speeds, acceleration
        valid = bool(info["interaction_valid"] and info["follower_forward_speed"] >= 15)
        self.trace.append({
            "tick": len(self.trace), "generator": asdict(g), "target": asdict(t),
            "valid_following": valid,
            "headway_error": abs(info["headway"] - env.target_headway) if valid else None,
            "pair_ttc": constant_velocity_ttc(g, t), "following_drac": following_drac(t, g),
            "acceleration": acceleration, "jerk": jerk,
            "generator_offroad": not body_on_straight_road(env.inner.vehicle, env.driving._lanes_count),
            "target_offroad": not body_on_straight_road(env.target, env.driving._lanes_count),
            "generator_wrong_way": g.vx < -.1, "target_wrong_way": t.vx < -.1,
            "shield_intervened": bool(info["shield_intervened"]),
            "stop_guard_intervened": bool(info["low_speed_stop_guard_intervened"]),
            "high_level_action": info["high_level_action"], "applied_action": info["applied_action"],
        })

    def finish(self, info):
        rows, horizon = self.trace, self.env.driving._duration
        steps = len(rows)
        finite_ttc = [r["pair_ttc"] for r in rows if r["pair_ttc"] is not None]
        drac = [r["following_drac"] for r in rows if r["following_drac"] is not None]
        errors = [r["headway_error"] for r in rows if r["headway_error"] is not None]
        crashed_ids = [i for i, v in enumerate(self.vehicles) if v.crashed]
        contacts = contact_outcome(self.recorder.events, steps - 1, self.pair_ids, crashed_ids)
        offroad = any(r["generator_offroad"] or r["target_offroad"] for r in rows)
        wrong_way = any(r["generator_wrong_way"] or r["target_wrong_way"] for r in rows)
        physical_failure = bool(crashed_ids or contacts["any_traffic_contact"] or offroad or wrong_way)
        reset_valid = not any(self.reset_invalid.values())
        result = {
            "measurement_version": VERSION, "initial_hash": self.initial_hash,
            "initial_world": self.initial_world, "reset_info": self.reset_info,
            "reset_valid": reset_valid, "reset_invalid": self.reset_invalid,
            "steps": steps, "horizon": horizon, "crashed_ids": crashed_ids, **contacts,
            "physical_failure": physical_failure, "offroad": offroad, "wrong_way": wrong_way,
            "full_horizon": steps == horizon, "merge_complete": bool(info.get("merge_complete")),
            "interaction_ever_valid": bool(errors),
            "deadline_failure": bool(info.get("interaction_deadline_failure")),
            "termination_reason": info.get("termination_reason", "reset_rejected"),
            "valid_following_steps": len(errors), "valid_fraction_observed": len(errors) / steps if steps else None,
            "valid_fraction_horizon": len(errors) / horizon,
            "headway_mae": sum(errors) / len(errors) if errors else None,
            "ttc_defined_steps": len(finite_ttc), "ttc_defined_fraction": len(finite_ttc) / steps if steps else None,
            "min_pair_ttc": min(finite_ttc) if finite_ttc else None,
            "drac_defined_steps": len(drac), "drac_defined_fraction": len(drac) / steps if steps else None,
            "max_following_drac": max(drac) if drac else None,
            "exclusive_ttc_event": bool(finite_ttc and min(finite_ttc) < 1.5 and not crashed_ids
                                        and not contacts["any_traffic_contact"] and reset_valid),
            "generator_progress_m": float(self.env.inner.vehicle.position[0]) - self.initial_x[0],
            "target_progress_m": float(self.env.target.position[0]) - self.initial_x[1],
            "shield_steps": sum(r["shield_intervened"] for r in rows),
            "stop_guard_steps": sum(r["stop_guard_intervened"] for r in rows),
            "trace": rows,
        }
        for index, role in enumerate(("generator", "target")):
            for field in ("acceleration", "jerk"):
                values = [abs(r[field][index]) for r in rows if r[field] is not None]
                result[f"{role}_abs_{field}_p95"] = float(np.quantile(values, .95)) if values else None
        return result

    def close(self):
        self.recorder.close()
