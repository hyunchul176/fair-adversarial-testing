"""Family-B causal, reversible longitudinal adversary with frozen lateral path.

Real simulator episodes run only on the research server. No longitudinal
window/reference is accepted, computed, loaded or exposed by this adapter.
"""
from copy import deepcopy
from dataclasses import asdict
import math

import gymnasium as gym
import numpy as np

from src.evaluation.adversarial_agent_core import RiskAccumulator, ttc_risk
from src.evaluation.adversarial_b1_core import (B1ObservationHistory, acceleration_request,
                                               causal_frame)
from src.evaluation.adversarial_native_action import acceleration_command, steering_command
from src.evaluation.generator_measurement import GeneratorMeasurement, contact_outcome, digest
from src.evaluation.observed_longitudinal_replay import setup, lateral_reference
from src.evaluation.predictive_recognition import controller
from src.evaluation.safety_metrics import VehicleState, constant_velocity_ttc


VERSION = 'direct-adversarial-cutin-B1-v1'


def unimpacted_state(vehicle, consumed_impact):
    state = asdict(VehicleState.from_vehicle(vehicle))
    if consumed_impact is not None:
        state['x'] -= float(consumed_impact[0])
        state['y'] -= float(consumed_impact[1])
    return state


def body_within_road(state):
    radius = .5*(state['length']*abs(math.sin(state['heading']))
                 + state['width']*abs(math.cos(state['heading'])))
    return state['y']-radius >= -2-1e-9 and state['y']+radius <= 10+1e-9


class DirectAdversarialCutInEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, base, case, protocol, target_h=0., observation_mode='reactive',
                 target='following', arm='continue'):
        super().__init__()
        if target != 'following' or arm != 'continue' or target_h not in (0., .5):
            raise ValueError('Family-B freezes following H0/H.5 and continue path')
        if (base['environment']['simulation_frequency'], base['environment']['policy_frequency'],
            base['environment']['duration']) != (10, 10, 120):
            raise ValueError('Frozen 10Hz/120-tick inner contract required')
        if (protocol['reward']['gamma'] != 1 or protocol['reward']['horizon_bootstrap'] != 0
                or protocol['action']['requests_mps2'] != [0, -3, 3]
                or protocol['observation']['shape'] != [108]):
            raise ValueError('Registered Family-B task contract required')
        self.base, self.case, self.protocol = map(deepcopy, (base, case, protocol))
        self.contract = self.protocol
        self.target_h, self.target_id, self.arm = float(target_h), target, arm
        self.observation_mode = observation_mode
        self.divisors = [f['divisor'] for f in protocol['observation']['features']]
        B1ObservationHistory(self.divisors, self.observation_mode)
        self.action_space = gym.spaces.Discrete(3)
        self.observation_space = gym.spaces.Box(-1, 1, shape=(108,), dtype=np.float32)
        self.sim = self.meter = None
        self.done = True

    def _reset_native(self):
        return setup(self.base, self.case, self.arm, source=False)

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.sim, reset = self._reset_native()
        if len(self.sim.inner.road.vehicles) != 2:
            self.close()
            raise RuntimeError('Family-B requires exactly two vehicles')
        self.meter = GeneratorMeasurement(self.sim, reset)
        if any(self.meter.reset_invalid.values()):
            self.close()
            raise ValueError('Invalid registered reset')
        self.initial = [asdict(VehicleState.from_vehicle(v)) for v in self.sim.inner.road.vehicles]
        self.initial_ttc = constant_velocity_ttc(*[VehicleState(**s) for s in self.initial])
        self.initial_speed = float(self.sim.inner.vehicle.speed)
        self.reward_state = RiskAccumulator(self.initial_ttc)
        self.target_control = controller(self.base['idm_reference'], self.target_h)
        self.cruise = self.case['event']['metrics']['p10']['target_speed']
        self.tick = 0
        self.previous_a = self.previous_steer = self.previous_target = self.previous_request = 0.
        self.acceleration = [0., 0.]
        self.ys = [self.initial[0]['y']]
        self.done = False
        self.maneuver_valid = True
        self.outcome = None
        self.last_info = reset
        self.decisions = []
        self.history = B1ObservationHistory(self.divisors, self.observation_mode)
        obs, _ = self._observation()
        return obs, dict(initial_hash=self.meter.initial_hash, version=VERSION,
                         initial_ttc=self.initial_ttc,
                         physical_initial_hash=digest(self.initial),
                         reset_seed_contract='fixed archived case seed')

    def _observation(self):
        # The returned next-step observation is cached. Calling step never
        # appends another copy of that decision frame or double-counts clipping.
        if self.history.snapshots and self.history.snapshots[-1]['tick'] == self.tick:
            return self.history.current()
        g, t = [asdict(VehicleState.from_vehicle(v)) for v in self.sim.inner.road.vehicles]
        raw = causal_frame(g, t, acceleration=self.acceleration, tick=self.tick,
            lateral_y=lateral_reference(self.base, self.case, self.arm, self.tick*.1),
            previous_steer=self.previous_steer, maximum=self.reward_state.maximum,
            initial_speed=self.initial_speed, scalar_speed=float(self.sim.inner.vehicle.speed),
            previous_request=self.previous_request)
        return self.history.push(raw, self.tick)

    def _generator_action(self, request):
        g = self.sim.inner.vehicle
        y = lateral_reference(self.base, self.case, self.arm, (self.tick+1)*.1)
        desired_vy = (y-float(g.position[1]))/.1
        beta = math.asin(float(np.clip(desired_vy/float(g.speed), -.99, .99)))-float(g.heading)
        steer_request = math.atan(2*math.tan(beta))
        steer_cmd, steer = steering_command(steer_request, self.previous_steer)
        accel_cmd, projection = acceleration_command(request, float(g.speed), self.previous_a)
        action = np.asarray([accel_cmd, steer_cmd], dtype=np.float32)
        return action, dict(y_reference=y, projection=projection, requested_mps2=request,
                            requested_steering=steer_request,
                            native_acceleration=projection['applied_mps2'],
                            native_steering=steer, tape_passthrough=False)

    def _check_motion(self, row, gc, tc, pending):
        tol = self.protocol['physical_bounds']['audit_tolerance']
        g, t = [unimpacted_state(v, p) for v, p in zip(self.sim.inner.road.vehicles, pending)]
        a, ta = row['acceleration']
        self.ys.append(g['y'])
        ay = (self.ys[-1]-2*self.ys[-2]+self.ys[-3])/.01 if len(self.ys) >= 3 else 0.
        jy = (self.ys[-1]-3*self.ys[-2]+3*self.ys[-3]-self.ys[-4])/.001 if len(self.ys) >= 4 else 0.
        checks = dict(acceleration=abs(a) <= 3+tol,
            jerk=abs((a-self.acceleration[0])/.1) <= 2+tol,
            speed=15-tol <= float(self.sim.inner.vehicle.speed) <= 40+tol,
            realization=abs(a-gc['native_acceleration']) <= tol,
            steering=abs(gc['native_steering']) <= .3+tol,
            steering_slew=abs((gc['native_steering']-self.previous_steer)/.1) <= .15+tol,
            path=abs(g['y']-gc['y_reference']) <= 1e-5,
            lateral_acceleration=abs(ay) <= 2+tol, lateral_jerk=abs(jy) <= 4+tol,
            road=body_within_road(g), forward=g['vx'] >= -tol)
        target_checks = dict(acceleration=abs(ta) <= 5+tol,
            jerk=abs((ta-self.acceleration[1])/.1) <= 2.5+tol,
            realization=abs(ta-tc['applied_mps2']) <= tol,
            road=body_within_road(t), forward=t['vx'] >= -tol)
        if not all(target_checks.values()):
            raise RuntimeError('Target actuator/road contract failed: '+str(target_checks))
        row.update(maneuver_checks=checks, target_checks=target_checks,
                   unimpacted_generator=g, unimpacted_target=t,
                   consumed_impacts=[None if p is None else p.tolist() for p in pending],
                   lateral_acceleration=ay, lateral_jerk=jy,
                   path_error=abs(g['y']-gc['y_reference']))
        return all(checks.values())

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset required after finite episode end')
        request = acceleration_request(action)
        observation, snapshot = self._observation()
        decision = dict(snapshot, index=len(self.decisions),
                        observation_hash=digest(observation), action=int(action),
                        requested_acceleration_mps2=request)
        self.decisions.append(decision)
        self.previous_request = request
        reward = 0.
        for _ in range(5):
            pending = [None if v.impact is None else v.impact.copy() for v in self.sim.inner.road.vehicles]
            if any(v.crashed for v in self.sim.inner.road.vehicles):
                raise RuntimeError('Continuing a crashed episode')
            self.previous_target, tc = self.target_control(
                self.sim, self.target_id, None, self.cruise, self.previous_target, 'viable_v1')
            applied, gc = self._generator_action(request)
            self.meter.before_step()
            _, _, term, trunc, info = self.sim.step(applied)
            info.update(high_level_action=applied.tolist(), applied_action=applied.tolist(),
                        shield_intervened=False, low_speed_stop_guard_intervened=False)
            self.meter.after_step(info)
            row = self.meter.trace[-1]
            row.update(target_control=tc, generator_control=gc,
                       generator_scalar_speed=float(self.sim.inner.vehicle.speed),
                       agent_action=int(action), decision_index=decision['index'])
            if (abs(float(self.sim.inner.vehicle.action['acceleration'])-gc['native_acceleration']) > 1e-8
                    or abs(float(self.sim.inner.vehicle.action['steering'])-gc['native_steering']) > 1e-8):
                raise RuntimeError('Native command mapping drift')
            self.maneuver_valid &= self._check_motion(row, gc, tc, pending)
            self.acceleration = row['acceleration'][:]
            # Measured acceleration, not the ideal request, owns next tick state.
            self.previous_a = self.acceleration[0]
            self.previous_steer = gc['native_steering']
            contacts = contact_outcome(self.meter.recorder.events, self.tick, self.meter.pair_ids,
                                      [i for i, v in enumerate(self.sim.inner.road.vehicles) if v.crashed])
            if contacts['unattributed_crashed_ids'] or contacts['background_contact']:
                raise RuntimeError('Unattributed/background collision in two-vehicle task')
            self.tick += 1
            outcome = 'ongoing'
            if not self.maneuver_valid:
                outcome = 'maneuver_invalid'
            elif contacts['pair_contact']:
                outcome = 'intended_contact'
            elif self.tick == 120:
                outcome = 'horizon'
            elif term or trunc:
                raise RuntimeError('Unexplained early simulator termination')
            r = self.reward_state.advance(row['pair_ttc'], outcome)
            row.update(agent_reward=r, agent_outcome=outcome)
            reward += r
            self.last_info = info
            if outcome != 'ongoing':
                self.done, self.outcome = True, outcome
                break
        next_observation, _ = self._observation()
        return next_observation, reward, self.done, False, dict(
            agent_outcome=self.outcome, physics_ticks=self.tick,
            maneuver_valid=self.maneuver_valid, agent_return=self.reward_state.total)

    def result(self):
        if not self.done or self.meter is None:
            raise RuntimeError('Only completed episodes have a result')
        row = self.meter.finish(self.last_info)
        valid = self.maneuver_valid
        risk = bool(valid and (row['pair_contact'] or row['exclusive_ttc_event']))
        initial_event = self.initial_ttc is not None and self.initial_ttc < 1.5
        all_q = [ttc_risk(self.initial_ttc)]+[ttc_risk(t['pair_ttc']) for t in row['trace']]
        row.update(agent_version=VERSION, agent_outcome=self.outcome,
            maneuver_valid_precontact=valid, valid_pair_contact=bool(valid and row['pair_contact']),
            valid_risk=risk, initial_risk_event=initial_event,
            agent_return=self.reward_state.total, initial_risk_score=self.reward_state.initial,
            initial_ttc=self.initial_ttc, initial_generator_scalar_speed=self.initial_speed,
            maximum_q=max(all_q), physical_initial_state=self.initial,
            physical_initial_hash=digest(self.initial), decisions=self.decisions,
            observation_snapshots=self.history.snapshots,
            observation_clip_counts=self.history.clip_counts.tolist(),
            raw_observation_clip_counts=self.history.raw_clip_counts.tolist(),
            observation_mode=self.observation_mode,
            tape=[t['applied_action'] for t in row['trace']],
            failed_maneuver_checks=sorted({k for t in row['trace'] for k, v in t['maneuver_checks'].items() if not v}),
            case_id=self.case['case_id'], event_id=self.case['event']['event_id'],
            target_id=self.target_id, target_h=self.target_h, arm=self.arm,
            speed_tracking_contract='retired_for_B; matched_zero_deviation_is_offline_descriptive')
        return row

    def finish(self):
        return self.result()

    def close(self):
        if self.meter is not None:
            self.meter.close()
            self.meter = None
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        self.done = True
