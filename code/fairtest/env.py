"""B7 development environment: B2 authority with continuous longitudinal requests and a SUT panel.

Unchanged from B2/B5: highway-env two-vehicle world, archived reset, generator actuator
(|a|<=3, jerk<=2, speed 15..40), quintic lateral planner, physical audit, contact attribution,
risk accumulation. Changes (all registered in the B7 design note):
  * longitudinal request is any value in [-3, 3] m/s^2 (the actuator projects jerk/speed);
  * target panel: following, reference_idm (H 0/.5/1) and the R157 CC driver;
  * lateral duration grid is configurable (default B2 grid);
  * lane-change corridor tolerance 1e-6 m instead of 1e-9 m, removing float-residual
    rejections (B5 MECHANISM_REVIEW: 188 spurious lane-4 rejections).
"""
from __future__ import annotations

from dataclasses import asdict
import math
import os

import numpy as np

from fairtest import devboot  # noqa: F401
from fairtest.cc_target import PERCEPTION_M, WANDER_M, CCDriver, LIMITS as CC_LIMITS, AEB_LIMITS as CC_AEB_LIMITS
from fairtest.sut_targets import FollowingAEB, LIMITS as AEB_LIMITS
from fairtest.ref_models import FSMDriver, RSSDriver, RSSLatSut, RSSSut, LIMITS as REF_LIMITS
from fairtest.oracle_refs import OracleBrake, LIMITS as ORACLE_LIMITS
from fairtest.b18_bc import BCHumanSUT, LIMITS as BC_LIMITS
from fairtest.sut_perception import PerceptionADS, LIMITS as ADS_LIMITS
from fairtest.cc_time import CCTimeDriver
from src.environments.causal_lateral_adversarial_env import CausalLateralAdversarialEnv
from src.environments.direct_adversarial_cutin_env import body_within_road, unimpacted_state
from src.evaluation import causal_lateral_reference as lateral
from src.evaluation.adversarial_agent_core import RiskAccumulator, release_distance
from src.evaluation.generator_measurement import contact_outcome, digest
from src.evaluation.safety_metrics import VehicleState, constant_velocity_ttc

TARGETS = ('following', 'reference_idm', 'cc_r157', 'cc_r157_react050', 'cc_r157_react100', 'following_aeb',
           'ref_fsm', 'ref_rss', 'rss_sut', 'cc_r157_strict', 'oracle_t0', 'oracle_react0', 'bc_human', 'ads', 'cc_r157_time',
           'rss_lat_sut', 'cc_r157_cal', 'drl_ext', 'drl_sut', 'acc_prod', 'acc_prod_aeb', 'cc_r157_aeb')
CC_REACTION = {'cc_r157': .75, 'cc_r157_react050': .5, 'cc_r157_react100': 1., 'cc_r157_strict': .75, 'cc_r157_cal': .85,
               'cc_r157_aeb': .75}  # B32: + the regulation's AEB layer (strongest reading, cc_target)
CC_EMERGENCY = {'cc_r157_cal': -7.0}  # CARLA-CC actuation: feed-forward braking reaches ~7.0 m/s^2 with ~0.1 s lag
CC_UNLATCH = {'cc_r157_strict': WANDER_M + PERCEPTION_M}  # post-B13 sensitivity: release below 1.095 m
CUSTOM_LIMITS = {**{k: CC_LIMITS for k in CC_REACTION}, 'following_aeb': AEB_LIMITS, **REF_LIMITS, **ORACLE_LIMITS, **BC_LIMITS, 'ads': ADS_LIMITS, 'cc_r157_time': CC_LIMITS,
                 'drl_ext': dict(acceleration=8., jerk=15.), 'drl_sut': dict(acceleration=8., jerk=15.),  # b25_drl.LIMITS
                 'acc_prod': dict(acceleration=3.5, jerk=2.5), 'acc_prod_aeb': dict(acceleration=9., jerk=30.),  # sut_acc.LIMITS
                 'cc_r157_aeb': CC_AEB_LIMITS}
HORIZONS = (0., .5, 1.)
_HUMAN_POP = []


def human_population():
    """B57: the sampled human-like reference drivers (JSON list at FAIRTEST_HUMAN_POP), loaded once per process."""
    if not _HUMAN_POP:
        import json
        _HUMAN_POP.append(json.load(open(os.environ['FAIRTEST_HUMAN_POP'], encoding='utf-8'))['drivers'])
    return _HUMAN_POP[0]
CORRIDOR_TOLERANCE_M = 1e-6
# G9 post-hoc authority intervention (unset = protocol unchanged): the generator's speed floor becomes
# max(15, initial speed - margin), i.e. the attacker keeps only `margin` m/s of deceleration headroom.
# Enforced like the protocol's own 15 m/s floor: the actuator saturates braking (jerk-limited release before
# the floor) and the plan stays valid; the audit check below only guards against an undershoot.
FLOOR_MARGIN = float(os.environ['FAIRTEST_FLOOR_MARGIN']) if os.environ.get('FAIRTEST_FLOOR_MARGIN') else None
# Relative variant: the floor is max(15, SUT initial speed - r) (capped at the generator's initial speed), i.e. the
# attacker can become at most r m/s slower than the SUT was, as NGSIM attackers (SUT speed - 15 = 1.6 m/s median).
FLOOR_REL = float(os.environ['FAIRTEST_FLOOR_REL']) if os.environ.get('FAIRTEST_FLOOR_REL') else None
# Lateral-clearance geometry (unset = protocol unchanged, 2.0 m): both vehicles' width. With the 4 m lanes the body
# clearance between lane-centred vehicles is 4 - width (highD median 3.84 - 1.92 = 1.92 m, p10-p90 1.41-2.24 m;
# CARLA Town04 with the Tesla box 3.5 - 2.16 = 1.34 m, i.e. width 2.66 here).
VEH_W = float(os.environ['FAIRTEST_VEH_W']) if os.environ.get('FAIRTEST_VEH_W') else None
# Body length of both vehicles (unset = 5 m, the protocol default); harness v2 twin matching (e.g. 4.718 m Mustang)
VEH_L = float(os.environ['FAIRTEST_VEH_L']) if os.environ.get('FAIRTEST_VEH_L') else None
# Perception threshold of the twin's R157 CC models (unset = 1.095 m from the cutter's lane centre). Harness v2 twin
# matching: the CARLA CC perceives at 1.095 m in CARLA metres, i.e. 1.095 * 4/w in twin units under the proportional map.
CC_PERCEIVE = float(os.environ['FAIRTEST_CC_PERCEIVE']) if os.environ.get('FAIRTEST_CC_PERCEIVE') else None
if FLOOR_MARGIN is not None and FLOOR_REL is not None:
    raise RuntimeError('Set at most one of FAIRTEST_FLOOR_MARGIN and FAIRTEST_FLOOR_REL')


def _brake_cap(distance, dt=.1):
    """Largest deceleration whose jerk-limited release stays within `distance` m/s (as the native actuator)."""
    budget = max(0., distance - 1e-6)
    lo, hi = 0., 3. - 1e-6
    for _ in range(60):
        mid = (lo + hi) / 2
        if release_distance(mid, dt, 1.99) <= budget:
            lo = mid
        else:
            hi = mid
    return lo


class B7Planner(lateral.CommittedLateralPlanner):
    """Committed quintic planner; identical except the corridor audit tolerance."""

    def request(self, time, lane_y, *, y, vy, ay):
        lateral._finite((time, lane_y, y, vy, ay))
        if (not 0 <= time < self.horizon or lane_y not in (0., 4.)
                or self.last_request_time is not None and time <= self.last_request_time):
            raise ValueError('Ordered in-horizon decision and lane goal 0/4 required')
        if self.reference is not None and lane_y == self.goal:
            self.last_request_time = time
            return self._receipt(time, lane_y, 'continue_committed_goal')
        candidate = None
        for duration in (t for t in self.durations if time + t <= self.horizon):
            reference = lateral.history_reference(y, vy, ay, lane_y, duration)
            if lateral.reference_audit(reference, tolerance=CORRIDOR_TOLERANCE_M)['eligible']:
                candidate = reference
                break
        if candidate is None and self.reference is None:
            raise ValueError('No initial eligible reference; no hidden fallback')
        self.last_request_time = time
        if candidate is None:
            return self._receipt(time, lane_y, 'switch_rejected_continue_existing_tail')
        self.reference, self.goal, self.start = candidate, float(lane_y), float(time)
        return self._receipt(time, lane_y, 'new_committed_goal')


class B7Env(CausalLateralAdversarialEnv):
    def __init__(self, base, case, b1_protocol, b2_protocol, initial_state, *, target='following',
                 target_h=0., durations=(2., 3., 4., 5., 6.), frozen_planner=False, observe=False,
                 fast_native_observation=True, vehicle_width=None, floor_rel=None):
        super().__init__(base, case, b1_protocol, b2_protocol, initial_state, target_h=0.)
        self.vehicle_width = VEH_W if vehicle_width is None else float(vehicle_width)
        self.vehicle_length = VEH_L
        if floor_rel is not None and FLOOR_MARGIN is not None:
            raise RuntimeError('floor_rel and FAIRTEST_FLOOR_MARGIN are exclusive')
        self.floor_rel = FLOOR_REL if floor_rel is None else float(floor_rel)  # relative deceleration headroom
        self.aeb_ttc = None
        self.ads_dp = None
        if target.startswith('ads@'):  # B19 tunable perception-threshold ADS: ads@<d_p metres>
            self.ads_dp = float(target.split('@', 1)[1])
            target = 'ads'
        self.drl_key = None
        if target.startswith('drl_sut@'):  # B25 learned SUT variants: drl_sut@<latency ticks> or drl_sut@p<threshold m>
            self.drl_key = target.split('@', 1)[1]
            target = 'drl_sut'
        if target.startswith('following_aeb@'):  # B12 tunable AEB: following_aeb@<ttc threshold>
            self.aeb_ttc = float(target.split('@', 1)[1])
            target = 'following_aeb'
        self.cc_reaction = None
        if target.startswith('cc_r157@r'):  # B36 reaction-time margin: the CC with a reaction of <r> seconds
            self.cc_reaction = float(target.split('@r', 1)[1])
            target = 'cc_r157'
        self.cc_human = None
        if target.startswith('cc_r157@h'):  # B57: human-like reference driver <k> of the sampled population
            self.cc_human = human_population()[int(target.split('@h', 1)[1])]
            target = 'cc_r157'
        if target not in TARGETS or float(target_h) not in HORIZONS + (1.5,):
            raise ValueError('Unsupported B7 target/horizon')
        if (target in CC_REACTION or target.startswith(('ref_', 'oracle_')) or target in ('rss_sut', 'rss_lat_sut', 'bc_human', 'ads', 'cc_r157_time', 'drl_ext', 'drl_sut', 'acc_prod', 'acc_prod_aeb')) and float(target_h) != 0.:
            raise ValueError('CC driver has its own perception model; H must be 0')
        self.target_id, self.target_h = target, float(target_h)
        self.durations = tuple(float(d) for d in durations)
        self.frozen_planner = bool(frozen_planner)
        # observe=False: open-loop plans need no 108-D policy observation (whose builder only
        # accepts the B1 request set). fast_native_observation: the highway-env ego
        # observation is discarded by this adapter, so its (pandas) computation is stubbed.
        self.observe = bool(observe)
        self.fast_native_observation = bool(fast_native_observation)

    def reset(self, *, seed=None, options=None):
        obs, info = super().reset(seed=seed, options=options)
        # Parent reset installed the B2 planner; replace before the first request.
        planner = lateral.CommittedLateralPlanner if self.frozen_planner else B7Planner
        self.lateral_planner = planner(durations=self.durations)
        if self.target_id in CC_REACTION and self.cc_human is not None:  # B57: the R157 model with this driver's values
            h = self.cc_human
            self.target_control = CCDriver(h['reaction_s'], WANDER_M, -h['decel_mps2'], perceive_m=h['perceive_m'],
                                           ttc_emergency_s=h['ttc_s'], jerk_mps3=h['jerk_mps3'])
        elif self.target_id in CC_REACTION:
            self.target_control = CCDriver(CC_REACTION[self.target_id] if self.cc_reaction is None else self.cc_reaction,
                                           CC_UNLATCH.get(self.target_id, WANDER_M),
                                           CC_EMERGENCY.get(self.target_id, -7.59), aeb=self.target_id == 'cc_r157_aeb',
                                           **({} if CC_PERCEIVE is None else dict(perceive_m=CC_PERCEIVE)))
        elif self.target_id == 'following_aeb':
            self.target_control = FollowingAEB(self.base['idm_reference'], self.target_h,
                                               *(() if self.aeb_ttc is None else (self.aeb_ttc,)))
        elif self.target_id == 'ref_fsm':
            self.target_control = FSMDriver()
        elif self.target_id == 'ref_rss':
            self.target_control = RSSDriver()
        elif self.target_id == 'rss_sut':
            self.target_control = RSSSut(self.base['idm_reference'])
        elif self.target_id == 'rss_lat_sut':
            self.target_control = RSSLatSut(self.base['idm_reference'])
        elif self.target_id in ('acc_prod', 'acc_prod_aeb'):  # B29 production-like ACC (literature parameters)
            from fairtest.sut_acc import ProductionACC
            self.target_control = ProductionACC(aeb=self.target_id == 'acc_prod_aeb')
        elif self.target_id in ('drl_ext', 'drl_sut'):  # B25 learned SUT (training-time external / trained policy)
            from fairtest.b25_drl import DRLSut, ExternalTarget
            self.target_control = ExternalTarget() if self.target_id == 'drl_ext' else DRLSut(key=self.drl_key)
        elif self.target_id in ORACLE_LIMITS:
            self.target_control = OracleBrake(self.target_id.split('_', 1)[1])
        elif self.target_id == 'bc_human':
            self.target_control = BCHumanSUT()
        elif self.target_id == 'ads':
            self.target_control = PerceptionADS(self.ads_dp)
        elif self.target_id == 'cc_r157_time':
            self.target_control = CCTimeDriver()
        self.sim.cc_clock = 0.
        if self.vehicle_width is not None or self.vehicle_length is not None:  # body geometry; highway-env caches the collision pre-check radius
            for v in self.sim.inner.road.vehicles:
                if self.vehicle_width is not None:
                    v.WIDTH = self.vehicle_width
                if self.vehicle_length is not None:
                    v.LENGTH = self.vehicle_length
                v.diagonal = float(np.sqrt(v.LENGTH ** 2 + v.WIDTH ** 2))
            self.initial = [asdict(VehicleState.from_vehicle(v)) for v in self.sim.inner.road.vehicles]
            # reset quantities that depend on the body geometry (the parent computed them for 2 m wide bodies)
            self.initial_ttc = constant_velocity_ttc(*[VehicleState(**s) for s in self.initial])
            self.reward_state = RiskAccumulator(self.initial_ttc)
        if FLOOR_MARGIN is not None:
            self.speed_floor = max(15., self.initial_speed - FLOOR_MARGIN)
        elif self.floor_rel is not None:
            sut0 = math.hypot(self.initial[1]['vx'], self.initial[1]['vy'])
            self.speed_floor = min(self.initial_speed, max(15., sut0 - self.floor_rel))
        else:
            self.speed_floor = 15.
        if self.fast_native_observation:
            obs_type = self.sim.inner.observation_type
            shape = obs_type.space().shape
            dtype = obs_type.space().dtype
            obs_type.observe = lambda: np.zeros(shape, dtype=dtype)
        return obs, info

    def _generator_action(self, request):
        if self.speed_floor > 15.:
            request = max(request, -_brake_cap(float(self.sim.inner.vehicle.speed) - self.speed_floor))
        return super()._generator_action(request)

    def _observation(self):
        if self.observe:
            return super()._observation()
        return np.zeros(108, dtype=np.float32), dict(tick=self.tick)

    def _check_motion(self, row, gc, tc, pending):
        ok = self._check_motion_contract(row, gc, tc, pending)
        if self.speed_floor > 15.:
            floor_ok = float(self.sim.inner.vehicle.speed) >= self.speed_floor - self.protocol['physical_bounds']['audit_tolerance']
            row['maneuver_checks']['speed_floor'] = floor_ok
            ok = ok and floor_ok
        return ok

    def _check_motion_contract(self, row, gc, tc, pending):
        if self.target_id not in CUSTOM_LIMITS:
            return super()._check_motion(row, gc, tc, pending)
        limits = CUSTOM_LIMITS[self.target_id]
        if getattr(self, 'cc_human', None) is not None:  # B57: this driver's own braking limits
            limits = dict(acceleration=max(limits['acceleration'], self.cc_human['decel_mps2']),
                          jerk=max(limits['jerk'], self.cc_human['jerk_mps3']))
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
        target_checks = dict(acceleration=abs(ta) <= limits['acceleration']+tol,
            jerk=abs((ta-self.acceleration[1])/.1) <= limits['jerk']+tol,
            realization=abs(ta-tc['applied_mps2']) <= 1e-4,
            road=body_within_road(t), forward=t['vx'] >= -tol)
        if not all(target_checks.values()):
            if self.target_id in ('drl_ext', 'drl_sut', 'acc_prod', 'acc_prod_aeb') and all(v for k, v in target_checks.items() if k != 'jerk'):
                # learned SUT: its own jerk excursions at crawling speed (viability floor, see ref_models.project)
                # are recorded, not fatal -- the attack's validity does not depend on the SUT's actuator
                self.sut_jerk_excursions = getattr(self, 'sut_jerk_excursions', 0) + 1
            else:
                raise RuntimeError(self.target_id+' target actuator/road contract failed: '+str(target_checks))
        row.update(maneuver_checks=checks, target_checks=target_checks,
                   unimpacted_generator=g, unimpacted_target=t,
                   consumed_impacts=[None if p is None else p.tolist() for p in pending],
                   lateral_acceleration=ay, lateral_jerk=jy,
                   path_error=abs(g['y']-gc['y_reference']))
        return all(checks.values())

    def step(self, action):
        """action = (lane_goal in {0,4}, longitudinal request in [-3, 3] m/s^2)."""
        lane_goal, request = float(action[0]), float(action[1])
        if lane_goal not in (0., 4.) or not -3. <= request <= 3. or not math.isfinite(request):
            raise ValueError('B7 action must be (0|4, request in [-3,3])')
        if self.done:
            raise RuntimeError('Reset required')
        y = float(self.sim.inner.vehicle.position[1])
        if self.tick == 0:
            vy, ay = self.initial[0]['vy'], 0.
        else:
            vy = (self.ys[-1]-self.ys[-2])/.1
            ay = (self.ys[-1]-2*self.ys[-2]+self.ys[-3])/.01
        receipt = self.lateral_planner.request(self.tick*.1, lane_goal, y=y, vy=vy, ay=ay)
        receipt.update(duration=self.lateral_planner.reference.duration)
        observation, snapshot = self._observation()
        decision = dict(tick=self.tick, index=len(self.decisions), lane_goal=lane_goal,
                        requested_acceleration_mps2=request, lateral_plan=receipt)
        self.decisions.append(decision)
        self.previous_request = request
        reward = 0.
        for _ in range(5):
            pending = [None if v.impact is None else v.impact.copy() for v in self.sim.inner.road.vehicles]
            if any(v.crashed for v in self.sim.inner.road.vehicles):
                raise RuntimeError('Continuing a crashed episode')
            self.sim.cc_clock = self.tick * .1
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
                       decision_index=decision['index'], lane_goal=lane_goal, request_mps2=request)
            if (abs(float(self.sim.inner.vehicle.action['acceleration'])-gc['native_acceleration']) > 1e-8
                    or abs(float(self.sim.inner.vehicle.action['steering'])-gc['native_steering']) > 1e-8):
                raise RuntimeError('Native command mapping drift')
            self.maneuver_valid &= self._check_motion(row, gc, tc, pending)
            self.acceleration = row['acceleration'][:]
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
            agent_outcome=self.outcome, physics_ticks=self.tick, maneuver_valid=self.maneuver_valid)

    def result(self):
        row = super().result()
        row.update(agent_version='fairtest-dev-v1', target_id=self.target_id, target_h=self.target_h,
                   lateral_durations=list(self.durations), frozen_planner=self.frozen_planner,
                   corridor_tolerance_m=1e-9 if self.frozen_planner else CORRIDOR_TOLERANCE_M)
        return row
