"""Causal online cut-in adversary, composed with the frozen physical simulator.

Run simulator episodes only on the research server. Old rewards are discarded;
old measurement fields remain available separately from valid adversarial outcomes.
"""
from copy import deepcopy
from dataclasses import asdict
import math

import gymnasium as gym
import numpy as np

from src.evaluation.adversarial_agent_core import OneShotDeficitPulse, RiskAccumulator, ttc_risk
from src.evaluation.adversarial_native_action import acceleration_command, steering_command, native_map
from src.evaluation.generator_measurement import GeneratorMeasurement, contact_outcome, digest
from src.evaluation.observed_longitudinal_replay import setup, lateral_reference, speed_reference
from src.evaluation.predictive_recognition import controller
from src.evaluation.safety_metrics import VehicleState, constant_velocity_ttc


VERSION = 'online-adversarial-cutin-v1'


def causal_observation(g, t, *, acceleration, tick, lateral_y, nominal_vx,
                       pulse, previous_steer, maximum, divisors):
    """Value-only interface: no actor commands, controller internals or future data."""
    ttc = constant_velocity_ttc(VehicleState(**g), VehicleState(**t))
    values = [g['x']-t['x'], g['y']-t['y'], g['vx'], g['vy'], t['vx'], t['vy'],
              math.sin(g['heading']), math.cos(g['heading']), math.sin(t['heading']), math.cos(t['heading']),
              *acceleration, float(tick > 0), tick*.1, lateral_y, nominal_vx,
              pulse.deficit, pulse.rate, pulse.goal, previous_steer,
              ttc_risk(ttc), float(ttc is not None), maximum,
              *[float(pulse.phase == phase) for phase in ('armed','active','release','done')]]
    normalized = np.asarray(values, dtype=np.float64) / np.asarray(divisors)
    if normalized.shape != (27,) or not np.isfinite(normalized).all():
        raise ValueError('Invalid causal observation')
    clipped = np.clip(normalized, -1, 1).astype(np.float32)
    clipped.setflags(write=False)
    return clipped, (np.abs(normalized) > 1).astype(int).tolist()


def unimpacted_state(vehicle, consumed_impact):
    state = asdict(VehicleState.from_vehicle(vehicle))
    if consumed_impact is not None:
        state['x'] -= float(consumed_impact[0])
        state['y'] -= float(consumed_impact[1])
    return state


def body_within_road(state):
    radius = .5 * (state['length']*abs(math.sin(state['heading'])) + state['width']*abs(math.cos(state['heading'])))
    return state['y']-radius >= -2-1e-9 and state['y']+radius <= 10+1e-9


class OnlineAdversarialCutInEnv(gym.Env):
    metadata = {'render_modes': []}

    def __init__(self, base, case, window, contract, target='following', arm='continue',
                 tape=None, reference_trace=None):
        super().__init__()
        if target not in ('following','reference_idm') or arm not in ('continue','abort'):
            raise ValueError('Unsupported fixed target/path')
        if (base['environment']['simulation_frequency'], base['environment']['policy_frequency'],
            base['environment']['duration']) != (10,10,120):
            raise ValueError('Frozen 10Hz/120-tick inner contract required')
        if contract['reward']['gamma'] != 1 or contract['reward']['horizon_bootstrap'] != 0:
            raise ValueError('Undiscounted finite task required')
        if (tape is None) != (reference_trace is None):
            raise ValueError('Tape sentinel requires its archived reference telemetry')
        self.base, self.case, self.window, self.contract = map(deepcopy, (base,case,window,contract))
        self.target_id, self.arm = target, arm
        self.tape, self.reference_trace = tape, reference_trace
        self.action_space = gym.spaces.Discrete(4)
        self.observation_space = gym.spaces.Box(-1,1,shape=(27,),dtype=np.float32)
        self.divisors = [f['divisor'] for f in contract['observation']['features']]
        self.sim = self.meter = None
        self.done = True

    def reset(self, *, seed=None, options=None):
        super().reset(seed=seed)
        self.close()
        self.sim, reset = setup(self.base, self.case, self.arm, source=False)
        self.meter = GeneratorMeasurement(self.sim, reset)
        if any(self.meter.reset_invalid.values()):
            self.close()
            raise ValueError('Invalid registered reset')
        self.initial = [asdict(VehicleState.from_vehicle(v)) for v in self.sim.inner.road.vehicles]
        self.initial_ttc = constant_velocity_ttc(*[VehicleState(**s) for s in self.initial])
        self.reward_state = RiskAccumulator(self.initial_ttc)
        self.pulse = OneShotDeficitPulse()
        self.target_control = controller(self.base['idm_reference'],0.)
        self.cruise = self.case['event']['metrics']['p10']['target_speed']
        self.tick = 0
        self.previous_a = self.previous_steer = self.previous_target = 0.
        self.acceleration = [0.,0.]
        self.ys = [self.initial[0]['y']]
        self.done = False
        self.maneuver_valid = True
        self.outcome = None
        self.last_info = reset
        self.decisions = []
        self.clip_counts = np.zeros(27,dtype=int)
        obs = self._observation()
        return obs, {'initial_hash':self.meter.initial_hash, 'version':VERSION,
                     'initial_ttc':self.initial_ttc, 'reset_seed_contract':'fixed archived case seed'}

    def _observation(self):
        g,t = [asdict(VehicleState.from_vehicle(v)) for v in self.sim.inner.road.vehicles]
        obs, clips = causal_observation(g,t,acceleration=self.acceleration,tick=self.tick,
            lateral_y=lateral_reference(self.base,self.case,self.arm,self.tick*.1),
            nominal_vx=speed_reference(self.case,self.window,'observed_continuation',self.tick*.1),
            pulse=self.pulse.state,previous_steer=self.previous_steer,
            maximum=self.reward_state.maximum,divisors=self.divisors)
        self.clip_counts += np.asarray(clips)
        return obs

    def _generator_action(self):
        g = self.sim.inner.vehicle
        if self.tape is not None:
            a = np.asarray(self.tape[self.tick],dtype=np.float32)
            control = deepcopy(self.reference_trace[self.tick]['generator_control'])
            control.update(native_acceleration=native_map(a[0],5), native_steering=native_map(a[1],.3), tape_passthrough=True)
            return a, control
        y = lateral_reference(self.base,self.case,self.arm,(self.tick+1)*.1)
        desired_vy = (y-float(g.position[1]))/.1
        beta = math.asin(float(np.clip(desired_vy/float(g.speed),-.99,.99)))-float(g.heading)
        steer_cmd, steer = steering_command(math.atan(2*math.tan(beta)),self.previous_steer)
        slip = math.atan(.5*math.tan(steer))
        next_heading = float(g.heading)+float(g.speed)*math.sin(slip)/(g.LENGTH/2)*.1
        vx = speed_reference(self.case,self.window,'observed_continuation',(self.tick+1)*.1)-self.pulse.state.deficit
        request = (vx/math.cos(next_heading)-float(g.speed))/.1
        accel_cmd, projection = acceleration_command(request,float(g.speed),self.previous_a)
        action = np.asarray([accel_cmd,steer_cmd],dtype=np.float32)
        return action, dict(vx_reference=vx,y_reference=y,projection=projection,
            requested_mps2=request,native_acceleration=projection['applied_mps2'],native_steering=steer,tape_passthrough=False)

    def _check_motion(self, row, action, gc, tc, pending):
        tol = self.base['generator_bounds']['tolerance']
        g,t = [unimpacted_state(v,p) for v,p in zip(self.sim.inner.road.vehicles,pending)]
        a,ta = row['acceleration']
        self.ys.append(g['y'])
        ay = (self.ys[-1]-2*self.ys[-2]+self.ys[-3])/.01 if len(self.ys)>=3 else 0.
        jy = (self.ys[-1]-3*self.ys[-2]+3*self.ys[-3]-self.ys[-4])/.001 if len(self.ys)>=4 else 0.
        checks = dict(
            acceleration=abs(a)<=3+tol, jerk=abs((a-self.acceleration[0])/.1)<=2+tol,
            speed=15-tol<=float(self.sim.inner.vehicle.speed)<=40+tol,
            realization=abs(a-gc['native_acceleration'])<=tol,
            steering=abs(gc['native_steering'])<=.3+tol,
            steering_slew=abs((gc['native_steering']-self.previous_steer)/.1)<=.15+tol,
            path=abs(g['y']-gc['y_reference'])<=1e-5,
            speed_tracking=abs(g['vx']-gc['vx_reference'])<=.5,
            lateral_acceleration=abs(ay)<=2+tol,lateral_jerk=abs(jy)<=4+tol,
            road=body_within_road(g),forward=g['vx']>=-tol)
        target_checks = dict(acceleration=abs(ta)<=5+tol,
            jerk=abs((ta-self.acceleration[1])/.1)<=2.5+tol,
            realization=abs(ta-tc['applied_mps2'])<=tol,road=body_within_road(t),forward=t['vx']>=-tol)
        if not all(target_checks.values()):
            raise RuntimeError('Target actuator/road contract failed: '+str(target_checks))
        row.update(maneuver_checks=checks,target_checks=target_checks,
            unimpacted_generator=g,unimpacted_target=t,consumed_impacts=[None if p is None else p.tolist() for p in pending],
            lateral_acceleration=ay,lateral_jerk=jy,
            speed_tracking_error=abs(g['vx']-gc['vx_reference']),path_error=abs(g['y']-gc['y_reference']))
        return all(checks.values())

    def step(self, action):
        if self.done:
            raise RuntimeError('Reset required after finite episode end')
        # Freeze the actual policy input before ANY target.act call.
        observation = self._observation()
        self.pulse.request(action)
        action = int(action)
        decision = dict(index=len(self.decisions),tick=self.tick,observation=observation.tolist(),
                        observation_hash=digest(observation),action=action)
        self.decisions.append(decision)
        reward = 0.
        for _ in range(5):
            self.pulse.advance()
            pending = [None if v.impact is None else v.impact.copy() for v in self.sim.inner.road.vehicles]
            if any(v.crashed for v in self.sim.inner.road.vehicles):
                raise RuntimeError('Continuing a crashed episode')
            self.previous_target, tc = self.target_control(self.sim,self.target_id,None,self.cruise,self.previous_target,'viable_v1')
            applied, gc = self._generator_action()
            self.meter.before_step()
            _,_,term,trunc,info = self.sim.step(applied)
            info.update(high_level_action=applied.tolist(),applied_action=applied.tolist(),
                        shield_intervened=False,low_speed_stop_guard_intervened=False)
            self.meter.after_step(info)
            row = self.meter.trace[-1]
            row.update(target_control=tc,generator_control=gc,generator_scalar_speed=float(self.sim.inner.vehicle.speed),
                       agent_action=action,decision_index=decision['index'],pulse=asdict(self.pulse.state))
            # Assert the real pinned simulator agrees with our native mapping.
            if abs(float(self.sim.inner.vehicle.action['acceleration'])-gc['native_acceleration'])>1e-8 or abs(float(self.sim.inner.vehicle.action['steering'])-gc['native_steering'])>1e-8:
                raise RuntimeError('Native command mapping drift')
            self.maneuver_valid &= self._check_motion(row,applied,gc,tc,pending)
            self.acceleration = row['acceleration'][:]
            self.previous_a = gc['native_acceleration']
            self.previous_steer = gc['native_steering']
            contacts = contact_outcome(self.meter.recorder.events,self.tick,self.meter.pair_ids,
                                      [i for i,v in enumerate(self.sim.inner.road.vehicles) if v.crashed])
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
            r = self.reward_state.advance(row['pair_ttc'],outcome)
            row.update(agent_reward=r,agent_outcome=outcome)
            reward += r
            self.last_info = info
            if outcome != 'ongoing':
                self.done, self.outcome = True,outcome
                break
        # Horizon is part of the finite task: no TimeLimit truncation/bootstrap.
        return self._observation(), reward, self.done, False, {'agent_outcome':self.outcome,
            'physics_ticks':self.tick,'maneuver_valid':self.maneuver_valid,'agent_return':self.reward_state.total}

    def result(self):
        if not self.done or self.meter is None:
            raise RuntimeError('Only completed episodes have a result')
        row = self.meter.finish(self.last_info)
        valid = self.maneuver_valid
        risk = bool(valid and (row['pair_contact'] or row['exclusive_ttc_event']))
        initial_event = self.initial_ttc is not None and self.initial_ttc < 1.5
        row.update(agent_version=VERSION,agent_outcome=self.outcome,maneuver_valid_precontact=valid,
                   valid_pair_contact=bool(valid and row['pair_contact']),valid_risk=risk,
                   initial_risk_event=initial_event,new_valid_risk=bool(risk and not initial_event),
                   agent_return=self.reward_state.total,initial_risk_score=self.reward_state.initial,
                   physical_initial_state=self.initial,physical_initial_hash=digest(self.initial),
                   decisions=self.decisions,observation_clip_counts=self.clip_counts.tolist(),
                   tape=[t['applied_action'] for t in row['trace']],
                   failed_maneuver_checks=sorted({k for t in row['trace'] for k,v in t['maneuver_checks'].items() if not v}),
                   case_id=self.case['case_id'],event_id=self.case['event']['event_id'],target_id=self.target_id,arm=self.arm)
        return row

    def close(self):
        if self.meter is not None:
            self.meter.close()
            self.meter = None
        if self.sim is not None:
            self.sim.close()
            self.sim = None
        self.done = True
