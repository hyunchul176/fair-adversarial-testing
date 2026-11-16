"""B2 lane-goal adapter. Native construction is research-server only.

The B1 tick loop, actuator, target law, attribution and reward are unchanged.
Only reset geometry, lateral reference, and audited composite actions differ.
"""
from copy import deepcopy
import math
from numbers import Integral

import gymnasium as gym
import numpy as np

from src.environments.direct_adversarial_cutin_env import DirectAdversarialCutInEnv
from src.evaluation.adversarial_b1_core import causal_frame
from src.evaluation.adversarial_native_action import acceleration_command, steering_command
from src.evaluation.causal_lateral_reference import CommittedLateralPlanner
from src.evaluation.generator_measurement import GeneratorMeasurement, digest
from src.evaluation.safety_metrics import VehicleState
from dataclasses import asdict


def decode_action(action):
    if isinstance(action, (bool, np.bool_)) or not isinstance(action, Integral) or action not in range(6):
        raise ValueError('B2 integer composite action must be in [0,5]')
    return (0., 4.)[int(action)//3], int(action)%3


class CausalLateralAdversarialEnv(DirectAdversarialCutInEnv):
    def __init__(self, base, case, b1_protocol, protocol, initial_state, target_h=0.):
        super().__init__(base, case, b1_protocol, target_h=target_h)
        self.b2_protocol = deepcopy(protocol)
        self.archived_initial = deepcopy(initial_state)
        if len(initial_state) != 2 or protocol['action']['count'] != 6:
            raise ValueError('Frozen B2 reset/action contract required')
        self.action_space = gym.spaces.Discrete(6)

    def _reset_native(self):
        # No observed lateral profile or future event time is used in B2 reset.
        from scripts.run_braking_response_feasibility import build_env
        from src.evaluation.geometry_target_cross import configure_geometry
        g0, t0 = self.archived_initial
        config = {**self.base, 'seed': self.base['seed']+self.case['case_id']}
        sim, reset = build_env(config, t0['vx'], self.case['event']['metrics']['p10']['headway'], 0)
        configure_geometry(sim, 'adjacent', 'plain')
        if len(sim.inner.road.vehicles) != 2:
            sim.close()
            raise ValueError('B2 requires two vehicles')
        for vehicle, initial in zip(sim.inner.road.vehicles, self.archived_initial):
            vehicle.position[:] = [initial['x'], initial['y']]
            vehicle.heading = initial['heading']
            vehicle.speed = math.hypot(initial['vx'], initial['vy'])
            vehicle.on_state_update()
            actual = asdict(VehicleState.from_vehicle(vehicle))
            if any(abs(actual[k]-initial[k]) > 1e-9 for k in initial):
                sim.close()
                raise ValueError('Archived initial physical state mismatch')
        sim.previous_potential = sim._potential()
        sim.driving._initial_x = sim.driving._previous_x = g0['x']
        reset.update(scenario_executed='adversarial-B2-causal-lateral-v1', source_rehearsal=False, **sim._metrics())
        return sim, reset

    def reset(self, *, seed=None, options=None):
        # Parent reset uses its module-level setup. Invoke the same reset body
        # through a narrow overridable hook, rather than patching shared globals.
        self.lateral_planner = CommittedLateralPlanner()
        return super().reset(seed=seed, options=options)

    def _observation(self):
        if self.history.snapshots and self.history.snapshots[-1]['tick'] == self.tick:
            return self.history.current()
        g, t = [asdict(VehicleState.from_vehicle(v)) for v in self.sim.inner.road.vehicles]
        reference_y = self.initial[0]['y'] if self.tick == 0 else self.meter.trace[-1]['generator_control']['y_reference']
        raw = causal_frame(g, t, acceleration=self.acceleration, tick=self.tick,
            lateral_y=reference_y, previous_steer=self.previous_steer, maximum=self.reward_state.maximum,
            initial_speed=self.initial_speed, scalar_speed=float(self.sim.inner.vehicle.speed),
            previous_request=self.previous_request)
        return self.history.push(raw, self.tick)

    def step(self, action):
        lane_goal, longitudinal = decode_action(action)
        if self.done:
            raise RuntimeError('Reset required')
        y = float(self.sim.inner.vehicle.position[1])
        if self.tick == 0:
            vy, ay = self.initial[0]['vy'], 0.
        else:
            vy = (self.ys[-1]-self.ys[-2])/.1
            ay = (self.ys[-1]-2*self.ys[-2]+self.ys[-3])/.01
        receipt = self.lateral_planner.request(self.tick*.1, lane_goal, y=y, vy=vy, ay=ay)
        receipt.update(history_state=dict(y=y, backward_vy=vy, backward_ay=ay),
                       coefficients=list(self.lateral_planner.reference.coefficients),
                       duration=self.lateral_planner.reference.duration,
                       reset_history_synthetic=self.tick == 0)
        first = len(self.meter.trace)
        # Parent records the scalar action. Convert only the composite request
        # identity afterward; previous acceleration request remains unchanged.
        before_decisions = len(self.decisions)
        try:
            return super().step(longitudinal)
        finally:
            if len(self.decisions) > before_decisions:
                self.decisions[-1].update(action=int(action), longitudinal_action=longitudinal, lateral_plan=receipt)
            for row in self.meter.trace[first:]:
                row.update(agent_action=int(action), longitudinal_action=longitudinal)

    def _generator_action(self, request):
        g = self.sim.inner.vehicle
        y = self.lateral_planner.evaluate((self.tick+1)*.1)
        desired_vy = (y-float(g.position[1]))/.1
        beta = math.asin(float(np.clip(desired_vy/float(g.speed), -.99, .99)))-float(g.heading)
        steer_request = math.atan(2*math.tan(beta))
        steer_cmd, steer = steering_command(steer_request, self.previous_steer)
        accel_cmd, projection = acceleration_command(request, float(g.speed), self.previous_a)
        return np.asarray([accel_cmd, steer_cmd], dtype=np.float32), dict(
            y_reference=y, projection=projection, requested_mps2=request,
            requested_steering=steer_request, native_acceleration=projection['applied_mps2'],
            native_steering=steer, tape_passthrough=False, applied_lane_goal=self.lateral_planner.goal,
            reference_start=self.lateral_planner.start,
            reference_deadline=self.lateral_planner.start+self.lateral_planner.reference.duration)

    def result(self):
        row = super().result()
        row.update(agent_version='adversarial-B2-causal-lateral-v1',
                   archived_initial_state=self.archived_initial,
                   archived_initial_hash=digest(self.archived_initial),
                   lateral_contract='three_completed_positions_quintic_committed_goal',
                   native_future_lateral_reference_used=False)
        return row
