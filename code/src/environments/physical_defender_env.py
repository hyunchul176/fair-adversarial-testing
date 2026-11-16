"""Versioned defender environment with physical failure and forward progress.

Keeps the original normalized Kinematics observation and continuous controls.
Changes reward/termination only; old checkpoints remain readable but are not
used as initialization for the new training study.
"""
from __future__ import annotations
import math
import numpy as np
from src.environments.defensive_env import DefensiveRLEnv
from src.evaluation.safety_metrics import VehicleState,constant_velocity_ttc

ENV_VERSION='physical-defender-v1'

def body_on_straight_road(vehicle,lanes_count=3,lane_width=4.):
    lateral_radius=.5*(vehicle.LENGTH*abs(math.sin(vehicle.heading))+vehicle.WIDTH*abs(math.cos(vehicle.heading)))
    lower=-lane_width/2;upper=(lanes_count-.5)*lane_width
    return bool(vehicle.on_road and vehicle.position[1]-lateral_radius>=lower-1e-9
                and vehicle.position[1]+lateral_radius<=upper+1e-9)

class PhysicalDefenderEnv(DefensiveRLEnv):
    def _make_inner_env(self):
        env=super()._make_inner_env()
        env.unwrapped.config.update(offroad_terminal=True,duration=self._duration/self._policy_freq)
        return env

    def reset(self,*,seed=None,options=None):
        obs,info=super().reset(seed=seed,options=options)
        ego=self._inner_env.unwrapped.vehicle
        self._initial_x=float(ego.position[0]);self._previous_x=self._initial_x
        info.update(environment_version=ENV_VERSION,body_onroad=body_on_straight_road(ego,self._lanes_count))
        return obs,info

    def step(self,action):
        obs,_,terminated,truncated,info=super().step(action)
        ego=self._inner_env.unwrapped.vehicle
        onroad=body_on_straight_road(ego,self._lanes_count)
        wrong_way=float(ego.velocity[0])<-.1
        collision=bool(ego.crashed)
        failure=collision or not onroad or wrong_way
        state=VehicleState.from_vehicle(ego)
        times=[constant_velocity_ttc(state,VehicleState.from_vehicle(v)) for v in ego.road.vehicles if v is not ego]
        finite=[v for v in times if v is not None];ttc=min(finite) if finite else None
        _,lateral=ego.lane.local_coordinates(ego.position)
        delta_x=float(ego.position[0])-self._previous_x;self._previous_x=float(ego.position[0])
        forward_speed=delta_x*self._policy_freq
        progress=.1*float(np.clip(forward_speed/30.,-1.,1.))
        centering=.05*max(0.,1.-abs(lateral)/2.)*max(0.,math.cos(ego.heading))
        speed_reward=.05*(1.-min(abs(forward_speed-25.)/25.,2.))
        risk=-.1*max(0.,1.-ttc/3.) if ttc is not None else 0.
        slow=-.1 if forward_speed<5. else 0.
        reward=-10. if failure else progress+centering+speed_reward+risk+slow
        terminated=bool(terminated or failure)
        truncated=bool((truncated or self._step_count>=self._duration) and not terminated)
        reason='collision' if collision else ('offroad' if not onroad else ('wrong_way' if wrong_way else ('terminated' if terminated else ('time_limit' if truncated else 'running'))))
        info.update(environment_version=ENV_VERSION,collision=collision,body_onroad=onroad,
                    wrong_way=wrong_way,termination_reason=reason,forward_speed=forward_speed,
                    progress_m=float(ego.position[0])-self._initial_x,min_traffic_ttc=ttc,
                    reward_components={'progress':progress,'centering':centering,'speed':speed_reward,'risk':risk,'slow':slow,'terminal_failure':-10. if failure else 0.})
        return obs,float(reward),terminated,truncated,info
