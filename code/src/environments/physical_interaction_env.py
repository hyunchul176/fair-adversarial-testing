"""Physical scenario generator, with explicit roles and shared 20D observation.

Version v1 is an environment repair, not an NDD-calibrated algorithm claim.
Default headway targets retain the hand-designed 3-2.5c curve. A separately
validated calibration curve can be supplied without changing other rewards.
"""
from __future__ import annotations
import math
import gymnasium as gym
import numpy as np
from src.environments.physical_defender_env import PhysicalDefenderEnv,body_on_straight_road
from src.evaluation.safety_metrics import VehicleState,following_headway,constant_velocity_ttc

ENV_VERSION='physical-interaction-v1'
OBS_VERSION='shared-generator-20d-v1'

def interaction_metrics(generator,target,scenario,generator_onroad=True,target_onroad=True):
    if scenario not in {'rear_end','cut_in'}:raise ValueError('Unknown scenario')
    follower,leader=(generator,target) if scenario=='rear_end' else (target,generator)
    headway=following_headway(follower,leader)
    aligned=abs(generator.heading)<=math.radians(15) and abs(target.heading)<=math.radians(15)
    valid=generator_onroad and target_onroad and aligned and headway is not None
    return {'headway':headway if valid else None,'interaction_valid':bool(valid),
            'ttc':constant_velocity_ttc(generator,target),
            'follower':'generator' if scenario=='rear_end' else 'target',
            'follower_forward_speed':follower.vx}

def shared_generator_observation(generator,target,others,scenario,c_level,previous_speed=None,dt=.1,lanes_count=3):
    """Single entry point for training and future NPC injection; no duplicate normalization."""
    a=VehicleState.from_vehicle(generator);b=VehicleState.from_vehicle(target)
    metrics=interaction_metrics(a,b,scenario,body_on_straight_road(generator,lanes_count),body_on_straight_road(target,lanes_count))
    _,lateral=generator.lane.local_coordinates(generator.position)
    acceleration=(generator.speed-previous_speed)/dt if previous_speed is not None else 0.
    candidates=[v for v in others if v is not generator and v is not target]
    nearest=min(candidates,key=lambda v:np.linalg.norm(v.position-generator.position)) if candidates else None
    distance=float(np.linalg.norm(nearest.position-generator.position)) if nearest is not None else 100.
    relative=a.vx-float(nearest.velocity[0]) if nearest is not None else 0.
    values=[a.vx/40,a.vy/40,math.sin(a.heading),math.cos(a.heading),lateral/2,acceleration/5,
            (b.x-a.x)/100,(b.y-a.y)/12,(b.vx-a.vx)/30,(b.vy-a.vy)/30,
            math.sin(b.heading),math.cos(b.heading),metrics['headway']/10 if metrics['headway'] is not None else 1.,
            metrics['ttc']/10 if metrics['ttc'] is not None else 1.,float(scenario=='rear_end'),float(scenario=='cut_in'),
            c_level/.8,float(metrics['interaction_valid']),distance/100,relative/30]
    obs=np.clip(np.asarray(values,dtype=np.float32),-1.,1.)
    if obs.shape!=(20,) or not np.isfinite(obs).all():raise ValueError('Invalid physical generator observation')
    return obs

def physical_interaction_reward(metrics,target_headway,forward_speed,action,potential_delta,failure):
    if failure:return -50.,{'terminal_failure':-50.,'tracking':0.,'interaction':0.}
    valid=metrics['interaction_valid'] and metrics['follower_forward_speed']>=15.
    tracking=math.exp(-abs(metrics['headway']-target_headway)/.5) if valid else -.2
    interaction=.25 if valid else 0.
    progress=.05*float(np.clip(forward_speed/30.,0.,1.))
    shaping=.2*potential_delta
    ttc=metrics['ttc'];risk=-.5*max(0.,1.-ttc/.5) if ttc is not None else 0.
    steering=-.02*float(action[1])**2
    parts={'tracking':tracking,'interaction':interaction,'progress':progress,'shaping':shaping,'risk':risk,'steering':steering,'terminal_failure':0.}
    return float(sum(parts.values())),parts

class PhysicalInteractionEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,scenario_type=None,c_level=None,target_curve=None,max_reset_attempts=32,**driving_kwargs):
        super().__init__()
        if scenario_type not in {None,'rear_end','cut_in'}:raise ValueError('Invalid scenario')
        if c_level is not None and not 0<=c_level<=.8:raise ValueError('c_level outside [0,.8]')
        self.fixed_scenario=scenario_type;self.fixed_c=c_level;self.max_reset_attempts=max_reset_attempts
        self.driving=PhysicalDefenderEnv(**driving_kwargs)
        self.action_space=self.driving.action_space
        self.observation_space=gym.spaces.Box(-1.,1.,shape=(20,),dtype=np.float32)
        curve=np.asarray(target_curve if target_curve is not None else [[0.,3.],[.8,1.]],dtype=float)
        if curve.ndim!=2 or curve.shape[1]!=2 or len(curve)<2 or not np.isfinite(curve).all():raise ValueError('Invalid target curve')
        if curve[0,0]!=0 or curve[-1,0]!=.8 or not np.all(np.diff(curve[:,0])>0) or not np.all(np.diff(curve[:,1])<=0) or not np.all(curve[:,1]>0):raise ValueError('Target curve must be positive and monotone on [0,.8]')
        self.curve=curve;self.target=None;self.previous_speed=None
    @property
    def inner(self):return self.driving._inner_env.unwrapped
    def _metrics(self):
        return interaction_metrics(VehicleState.from_vehicle(self.inner.vehicle),VehicleState.from_vehicle(self.target),self.scenario,
                                   body_on_straight_road(self.inner.vehicle,self.driving._lanes_count),body_on_straight_road(self.target,self.driving._lanes_count))
    def _observation(self):
        return shared_generator_observation(self.inner.vehicle,self.target,self.inner.road.vehicles,self.scenario,self.c,self.previous_speed,
                                             dt=1/self.driving._policy_freq,lanes_count=self.driving._lanes_count)
    def _potential(self):
        a=self.inner.vehicle;b=self.target
        # Signed longitudinal target gap: generator follows in rear-end and leads after cut-in.
        follower_speed=a.velocity[0] if self.scenario=='rear_end' else b.velocity[0]
        desired=self.target_headway*max(float(follower_speed),15.)+(a.LENGTH+b.LENGTH)/2
        actual=float(b.position[0]-a.position[0]) if self.scenario=='rear_end' else float(a.position[0]-b.position[0])
        return -min(abs(actual-desired)/50.+abs(float(a.position[1]-b.position[1]))/4.,4.)
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self.scenario=self.fixed_scenario or str(self.np_random.choice(['rear_end','cut_in']))
        self.c=float(self.fixed_c if self.fixed_c is not None else self.np_random.uniform(0.,.8))
        self.target_headway=float(np.interp(self.c,self.curve[:,0],self.curve[:,1]));attempted=[]
        for attempt in range(self.max_reset_attempts):
            traffic_seed=int(self.np_random.integers(0,2**31-1));attempted.append(traffic_seed)
            self.driving.reset(seed=traffic_seed);ego=self.inner.vehicle;candidates=[]
            for vehicle in self.inner.road.vehicles:
                if vehicle is ego or not body_on_straight_road(vehicle,self.driving._lanes_count):continue
                dx=float(vehicle.position[0]-ego.position[0]);lane_delta=abs(vehicle.lane_index[2]-ego.lane_index[2])
                allowed=(lane_delta==0 and 10<=dx<=80) if self.scenario=='rear_end' else (lane_delta==1 and abs(dx)<=50)
                if allowed:candidates.append(vehicle)
            if candidates:
                self.target=min(candidates,key=lambda v:abs(v.position[0]-ego.position[0]));break
        else:raise RuntimeError(f'No valid {self.scenario} scene; attempted traffic seeds={attempted}')
        self.target_id=self.inner.road.vehicles.index(self.target);self.previous_speed=None;self.previous_potential=self._potential()
        info={'environment_version':ENV_VERSION,'observation_version':OBS_VERSION,'scenario_requested':self.scenario,'scenario_actual':self.scenario,
              'c_level':self.c,'target_headway':self.target_headway,'traffic_seed':traffic_seed,'attempted_traffic_seeds':attempted,
              'target_id':self.target_id,'reset_attempts':attempt+1,**self._metrics()}
        return self._observation(),info
    def step(self,action):
        if self.target is None or self.target not in self.inner.road.vehicles:raise RuntimeError('Designated target is unavailable')
        action=np.clip(np.asarray(action,dtype=np.float32),-1,1)
        if action.shape!=(2,) or not np.isfinite(action).all():raise ValueError('Invalid action')
        old_speed=float(self.inner.vehicle.speed)
        _,_,term,trunc,info=self.driving.step(action)
        target_onroad=body_on_straight_road(self.target,self.driving._lanes_count)
        target_collision=bool(self.target.crashed)
        failure=info['collision'] or not info['body_onroad'] or info['wrong_way'] or not target_onroad or target_collision
        term=bool(term or failure);trunc=bool(trunc and not term)
        metrics=self._metrics();potential=0. if term or trunc else self._potential()
        reward,parts=physical_interaction_reward(metrics,self.target_headway,info['forward_speed'],action,.99*potential-self.previous_potential,failure)
        self.previous_potential=potential;self.previous_speed=old_speed
        info.update(environment_version=ENV_VERSION,observation_version=OBS_VERSION,scenario_requested=self.scenario,scenario_actual=self.scenario,
                    target_id=self.target_id,c_level=self.c,target_headway=self.target_headway,target_onroad=target_onroad,
                    target_collision=target_collision,reward_components=parts,**metrics)
        if target_collision and not info['collision']:info['termination_reason']='target_collision'
        elif not target_onroad and info['body_onroad']:info['termination_reason']='target_offroad'
        return self._observation(),reward,term,trunc,info
    def close(self):self.driving.close()
