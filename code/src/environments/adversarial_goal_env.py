"""Goal wrapper composed around the immutable A2 sampler and A1 physics.

Legacy episodes are persisted before goal processing. Goal records are separate;
they cannot silently overwrite the original reward or 27D observation evidence.
"""
from copy import deepcopy
import gymnasium as gym
import numpy as np
from src.evaluation.adversarial_goal_core import GoalAccumulator,check_goal,category


class AdversarialGoalEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,base,on_goal,on_goal_partial):
        super().__init__();self.base=base;self.on_goal=on_goal;self.on_goal_partial=on_goal_partial
        self.action_space=base.action_space
        self.observation_space=gym.spaces.Box(-1,1,shape=(28,),dtype=np.float32)
        self.active=False;self.trace=[];self.decisions=[];self.last_obs=None;self.acc=None
    def _obs(self,obs):
        result=np.concatenate((np.asarray(obs,dtype=np.float32),np.asarray([self.goal],dtype=np.float32)))
        if result.shape!=(28,) or not np.isfinite(result).all():raise ValueError('Invalid goal observation')
        return result
    def reset(self,*,seed=None,options=None):
        self.close();options=deepcopy(options or {})
        self.goal=check_goal(options.pop('goal'))
        options.setdefault('metadata',{})['goal']=self.goal
        self.trace=[];self.decisions=[];self.acc=None;self.last_obs=None
        obs,info=self.base.reset(seed=seed,options=options)
        self.acc=GoalAccumulator(self.base.env.initial_ttc,self.goal)
        self.last_obs=self._obs(obs);self.active=True
        return self.last_obs.copy(),dict(info,goal=self.goal)
    def step(self,action):
        if not self.active:raise RuntimeError('Reset before goal step')
        before=len(self.base.env.meter.trace)
        decision=dict(tick=before,action=int(action),observation=self.last_obs.tolist())
        self.decisions.append(decision)
        obs,_,term,trunc,info=self.base.step(action)
        reward=0.
        for tick,row in enumerate(self.base.env.meter.trace[before:],start=before):
            value=self.acc.advance(row['pair_ttc'],row['agent_outcome']);reward+=value
            self.trace.append(dict(tick=tick,peak=self.acc.peak,utility=self.acc.utility,
                                   reward=value,outcome=row['agent_outcome']))
        decision['reward']=reward;self.last_obs=self._obs(obs)
        if term:
            row=dict(self.base.meta,status='complete',goal=self.goal,trace=self.trace,decisions=self.decisions,
                initial_utility=self.acc.initial,final_utility=self.acc.utility,goal_return=self.acc.total,
                realized_peak=self.acc.peak,category=category(self.acc.peak,self.goal,self.base.env.outcome),
                terminal_observation=self.last_obs.tolist())
            row['success']=row['category']=='noncontact_hit'
            self.on_goal(deepcopy(row));self.active=False
        return self.last_obs.copy(),reward,term,trunc,dict(info,goal_return=self.acc.total)
    def close(self):
        if self.active:
            self.on_goal_partial(deepcopy(dict(self.base.meta,status='incomplete',goal=self.goal,
                trace=self.trace,decisions=self.decisions,current_observation=None if self.last_obs is None else self.last_obs.tolist(),
                accumulated_goal_return=None if self.acc is None else self.acc.total)))
            self.active=False
        self.base.close()
