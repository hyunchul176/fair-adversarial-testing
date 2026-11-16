"""Case sampling and durable terminal capture around the frozen A1 environment.

Tests inject a fake factory; the production factory refuses local simulation.
"""
from copy import deepcopy
import os
from pathlib import Path
import gymnasium as gym
import numpy as np
from src.evaluation.adversarial_a2_support import InputCycle, read, sha
from src.evaluation.local_input_sensitivity import perturb


class A2Cases:
    def __init__(self,root,spec):
        self.root=Path(root);self.spec=spec
        for name,h in spec['cached_inputs']['hashes'].items():
            if sha(self.root/name)!=h:raise ValueError('Cached input drift: '+name)
        self.base=read(self.root/'data/base_protocol.json');self.contract=read(self.root/'data/contract.json')
        self.cases={c['case_id']:c for c in read(self.root/'data/selected_cases.json')}
        self.windows=read(self.root/'data/windows.json')
        self.inputs={i['input_id']:i for i in spec['inputs']}
    def case(self,input_id):
        i=self.inputs[input_id]
        return perturb(self.cases[i['case_id']],i['gap_delta_m'],i['speed_delta_mps'])
    def make(self,input_id):
        code=Path(__file__).resolve().parents[2]
        if not str(code).startswith(os.path.expanduser('~/ASG/')):
            raise RuntimeError('Actual simulator execution is research-server-only')
        from src.environments.online_adversarial_cutin_env import OnlineAdversarialCutInEnv
        i=self.inputs[input_id]
        return OnlineAdversarialCutInEnv(self.base,self.case(input_id),self.windows[i['event_id']],self.contract,'following')


class A2SamplingEnv(gym.Env):
    metadata={'render_modes':[]}
    def __init__(self,factory,ledger,phase,on_episode,on_partial,on_reset,*,sampler=None):
        super().__init__()
        self.factory=factory;self.ledger=ledger;self.phase=phase
        self.on_episode=on_episode;self.on_partial=on_partial;self.on_reset=on_reset
        self.sampler=sampler;self.env=None;self.saved=True;self.input_id=None
        self.action_space=gym.spaces.Discrete(4)
        self.observation_space=gym.spaces.Box(-1,1,shape=(27,),dtype=np.float32)
        self.last_obs=None;self.meta={};self.reward_sum=0.
    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self._close_current('reset')
        if options is not None and 'input_id' in options:
            if self.sampler is not None:raise ValueError('Explicit evaluations cannot consume training sampler')
            self.input_id=int(options['input_id'])
        elif self.sampler is not None:self.input_id=self.sampler.next()
        else:raise ValueError('Evaluation reset requires an explicit input ID')
        self.ledger.start(self.phase)
        self.meta=dict((options or {}).get('metadata',{}))
        self.meta.update(run_id=f"{self.phase}_{self.ledger.rows[self.phase]['starts']:06d}",
                         input_id=self.input_id,group=self.phase,policy=self.meta.get('policy','ppo'))
        self.env=self.factory.make(self.input_id);self.saved=False;self.reward_sum=0.
        obs,info=self.env.reset(seed=seed)
        self.last_obs=obs.copy()
        self.on_reset(dict(self.meta,initial_hash=info.get('initial_hash'),
                           sampler_state=self.sampler.state() if self.sampler else None))
        return obs,info
    def _physics(self):
        if self.env is None:return 0
        native=getattr(getattr(self.env,'sim',None),'inner',None)
        return int(getattr(native,'steps',getattr(self.env,'tick',0)))
    def step(self,action):
        if self.env is None or self.saved:raise RuntimeError('Reset before step')
        if not self.action_space.contains(action):raise ValueError('Discrete integer action required')
        self.ledger.reserve_action(self.phase)
        before=self._physics()
        try:obs,reward,term,trunc,info=self.env.step(int(action))
        except Exception:
            self.ledger.charge(self.phase,max(0,min(5,self._physics()-before)),failed=True)
            self._save_partial('step_exception');raise
        self.ledger.charge(self.phase,self._physics()-before)
        if trunc:raise RuntimeError('Finite A1 task must terminate, never truncate')
        if not np.array_equal(self.last_obs,np.asarray(self.env.decisions[-1]['observation'],dtype=np.float32)):
            raise RuntimeError('Policy observation differs from causal decision snapshot')
        self.reward_sum+=float(reward);self.last_obs=obs.copy()
        if term:
            row=deepcopy(self.env.result());row.update(self.meta,status='complete')
            if abs(row['agent_return']-self.reward_sum)>1e-10:raise RuntimeError('Reward mismatch')
            row['terminal_observation']=obs.tolist()
            self.saved=True;self.ledger.complete(self.phase)
            # Persist and audit now. DummyVecEnv resets immediately after return.
            self.on_episode(row)
            info=dict(info,a2_run_id=row['run_id'],a2_input_id=self.input_id)
        return obs,float(reward),bool(term),False,info
    def _save_partial(self,reason):
        if self.env is None or self.saved:return
        meter=getattr(self.env,'meter',None)
        self.on_partial(deepcopy(dict(self.meta,status='incomplete',reason=reason,
            physics_ticks=self._physics(),tick=getattr(self.env,'tick',0),
            trace=getattr(meter,'trace',[]),decisions=getattr(self.env,'decisions',[]),
            physical_initial_state=getattr(self.env,'initial',None),
            current_observation=None if self.last_obs is None else self.last_obs.tolist(),
            accumulated_return=self.reward_sum,sampler_state=self.sampler.state() if self.sampler else None)))
        self.saved=True
    def _close_current(self,reason):
        self._save_partial(reason)
        if self.env is not None:self.env.close();self.env=None
    def close(self):self._close_current('close_without_drain')
