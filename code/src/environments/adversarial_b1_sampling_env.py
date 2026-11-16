"""Durable Family B sampling. Native execution is confined to the research server.

Unit tests inject a fake factory and never instantiate a vehicle environment.
"""
from copy import deepcopy
from pathlib import Path

import gymnasium as gym
import numpy as np

from src.evaluation.adversarial_a2_support import InputCycle, read, sha
from src.evaluation.adversarial_b1_runtime import assert_native_location
from src.evaluation.local_input_sensitivity import perturb


class B1Cases:
    def __init__(self, data, protocol):
        self.data=Path(data)
        self.protocol=protocol
        self.base=read(self.data/'base_protocol.json')
        self.cases={r['case_id']:r for r in read(self.data/'selected_cases.json')}
        self.inputs={r['input_id']:r for r in protocol['inputs']}
        for name,expected in protocol['data']['hashes'].items():
            if sha(self.data/name)!=expected:
                raise ValueError('Frozen B input drift: '+name)

    def case(self, input_id):
        item=self.inputs[input_id]
        return perturb(self.cases[item['case_id']],item['gap_delta_m'],item['speed_delta_mps'])

    def make(self, condition, mode):
        assert_native_location()
        from src.environments.direct_adversarial_cutin_env import DirectAdversarialCutInEnv
        input_id,target_h=condition//2,(0.,.5)[condition%2]
        return DirectAdversarialCutInEnv(self.base,self.case(input_id),self.protocol,
                                        target_h=target_h,observation_mode=mode)


class B1SamplingEnv(gym.Env):
    metadata={'render_modes':[]}

    def __init__(self,factory,ledger,phase,on_episode,on_partial,on_reset,*,
                 mode='reactive',sampler=None,run_key=None,metadata=None):
        super().__init__()
        self.factory,self.ledger,self.phase=factory,ledger,phase
        self.on_episode,self.on_partial,self.on_reset=on_episode,on_partial,on_reset
        self.mode,self.sampler,self.run_key=mode,sampler,run_key
        self.fixed_metadata=dict(metadata or {})
        self.action_space=gym.spaces.Discrete(3)
        self.observation_space=gym.spaces.Box(-1,1,(108,),dtype=np.float32)
        self.env=None
        self.saved=True
        self.last_obs=None
        self.reward_sum=0.
        self.meta={}
        self.local_starts=0

    def reset(self,*,seed=None,options=None):
        super().reset(seed=seed)
        self._close_current('reset_without_terminal')
        if options is not None and 'condition_id' in options:
            if self.sampler is not None:
                raise ValueError('Training sampler cannot accept selected conditions')
            condition=int(options['condition_id'])
        elif self.sampler is not None:
            condition=int(self.sampler.next())
        else:
            raise ValueError('Explicit condition or sampler required')
        if condition not in range(32):
            raise ValueError('Condition outside registered32')
        self.local_starts+=1
        meta={**self.fixed_metadata,**(options or {}).get('metadata',{})}
        label=meta.get('checkpoint','train')
        run_id=f'{self.phase}_{meta.get("method",self.mode)}_{meta.get("seed",0)}_{label}_{self.local_starts:06d}'
        self.meta=dict(meta,run_id=run_id,phase=self.phase,group=self.phase,
                       condition_id=condition,input_id=condition//2,H=(0.,.5)[condition%2],
                       target_h=(0.,.5)[condition%2],observation_mode=self.mode)
        self.ledger.start(self.phase,run_id,condition=self.meta,run_key=self.run_key,
                          fixed_episode=self.phase!='training')
        self.saved=False
        self.reward_sum=0.
        self.last_obs=None
        try:
            self.env=self.factory.make(condition,self.mode)
            obs,info=self.env.reset(seed=seed)
            self.last_obs=obs.copy()
            self.on_reset(dict(self.meta,initial_hash=info.get('initial_hash'),
                               observation=obs.tolist(),sampler_state=self.sampler.state() if self.sampler else None))
            return obs,info
        except BaseException:
            self._save_partial('reset_exception',failed=True)
            raise

    def _physics(self):
        if self.env is None:return 0
        native=getattr(getattr(self.env,'sim',None),'inner',None)
        return int(getattr(native,'steps',getattr(self.env,'tick',0)))

    def step(self,action):
        if self.env is None or self.saved:raise RuntimeError('Reset before step')
        if not self.action_space.contains(action):raise ValueError('Discrete integer action required')
        run_id=self.meta['run_id']
        if self.phase=='training':self.ledger.reserve_action(run_id)
        before=self._physics()
        try:
            obs,reward,term,trunc,info=self.env.step(int(action))
        except BaseException:
            try:self.ledger.charge(run_id,max(0,min(5,self._physics()-before)),failed=True)
            finally:self._save_partial('step_exception',failed=True)
            raise
        try:
            self.ledger.charge(run_id,self._physics()-before)
            self.reward_sum+=float(reward)
            if trunc:raise RuntimeError('B finite task terminates; it does not truncate')
            if not np.array_equal(self.last_obs,np.asarray(self.env.decisions[-1]['observation'],dtype=np.float32)):
                raise RuntimeError('Supplied actor observation differs from decision snapshot')
            self.last_obs=obs.copy()
            if term:
                row=deepcopy(self.env.result())
                row.update(self.meta,status='complete',terminal_observation=obs.tolist())
                if abs(row['agent_return']-self.reward_sum)>1e-10:
                    raise RuntimeError('B reward sum mismatch')
                # Store and audit before vector auto-reset. Failed audit remains
                # failed, with its completed raw trajectory retained.
                self.on_episode(row)
                self.ledger.complete(run_id)
                self.saved=True
                info=dict(info,b1_run_id=run_id,b1_condition_id=self.meta['condition_id'])
            return obs,float(reward),bool(term),False,info
        except BaseException:
            self._save_partial('poststep_accounting_or_audit_exception',failed=True)
            raise

    def _save_partial(self,reason,failed=False):
        if self.saved:return
        meter=getattr(self.env,'meter',None)
        history=getattr(self.env,'history',None)
        case=getattr(self.env,'case',{})
        reward_state=getattr(self.env,'reward_state',None)
        row=deepcopy(dict(self.meta,status='failed' if failed else 'incomplete',reason=reason,
          steps=self._physics(),tick=getattr(self.env,'tick',0),trace=getattr(meter,'trace',[]),
          decisions=getattr(self.env,'decisions',[]),physical_initial_state=getattr(self.env,'initial',None),
          case_id=case.get('case_id'),event_id=case.get('event',{}).get('event_id'),
          observation_snapshots=getattr(history,'snapshots',[]),
          observation_clip_counts=None if history is None else history.clip_counts.tolist(),
          raw_observation_clip_counts=None if history is None else history.raw_clip_counts.tolist(),
          initial_risk_score=getattr(reward_state,'initial',None),
          agent_return=self.reward_sum,
          current_observation=None if self.last_obs is None else self.last_obs.tolist(),
          accumulated_return=self.reward_sum,sampler_state=self.sampler.state() if self.sampler else None))
        # Prevent close/reset from retrying a failed archive callback. Close the
        # accounting state first; preserve its error even if archival also fails.
        self.saved=True
        ledger_error=None
        try:
            if failed:self.ledger.fail(self.meta['run_id'],reason)
            else:self.ledger.partial(self.meta['run_id'],reason)
        except BaseException as exc:ledger_error=exc
        try:self.on_partial(row)
        except BaseException:
            if ledger_error is not None:raise ledger_error
            raise
        if ledger_error is not None:raise ledger_error

    def _close_current(self,reason):
        error=None
        try:self._save_partial(reason)
        except BaseException as exc:error=exc
        try:
            if self.env is not None:self.env.close()
        except BaseException:
            if error is not None:raise error
            raise
        finally:self.env=None
        if error is not None:raise error

    def close(self):self._close_current('budget_end_without_drain')
