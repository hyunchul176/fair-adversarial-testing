"""PPO return/optimizer auditing. Synthetic-array tests do not run a simulator."""
from contextlib import contextmanager
import random
import numpy as np
import torch
from stable_baselines3 import PPO
from stable_baselines3.common.buffers import RolloutBuffer
from src.evaluation.adversarial_a2_support import digest


@contextmanager
def isolated_rng():
    py=random.getstate();np_state=np.random.get_state();torch_state=torch.get_rng_state()
    try:yield
    finally:
        random.setstate(py);np.random.set_state(np_state);torch.set_rng_state(torch_state)


def parameter_hash(model):
    import hashlib
    h=hashlib.sha256()
    for name,t in sorted(model.policy.state_dict().items()):
        h.update(name.encode());h.update(t.detach().cpu().numpy().tobytes())
    return h.hexdigest()


class AuditedRolloutBuffer(RolloutBuffer):
    def __init__(self,*args,audit_callback=None,**kwargs):
        self.audit_callback=audit_callback;super().__init__(*args,**kwargs)
        if self.gamma!=1 or self.gae_lambda!=1:raise ValueError('Frozen gamma/GAE contract')
    def compute_returns_and_advantage(self,last_values,dones):
        bootstrap=last_values.detach().cpu().numpy().reshape(-1).copy()
        super().compute_returns_and_advantage(last_values,dones)
        expected=np.empty_like(self.rewards,dtype=np.float64)
        following=bootstrap.astype(np.float64)
        for i in reversed(range(self.buffer_size)):
            live=1-np.asarray(dones,dtype=float) if i==self.buffer_size-1 else 1-self.episode_starts[i+1]
            following=self.rewards[i].astype(np.float64)+live*following
            expected[i]=following
        if not np.allclose(self.returns,expected,rtol=1e-6,atol=2e-6):raise RuntimeError('Independent undiscounted return check failed')
        if not all(np.isfinite(a).all() for a in (self.rewards,self.returns,self.advantages,self.values,self.log_probs)):
            raise RuntimeError('Non-finite rollout buffer')
        if self.audit_callback:
            self.audit_callback(dict(observations=self.observations.copy(),actions=self.actions.copy(),
                rewards=self.rewards.copy(),episode_starts=self.episode_starts.copy(),values=self.values.copy(),
                log_probs=self.log_probs.copy(),advantages=self.advantages.copy(),returns=self.returns.copy(),
                last_values=bootstrap,dones=np.asarray(dones).copy(),expected_returns=expected))


class AuditedPPO(PPO):
    def __init__(self,*args,update_callback=None,**kwargs):
        self.update_callback=update_callback;super().__init__(*args,**kwargs)
    def _excluded_save_params(self):
        # Logging closures own open output streams. Checkpoints store the policy,
        # optimizer and hyperparameters, never a live campaign writer.
        return super()._excluded_save_params()+['update_callback','rollout_buffer_kwargs']
    def train(self):
        super().train()
        metrics={k:float(v) for k,v in self.logger.name_to_value.items() if k.startswith('train/') and np.isscalar(v)}
        undefined=[]
        if 'train/explained_variance' in metrics and not np.isfinite(metrics['train/explained_variance']):
            if np.var(self.rollout_buffer.returns)!=0:raise RuntimeError('Unexpected undefined explained variance')
            undefined.append('train/explained_variance');del metrics['train/explained_variance']
        if not metrics or not all(np.isfinite(v) for v in metrics.values()):raise RuntimeError('Missing/nonfinite optimizer metrics')
        if not all(torch.isfinite(p).all().item() for p in self.policy.parameters()):raise RuntimeError('Nonfinite parameters')
        if self.update_callback:self.update_callback(dict(actor_transitions=self.num_timesteps,metrics=metrics,undefined_zero_variance_diagnostics=undefined,parameter_hash=parameter_hash(self)))
