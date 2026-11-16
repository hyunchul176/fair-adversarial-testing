"""Single pair-reset path shared by controller evaluation and PPO training."""
from src.environments.hierarchical_interaction_env import HierarchicalInteractionEnv
from src.evaluation.generator_contract import configure_pair_target
from src.evaluation.generator_measurement import digest, world_snapshot


class GeneratorContractEnv(HierarchicalInteractionEnv):
    def __init__(self, *, target_kind='idm', record_parent=False, **kwargs):
        if target_kind not in {'idm', 'fixed_speed'}:
            raise ValueError('Unknown target behavior')
        self.target_kind = target_kind
        self.record_parent = record_parent
        super().__init__(**kwargs)

    def reset(self, *, seed=None, options=None):
        _, info = super().reset(seed=seed, options=options)
        if self.record_parent:
            self.parent_world = world_snapshot(self)
            info['parent_world_hash'] = digest(self.parent_world)
        configure_pair_target(self, self.target_kind)
        info.update(target_id=self.target_id, target_kind=self.target_kind,
                    reset_contract='pair-reset-v1')
        # The original reset observation still contains background traffic.
        # Regenerate after registry/behavior changes for both train and eval.
        return self._observation(), info
