"""PPO longitudinal retention versus removal; no learned method or safety claim."""
from src.evaluation.anticipatory_nominal_control import following_command,bounded_acceleration
ARMS=('generator_anticipatory','both_anticipatory','ppo_jerk_limited','zero_jerk_limited')
def flags(arm):
    if arm not in ARMS:raise ValueError('Unregistered arm')
    return True,arm=='both_anticipatory'
def target_jerk_limited(arm):
    flags(arm)
    return arm!='generator_anticipatory'
