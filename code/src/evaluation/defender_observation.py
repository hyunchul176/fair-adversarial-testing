"""Reuse the exact Kinematics observation used by DefensiveHighwayEnv training."""
import numpy as np

def native_defender_observation(inner_env, observer_vehicle=None):
    """Return the native 25D observation, optionally from another vehicle.

    The optional observer is needed when a trained defender is injected as a
    non-ego vehicle. A fresh observation object avoids mutating the generator's
    observation state.
    """
    config=inner_env.config['observation']
    if config['type']!='Kinematics' or config['features']!=['x','y','vx','vy','heading'] or config['vehicles_count']!=5:
        raise ValueError('Defender observation configuration mismatch')
    if not config.get('normalize',True):raise ValueError('Defender requires normalized Kinematics')
    if observer_vehicle is None:
        raw=inner_env.observation_type.observe()
    else:
        from highway_env.envs.common.observation import observation_factory
        observation_type=observation_factory(inner_env,dict(config))
        observation_type.observer_vehicle=observer_vehicle
        raw=observation_type.observe()
    obs=np.asarray(raw,dtype=np.float32).flatten()
    if obs.shape!=(25,) or not np.isfinite(obs).all():raise ValueError('Invalid defender observation')
    return np.clip(obs,-1.,1.)
