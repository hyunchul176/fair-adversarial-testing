"""Versioned physical surrogates, independent of policy observations and rewards.

TTC assumes constant planar velocity and fixed heading for oriented rectangles.
It is a collision prediction under that assumption, not a guarantee of a crash.
"""
from __future__ import annotations
from dataclasses import dataclass
import math
from typing import Optional

METRIC_VERSION = 'fairtest-safety-v2'

@dataclass(frozen=True)
class VehicleState:
    x: float
    y: float
    vx: float
    vy: float
    heading: float
    length: float
    width: float

    def __post_init__(self):
        if not all(math.isfinite(v) for v in self.__dict__.values()):
            raise ValueError('Non-finite vehicle state')
        if self.length <= 0 or self.width <= 0:
            raise ValueError('Vehicle dimensions must be positive')

    @classmethod
    def from_vehicle(cls, vehicle):
        return cls(float(vehicle.position[0]), float(vehicle.position[1]),
                   float(vehicle.velocity[0]), float(vehicle.velocity[1]),
                   float(vehicle.heading), float(vehicle.LENGTH), float(vehicle.WIDTH))

def _axes(v):
    c,s=math.cos(v.heading),math.sin(v.heading)
    return ((c,s),(-s,c))

def _radius(v, axis):
    u,w=_axes(v)
    return v.length/2*abs(u[0]*axis[0]+u[1]*axis[1])+v.width/2*abs(w[0]*axis[0]+w[1]*axis[1])

def constant_velocity_ttc(a: VehicleState, b: VehicleState) -> Optional[float]:
    """Earliest nonnegative contact time using swept separating-axis intervals.

    None means no predicted contact; 0 means current contact/overlap.
    No finite sentinel is treated as a physical measurement.
    """
    entry,leave=0.0,math.inf
    for axis in (*_axes(a),*_axes(b)):
        pos=(b.x-a.x)*axis[0]+(b.y-a.y)*axis[1]
        vel=(b.vx-a.vx)*axis[0]+(b.vy-a.vy)*axis[1]
        radius=_radius(a,axis)+_radius(b,axis)
        if abs(vel)<1e-12:
            if abs(pos)>radius: return None
            continue
        t1,t2=(-radius-pos)/vel,(radius-pos)/vel
        entry=max(entry,min(t1,t2));leave=min(leave,max(t1,t2))
        if entry>leave: return None
    return entry if leave>=entry else None

def following_headway(follower: VehicleState, leader: VehicleState) -> Optional[float]:
    """Bumper headway along the follower heading, only with lateral overlap."""
    longitudinal,lateral=_axes(follower)
    dx,dy=leader.x-follower.x,leader.y-follower.y
    along=dx*longitudinal[0]+dy*longitudinal[1]
    side=dx*lateral[0]+dy*lateral[1]
    speed=follower.vx*longitudinal[0]+follower.vy*longitudinal[1]
    if along<=0 or speed<=0.1: return None
    if abs(side)>_radius(follower,lateral)+_radius(leader,lateral): return None
    gap=max(0.0,along-_radius(follower,longitudinal)-_radius(leader,longitudinal))
    return gap/speed

def classify_episode(collision: bool, min_ttc: Optional[float], threshold: float=1.5):
    """Classify mutually exclusive outcomes and also retain inclusive TTC events."""
    if threshold<=0 or not math.isfinite(threshold): raise ValueError('Invalid threshold')
    if min_ttc is not None and (not math.isfinite(min_ttc) or min_ttc<0):
        raise ValueError('TTC must be nonnegative finite or None')
    event=min_ttc is not None and min_ttc<threshold
    return {'collision':bool(collision),'ttc_event':event,
            'near_miss':bool(event and not collision),
            'other':bool(not collision and not event)}
