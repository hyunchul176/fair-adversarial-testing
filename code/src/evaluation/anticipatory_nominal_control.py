"""Plain following-law engineering baseline. No learned method or safety claim.

The gap/relative-speed law reuses the existing evasive reference's coefficients,
with oriented current-state headway. It is not an IDM or MPC implementation.
"""
import math
from src.evaluation.safety_metrics import VehicleState, following_headway

ARMS = ('target_lane_hold', 'combined', 'generator_anticipatory',
        'target_anticipatory', 'both_anticipatory')

def flags(arm):
    if arm not in ARMS:raise ValueError('Unregistered arm')
    return arm in ('generator_anticipatory','both_anticipatory'), arm in ('target_anticipatory','both_anticipatory')

def following_command(ego, traffic, cruise_speed):
    ego=VehicleState(**ego)
    speed=ego.vx*math.cos(ego.heading)+ego.vy*math.sin(ego.heading)
    command=min(2.,max(-5.,.4*(cruise_speed-speed)))
    for data in traffic:
        leader=VehicleState(**data);headway=following_headway(ego,leader)
        if headway is None:continue
        gap=headway*speed
        relative=(leader.vx-ego.vx)*math.cos(ego.heading)+(leader.vy-ego.vy)*math.sin(ego.heading)
        command=min(command,.18*(gap-5.-1.8*speed)+.8*relative)
    return max(-5.,min(2.,command))

def bounded_acceleration(request, previous, dt):
    if not all(math.isfinite(x) for x in (request,previous,dt)) or dt<=0:raise ValueError('Invalid actuator state')
    return max(-5.,min(5.,max(previous-2.5*dt,min(previous+2.5*dt,request))))
