"""B69 third-party SUT: a car-following model of Eclipse SUMO (libsumo 1.27.1, EPL-2.0 OR GPL-2.0-or-later) decides the
SUT's longitudinal acceleration in this simulator; this simulator stays the physics and the judge.

Every 0.1 s tick both vehicles are placed into a SUMO network that mirrors the road (one straight edge, two 4 m lanes;
SUMO y = -2 - y, so the SUT's lane is SUMO lane 0 and the adversary's origin lane SUMO lane 1) with moveToXY, SUMO
advances one 0.1 s step, then SUMO itself chooses the SUT's leader (vehicle.getLeader, by its own lane model) and its
car-following model returns the speed for the next step (vehicle.getFollowSpeed with the true speeds, gap and the
adversary's 3 m/s^2 braking authority). As in SUMO's own speed composition, the next speed is the smaller of that safe
speed and the free speed, min(model response to a virtual leader 1 km ahead at the set speed, v + accel * dt, set speed)
(getFollowSpeed alone does not apply the vehicle's speed cap); the SUT applies a = (v_next - v) / 0.1. The set speed is
the SUT's initial speed, as for every other SUT here. Nothing else is tuned: SUMO's default passenger vType values (accel 2.6, decel 4.5,
emergencyDecel 9.0 m/s^2, tau 1.0 s, minGap 2.5 m) with the vehicle size of this simulator (5 x 2 m).

Variants (FAIRTEST_SUMO), all with SUMO's standard lane model (the cut-in vehicle becomes the leader when SUMO maps
its centre into the SUT's lane):
  acc      carFollowModel ACC (SUMO's implementation of Milanes & Shladover 2014)
  krauss   carFollowModel Krauss (SUMO's default safe-speed model) with sigma 0 (no dawdling)
  idm      carFollowModel IDM (SUMO's implementation of Treiber et al. 2000)
SUMO's sublane model was probed and dropped: a vehicle placed across the lane line by moveToXY is not returned by
getLeader before its centre crosses, so the sublane variant behaved exactly like `acc` (probe 2026-10-04).
The applied acceleration keeps this simulator's actuator rules (never reverse; at most +2 m/s^2) and the 9 m/s^2 contract
of the `acc_prod_aeb` slot it is run in; SUMO's step-to-step changes are not jerk-limited.
"""
from __future__ import annotations

import math
import os
import tempfile

from fairtest.ref_models import _states, project
from src.environments.hierarchical_interaction_env import lane_controller

DT = .1
EDGE_LEN = 5000.
VARIANTS = dict(acc=dict(model='ACC', sublane=False), krauss=dict(model='Krauss', sublane=False),
                idm=dict(model='IDM', sublane=False))
NET = """<?xml version="1.0" encoding="UTF-8"?>
<net version="1.20" junctionCornerDetail="5" limitTurnSpeed="5.50">
    <location netOffset="0.00,0.00" convBoundary="0.00,0.00,{L:.2f},0.00" origBoundary="0.00,0.00,{L:.2f},0.00" projParameter="!"/>
    <edge id="E0" from="J0" to="J1" priority="-1">
        <lane id="E0_0" index="0" speed="60.00" length="{L:.2f}" width="4.00" shape="0.00,-6.00 {L:.2f},-6.00"/>
        <lane id="E0_1" index="1" speed="60.00" length="{L:.2f}" width="4.00" shape="0.00,-2.00 {L:.2f},-2.00"/>
    </edge>
    <junction id="J0" type="dead_end" x="0.00" y="0.00" incLanes="" intLanes="" shape="0.00,0.00 0.00,-8.00"/>
    <junction id="J1" type="dead_end" x="{L:.2f}" y="0.00" incLanes="E0_0 E0_1" intLanes="" shape="{L:.2f},-8.00 {L:.2f},0.00"/>
</net>
"""
ROUTES = """<?xml version="1.0" encoding="UTF-8"?>
<routes>
    <vType id="sut" carFollowModel="{model}" length="5.0" width="2.0" maxSpeed="60.0"{sigma}/>
    <vType id="adv" carFollowModel="Krauss" sigma="0" length="5.0" width="2.0" maxSpeed="60.0" decel="3.0" emergencyDecel="3.0"/>
    <route id="r0" edges="E0"/>
</routes>
"""
LEADER_DECEL = 3.  # the adversary's braking authority (attack contract)
_SUMO = {}


def _variant():
    name = os.environ.get('FAIRTEST_SUMO', '')
    if name not in VARIANTS:
        raise RuntimeError(f'FAIRTEST_SUMO must be one of {sorted(VARIANTS)}')
    return name


def _lib():
    """libsumo, started once per worker process (episodes reuse it with fresh vehicles)."""
    if _SUMO.get('pid') == os.getpid():
        return _SUMO['lib']
    if _SUMO:
        raise RuntimeError('libsumo was started in another process (fork after start)')
    import libsumo
    name = _variant()
    v = VARIANTS[name]
    d = tempfile.mkdtemp(prefix='b69_sumo_')
    net, rou = os.path.join(d, 'road.net.xml'), os.path.join(d, 'road.rou.xml')
    open(net, 'w', encoding='utf-8').write(NET.format(L=EDGE_LEN))
    open(rou, 'w', encoding='utf-8').write(ROUTES.format(model=v['model'], sigma=' sigma="0"' if v['model'] == 'Krauss' else ''))
    args = ['sumo', '-n', net, '-r', rou, '--step-length', str(DT), '--no-step-log', 'true', '--no-warnings', 'true',
            '--collision.action', 'none', '--collision.check-junctions', 'false', '--seed', '0', '--begin', '0',
            '--time-to-teleport', '-1']
    if v['sublane']:
        args += ['--lateral-resolution', '0.4']
    libsumo.start(args)
    _SUMO.update(pid=os.getpid(), lib=libsumo, variant=name, n=0, dir=d)
    return libsumo


def _angle(vx, vy):
    """SUMO angle (degrees from north, clockwise) of a velocity in this simulator's frame (SUMO y = -2 - y)."""
    return (90. + math.degrees(math.atan2(vy, vx))) % 360.


class SumoSUT:
    """Per-episode controller with the target-control interface of this simulator (see sut_acc.ProductionACC)."""

    def __init__(self):
        lib = _lib()
        for vid in list(lib.vehicle.getIDList()):
            lib.vehicle.remove(vid)
        _SUMO['n'] += 1
        n = _SUMO['n']
        self.ego, self.adv = f'sut{n}', f'adv{n}'
        lib.vehicle.add(self.ego, 'r0', typeID='sut', depart='now', departLane='0', departPos='100', departSpeed='0')
        lib.vehicle.add(self.adv, 'r0', typeID='adv', depart='now', departLane='1', departPos='400', departSpeed='0')
        lib.simulationStep()
        self.lib, self.cruise_set, self.clock = lib, False, 0.
        self.accel = float(lib.vehicle.getAccel(self.ego))

    def _place(self, vid, s):
        self.lib.vehicle.moveToXY(vid, 'E0', -1, s.x, -2. - s.y, _angle(s.vx, s.vy), 1)
        self.lib.vehicle.setPreviousSpeed(vid, math.hypot(s.vx, s.vy))

    def __call__(self, env, target_id, model, cruise, previous_a, actuator='viable_v1'):
        lib = self.lib
        g, t = _states(env)
        if not self.cruise_set:
            lib.vehicle.setMaxSpeed(self.ego, float(cruise))
            self.cruise_set = True
        self._place(self.adv, g)
        self._place(self.ego, t)
        lib.simulationStep()
        v = math.hypot(t.vx, t.vy)
        lead = lib.vehicle.getLeader(self.ego, 300.)
        leader = lead[0] if lead and lead[0] else None
        v_free = min(float(lib.vehicle.getFollowSpeed(self.ego, v, 1000., float(cruise), LEADER_DECEL)),
                     v + self.accel * DT, float(cruise))
        v_next, gap = v_free, None
        if leader:
            gap = float(lead[1])
            v_next = min(v_free, float(lib.vehicle.getFollowSpeed(self.ego, v, gap, math.hypot(g.vx, g.vy), LEADER_DECEL, leader)))
        request = (v_next - v) / DT
        applied = float(project(request, v, previous_a, 1e3, 9.))
        steer = .3 * lane_controller(4. - t.y, t.heading, preview_distance=20.)
        env.target.act(dict(acceleration=applied, steering=steer))
        self.clock += DT
        return applied, dict(raw_action=[applied / 5, 0.], following_mps2=request, requested_mps2=request,
                             post_shield_mps2=request, post_stop_mps2=request, previous_mps2=previous_a,
                             applied_mps2=applied, steering=steer, viability=dict(constraint_infeasible=False),
                             shield=dict(intervened=False),
                             recognition=dict(phase='follow' if leader else 'free', leader=leader is not None,
                                              gap=None if gap is None else round(gap, 3), model=f'sumo_{_SUMO["variant"]}'))
