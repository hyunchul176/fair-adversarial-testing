"""B2 saved-record audit. No production lateral planner or simulator imported.

Fit coefficients in PHYSICAL seconds, independently from normalized-u production
coefficients. Reuse the B1 independent native dynamics audit after an explicit
composite-to-longitudinal action projection, then audit B2 references separately.
"""
from copy import deepcopy
import math
import numpy as np
from numpy.polynomial import Polynomial

from src.evaluation.adversarial_b1_audit import audit_episode as audit_b1, close


def candidate(y, vy, ay, goal, duration):
    past = [y-.2*vy+.01*ay, y-.1*vy, y]
    matrix = [[t**k for k in range(6)] for t in (-.2, -.1, 0., duration)]
    matrix += [[0.]+[k*duration**(k-1) for k in range(1,6)],
               [0.,0.]+[k*(k-1)*duration**(k-2) for k in range(2,6)]]
    p = Polynomial(np.linalg.solve(matrix, [*past, goal, 0., 0.]))
    def bounds(order, lower):
        q = p.deriv(order)
        roots = np.roots(q.deriv().coef[::-1]) if len(q.deriv().coef)>1 else []
        points = [lower, duration]+[float(r.real) for r in roots if abs(r.imag)<1e-7 and lower<r.real<duration]
        values = [float(q(t)) for t in points]
        return min(values), max(values)
    yb, ab, jb = bounds(0,0.), bounds(2,-.2), bounds(3,-.2)
    eligible = yb[0]>=-1e-9 and yb[1]<=4+1e-9 and max(map(abs,ab))<=1.996+1e-9 and max(map(abs,jb))<=3.92+1e-9
    return p, eligible


def audit_lateral(row):
    initial = row['physical_initial_state'][0]
    ys = [initial['y']]+[t['generator']['y'] for t in row['trace']]
    active = None
    rejected = 0
    for decision in row['decisions']:
        action = decision['action']
        assert isinstance(action,int) and not isinstance(action,bool) and action in range(6)
        assert decision['longitudinal_action'] == action%3
        tick = decision['tick']; time=tick*.1; goal=(0.,4.)[action//3]
        y=ys[tick]
        vy=initial['vy'] if tick==0 else (ys[tick]-ys[tick-1])/.1
        ay=0. if tick==0 else (ys[tick]-2*ys[tick-1]+ys[tick-2])/.01
        receipt=decision['lateral_plan']
        assert receipt['reset_history_synthetic'] == (tick==0)
        assert receipt['native_validated'] is False
        for k,v in dict(y=y,backward_vy=vy,backward_ay=ay).items():
            close(receipt['history_state'][k],v,1e-8)
        close(receipt['time'],time,0); close(receipt['requested_lane_goal'],goal,0)
        if active is not None and goal==active['goal']:
            status='continue_committed_goal'
        else:
            chosen=None
            for duration in (2.,3.,4.,5.,6.):
                if time+duration>12:continue
                polynomial,eligible=candidate(y,vy,ay,goal,duration)
                if eligible:
                    chosen=dict(p=polynomial,goal=goal,start=time,duration=duration)
                    break
            if chosen is None:
                assert active is not None,'No eligible initial reference'
                status='switch_rejected_continue_existing_tail';rejected+=1
            else:
                active=chosen;status='new_committed_goal'
        assert receipt['status']==status
        close(receipt['applied_lane_goal'],active['goal'],0)
        close(receipt['reference_start'],active['start'],0)
        close(receipt['reference_deadline'],active['start']+active['duration'],1e-12)
        close(receipt['duration'],active['duration'],0)
        coefficients=[v/active['duration']**k for k,v in enumerate(receipt['coefficients'])]
        np.testing.assert_allclose(coefficients,active['p'].coef,rtol=0,atol=2e-7)
        for step in row['trace'][tick:min(tick+5,len(row['trace']))]:
            elapsed=(step['tick']+1)*.1-active['start']
            expected=active['goal'] if elapsed>=active['duration'] else float(active['p'](elapsed))
            gc=step['generator_control']
            close(gc['y_reference'],expected,2e-8)
            close(gc['applied_lane_goal'],active['goal'],0)
            close(gc['reference_start'],active['start'],0)
            close(gc['reference_deadline'],active['start']+active['duration'],1e-12)
            assert step['longitudinal_action']==action%3 and step['agent_action']==action
    return dict(lateral_decisions=len(row['decisions']), rejected_lane_switches=rejected,
                lateral_reference_independently_audited=True)


def audit_episode(row, protocol=None, base=None, case=None):
    assert row['agent_version']=='adversarial-B2-causal-lateral-v1'
    assert row['native_future_lateral_reference_used'] is False
    for initial,archived in zip(row['physical_initial_state'],row['archived_initial_state']):
        for key in archived:close(initial[key],archived[key],1e-9)
    lateral=audit_lateral(row)
    normalized=deepcopy(row)
    for d in normalized['decisions']:d['action']%=3
    for t in normalized['trace']:t['agent_action']%=3
    audited=audit_b1(normalized,protocol,base,None)
    return dict(audited,**lateral)
