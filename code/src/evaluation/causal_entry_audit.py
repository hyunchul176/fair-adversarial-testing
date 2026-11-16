"""Independent physical-seconds reference/CV check for B4 uncommitted requests.

Does not call the production policy, planner, or normalized reference evaluator.
The native analyzer must separately reconstruct EntryView from archived completed
states and verify commitment/lateral receipts with the independent B2 auditor.
"""
import math
from src.evaluation.adversarial_b2_audit import candidate


def expected_uncommitted_request(view):
    g,t=view.generator,view.target
    time=view.tick/10
    chosen=None
    for duration in (2.,3.,4.,5.,6.):
        if time+duration>12.:continue
        polynomial,eligible=candidate(g.y,view.backward_vy,view.backward_ay,4.,duration)
        if eligible:
            chosen=polynomial,duration
            break
    if chosen is None:return dict(action=1,samples=0,overlap_delay_s=None)
    polynomial,duration=chosen
    gx=(abs(math.cos(g.heading))*g.length+abs(math.sin(g.heading))*g.width)/2
    tx=(abs(math.cos(t.heading))*t.length+abs(math.sin(t.heading))*t.width)/2
    gy=(abs(math.sin(g.heading))*g.length+abs(math.cos(g.heading))*g.width)/2
    ty=(abs(math.sin(t.heading))*t.length+abs(math.cos(t.heading))*t.width)/2
    for offset in range(121-view.tick):
        tau=offset/10
        y=4. if tau>=duration else float(polynomial(tau))
        if abs(y-t.y-t.vy*tau)<=gy+ty:
            gap=g.x-t.x-gx-tx+(g.vx-t.vx)*tau
            closing=t.vx-g.vx
            action=4 if closing>0 and 0<=gap<=1.5*closing else 1
            return dict(action=action,samples=offset+1,overlap_delay_s=tau)
    return dict(action=1,samples=121-view.tick,overlap_delay_s=None)
