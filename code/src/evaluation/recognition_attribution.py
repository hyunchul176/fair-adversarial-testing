"""Fixed-state command attribution, never advances or edits a simulated world."""
from dataclasses import asdict
import math
import numpy as np
from src.evaluation.safety_metrics import VehicleState,following_headway
from src.environments.physical_defender_shield import shield_action
from src.evaluation.stopping_viability import project


def commands(generator,target,cruise,parameters,previous,force_gate=False):
    g,t=VehicleState(**generator),VehicleState(**target)
    c,s=math.cos(t.heading),math.sin(t.heading)
    v=t.vx*c+t.vy*s;along=(g.x-t.x)*c+(g.y-t.y)*s
    delta=g.heading-t.heading
    radius=.5*(g.length*abs(math.cos(delta))+g.width*abs(math.sin(delta)))
    gap=max(0.,along-t.length/2-radius)
    follow_gate=following_headway(t,g) is not None
    idm_gap=g.x-t.x-(g.length+t.length)/2
    idm_gate=abs(g.y-t.y)<(g.width+t.width)/2 and idm_gap>0
    follow_active=(along>0 and v>.1) if force_gate else follow_gate
    idm_active=idm_gap>0 if force_gate else idm_gate
    free=float(np.clip(.4*(cruise-v),-5,2))
    follow_raw=min(free,.18*(gap-5-1.8*v)+.8*((g.vx-t.vx)*c+(g.vy-t.vy)*s)) if follow_active else free
    speed=math.hypot(t.vx,t.vy);gv=math.hypot(g.vx,g.vy);p=parameters
    idm_raw=p['a']*(1-(speed/cruise)**p['delta'])
    if idm_active:
        desired=p['standstill']+max(0.,p['headway']*speed+speed*(speed-gv)/(2*math.sqrt(p['a']*p['b'])))
        idm_raw-=p['a']*(desired/max(idm_gap,.01))**2
    result={}
    for name,raw,visible,active in [('following',follow_raw,follow_gate,follow_active),('reference_idm',idm_raw,idm_gate,idm_active)]:
        clipped=float(np.clip(raw,-5,2));request=clipped if name=='following' else 5*float(np.float32(clipped/5))
        action,shield=shield_action(t,[g],[request/5,0.])
        post=5*float(action[0]);stop=max(post,-speed/.1);viability=project(stop,speed,previous,.1,jerk=2.5)
        result[name]=dict(gate=visible,active=active,unclipped=raw,request=request,post_shield=post,post_stop=stop,
            applied=viability['acceleration'],shield_intervened=shield['intervened'],viability=viability)
    return result
