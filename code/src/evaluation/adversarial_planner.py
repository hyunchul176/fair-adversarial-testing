"""Revision2 pure causal predictor/planner. No environment, controller or data access.

Candidate geometry is an approximation, never a native validity certificate.
Array columns: generator x/y/vx/vy/heading, target x/y/vx/vy/heading.
"""
from dataclasses import asdict
import math
import time
import numpy as np
from src.evaluation.adversarial_agent_core import OneShotDeficitPulse, project_bounded_rate
from src.evaluation.safety_metrics import VehicleState, constant_velocity_ttc

DIVISORS=np.array([100,4,40,10,40,10,1,1,1,1,3,5,1,12,4,40,2,1,2,.3,1,1,1,1,1,1,1],dtype=float)
DT=.1


def decode(observation,tick):
    o=np.asarray(observation,dtype=np.float32)
    if o.shape!=(28,) or not np.isfinite(o).all():raise ValueError('Bad/nonfinite observation')
    if np.any(np.abs(o[:27])>1) or float(o[27]) not in(.25,.5,.75):raise ValueError('Observation domain')
    if tick not in range(0,120,5) or abs(float(o[13])*12-tick*.1)>1e-6:raise ValueError('Observation clock mismatch')
    if list(o[23:27]).count(1)!=1 or any(x not in(0,1) for x in o[23:27]):raise ValueError('Bad pulse phase')
    if o[12] != float(tick>0):raise ValueError('Bad history-valid flag')
    v=o[:27].astype(float)*DIVISORS
    if v[2]<0 or v[4]<0:raise ValueError('Negative forward velocity')
    gh=math.atan2(v[6],v[7]);th=math.atan2(v[8],v[9])
    state=np.array([v[0],v[1],v[2],v[3],gh,0,0,v[4],v[5],th])
    sg=math.hypot(v[2],v[3]);ag=v[10] if tick else 0.
    domain=None;snaps=[]
    if sg<15 or sg>40:
        if abs(sg-min(40,max(15,sg)))<=1e-5:
            snaps.append(dict(field='scalar_speed',before=sg,after=min(40,max(15,sg))));sg=min(40,max(15,sg))
        else:domain='scalar_speed_domain'
    if abs(ag)>3:
        if abs(ag)<=3+1e-6:
            snaps.append(dict(field='acceleration',before=ag,after=min(3,max(-3,ag))));ag=min(3,max(-3,ag))
        else:domain='acceleration_domain'
    q=risk_scalar(state)
    return dict(state=state,speed=sg,acceleration=ag,target_ax=v[11]*math.cos(th) if tick else 0.,
                peak=max(v[22],v[20],q),nominal=v[15],goal=float(o[27]),domain=domain,snaps=snaps,
                q_reconstruction_residual=q-v[20])


def risk_scalar(state):
    a=VehicleState(*state[:5],5.,2.)
    b=VehicleState(*state[5:7],*state[7:9],state[9],5.,2.)
    t=constant_velocity_ttc(a,b)
    return 0. if t is None else max(0.,min(1.,1-t/3))


def scalar_ttc(state):
    return constant_velocity_ttc(VehicleState(*state[:5],5.,2.),VehicleState(*state[5:],5.,2.))


def ttc_batch(s):
    """Vectorized frozen-heading oriented-box TTC; NaN represents explicit None."""
    n=len(s);entry=np.zeros(n);leave=np.full(n,np.inf);possible=np.ones(n,dtype=bool)
    c1=np.cos(s[:,4]);s1=np.sin(s[:,4]);c2=np.cos(s[:,9]);s2=np.sin(s[:,9])
    for ax,ay in ((c1,s1),(-s1,c1),(c2,s2),(-s2,c2)):
        pos=(s[:,5]-s[:,0])*ax+(s[:,6]-s[:,1])*ay
        vel=(s[:,7]-s[:,2])*ax+(s[:,8]-s[:,3])*ay
        radius=2.5*(abs(c1*ax+s1*ay)+abs(c2*ax+s2*ay))+abs(-s1*ax+c1*ay)+abs(-s2*ax+c2*ay)
        moving=abs(vel)>=1e-12
        possible &= moving | (abs(pos)<=radius)
        den=np.where(moving,vel,1.)
        t1=(-radius-pos)/den;t2=(radius-pos)/den
        entry=np.maximum(entry,np.where(moving,np.minimum(t1,t2),-np.inf))
        leave=np.minimum(leave,np.where(moving,np.maximum(t1,t2),np.inf))
        possible &= entry<=leave
    return np.where(possible & (leave>=entry),entry,np.nan)


def projected_speed(request,value,previous):
    """Vector form of exact project_bounded_rate, with explicit empty-interval mask."""
    def cap(distance):
        lo=np.zeros_like(value);hi=np.full_like(value,3.)
        def release(r):
            n=np.ceil(r/.2)
            return .1*(n*r-.2*n*(n-1)/2)
        free=release(hi)<=distance
        if free.all():return hi
        for _ in range(64):
            mid=(lo+hi)/2;ok=release(mid)<=distance
            lo=np.where(ok,mid,lo);hi=np.where(ok,hi,mid)
        return np.where(free,3.,lo)
    lo=np.maximum.reduce([np.full_like(value,-3.),previous-.2,-cap(value-15)])
    hi=np.minimum.reduce([np.full_like(value,3.),previous+.2,cap(40-value)])
    bad=lo>hi+1e-12
    rate=np.minimum(hi,np.maximum(lo,request))
    return value+rate*.1,rate,bad


def forecast(observation,deficits,tick,scalar=False):
    """Forecast all supplied planned pulse continuations; never read an actual world.

    scalar=True uses independent scalar TTC and speed projection as a reference.
    The geometric/integration order is identical to the frozen protocol.
    """
    d=decode(observation,tick);deficits=np.asarray(deficits,dtype=float)
    if deficits.ndim!=2 or deficits.shape[1]!=120-tick or not np.isfinite(deficits).all():raise ValueError('Bad deficit continuation')
    n=len(deficits);state=np.tile(d['state'],(n,1));sg=np.full(n,d['speed']);ag=np.full(n,d['acceleration'])
    at=np.full(n,d['target_ax']);peak=np.full(n,d['peak']);contact=np.zeros(n,dtype=bool)
    errors=[d['domain'] for _ in range(n)];end=np.full(n,-1,dtype=int)
    active=np.full(n,d['domain'] is None,dtype=bool)
    history=np.empty((n,121-tick,10));history[:,0]=state
    peaks=np.empty((n,121-tick));peaks[:,0]=peak;evaluated=0
    def ttc(s):
        if not scalar:return ttc_batch(s)
        out=[scalar_ttc(x) for x in s]
        return np.array([np.nan if v is None else v for v in out])
    for j in range(120-tick):
        ids=np.flatnonzero(active)
        if not len(ids):
            history[:,j+1:]=state[:,None,:];peaks[:,j+1:]=peak[:,None];break
        evaluated+=len(ids);s=state[ids].copy();old=s.copy()
        r=s[:,1]-s[:,6];w=s[:,3]-s[:,8]
        with np.errstate(divide='ignore',invalid='ignore'):tau=np.where(r*w<0,-r/w,np.inf)
        tau=np.where(r==0,0.,tau);joined=(tau>=0)&(tau<=DT)
        split=np.where(joined,tau,DT);nextvy=np.where(joined,s[:,8],s[:,3])
        desired=np.hypot(d['nominal']-deficits[ids,j],nextvy)
        if scalar:
            ss=[];aa=[];bb=[]
            for request,v,a in zip((desired-sg[ids])/DT,sg[ids],ag[ids]):
                try:
                    p=project_bounded_rate(float(request),float(v),float(a),lower=15,upper=40,rate_limit=3,slew_limit=2,dt=DT)
                    ss.append(p['next_value']);aa.append(p['rate']);bb.append(False)
                except ValueError as exc:
                    if str(exc)!='No joint position/rate/slew feasible command':raise
                    ss.append(v);aa.append(a);bb.append(True)
            sn,an,bad=np.array(ss),np.array(aa),np.array(bb)
        else:sn,an,bad=projected_speed((desired-sg[ids])/DT,sg[ids],ag[ids])
        lateral_bad=sn<abs(nextvy)
        for idx in ids[bad]:errors[idx]='empty_speed_rate_slew_interval'
        for idx in ids[~bad&lateral_bad]:errors[idx]='lateral_speed_domain'
        okay=~(bad|lateral_bad)
        active[ids[~okay]]=False;end[ids[~okay]]=j
        # Geometry follows old velocities, split only at the lateral clamp.
        seg=old.copy();seg[joined & (split==0),3]=seg[joined & (split==0),8]
        t=ttc(seg);hit=(split>0)&np.isfinite(t)&(t<=split)
        mid=old.copy();mid[:,0]+=old[:,2]*split;mid[:,5]+=old[:,7]*split
        mid[:,1]+=old[:,3]*split;mid[:,6]+=old[:,8]*split;mid[:,3]=np.where(joined,old[:,8],old[:,3])
        remainder=DT-split;t=ttc(mid)
        hit|=(remainder>0)&np.isfinite(t)&(t<=remainder)
        s[:,0]+=old[:,2]*DT;s[:,5]+=old[:,7]*DT
        s[:,6]+=old[:,8]*DT;s[:,1]=old[:,1]+old[:,3]*split+nextvy*remainder
        atn=np.sign(at[ids])*np.maximum(0,abs(at[ids])-.25)
        s[:,7]=np.maximum(0,old[:,7]+atn*DT);s[:,3]=nextvy
        s[:,2]=np.sqrt(np.maximum(0,sn**2-nextvy**2))
        for vxi,vyi,hi in ((2,3,4),(7,8,9)):
            s[:,hi]=np.where((s[:,vxi]==0)&(s[:,vyi]==0),s[:,hi],np.arctan2(s[:,vyi],s[:,vxi]))
        t=ttc(s);hit|=np.isfinite(t)&(t==0)
        q=np.where(np.isfinite(t),np.clip(1-t/3,0,1),0.)
        idx=ids[okay];state[idx]=s[okay];sg[idx]=sn[okay];ag[idx]=an[okay];at[idx]=atn[okay]
        peak[idx]=np.maximum(peak[idx],q[okay]);contact[idx]|=hit[okay]
        stopped=idx[hit[okay]];end[stopped]=j+1;active[stopped]=False
        history[:,j+1]=state;peaks[:,j+1]=peak
    if not np.isfinite(history).all() or not np.isfinite(peak).all():raise ValueError('Nonfinite model output')
    utilities=[None if errors[i] else -1. if contact[i] else 1-abs(peak[i]-d['goal']) for i in range(n)]
    return dict(peak=peak,contact=contact,utility=utilities,errors=errors,end=end,states=history,peaks=peaks,
                evaluated_steps=evaluated,snaps=d['snaps'],q_reconstruction_residual=d['q_reconstruction_residual'])


class PulseBank:
    def __init__(self,library):
        self.ids=np.array([r['schedule_id'] for r in library]);self.actions=np.array([r['actions'] for r in library],dtype=int)
        if not np.array_equal(self.ids,np.arange(953)) or self.actions.shape!=(953,24):raise ValueError('Canonical library mismatch')
        self.deficits=np.empty((953,120));self.pulse_math_ticks=0
        for i,actions in enumerate(self.actions):
            p=OneShotDeficitPulse()
            for t in range(120):
                if t%5==0:p.request(int(actions[t//5]))
                self.deficits[i,t]=p.advance().deficit;self.pulse_math_ticks+=1


class Planner:
    def __init__(self,bank,mode,reserve=lambda *args:None,record=lambda row:None):
        if mode not in('initial_context','online_replan'):raise ValueError('Unknown planner mode')
        self.bank=bank;self.mode=mode;self.reserve=reserve;self.record=record
        self.prefix=[];self.pulse=OneShotDeficitPulse();self.selected=None
        self.candidates=bank.ids.copy();self.fallbacks=0
    def action(self,observation=None):
        k=len(self.prefix)
        if k>=24:raise RuntimeError('Planner horizon exhausted')
        planning=k==0 or (self.mode=='online_replan' and len(self.candidates)>1)
        if planning:
            if observation is None:raise ValueError('Planning needs observation')
            d=decode(observation,5*k);o=np.asarray(observation)
            ps=self.pulse.state;phase=('armed','active','release','done').index(ps.phase)
            if max(abs(float(o[16])*2-ps.deficit),abs(float(o[17])-ps.rate),abs(float(o[18])*2-ps.goal))>1e-6 or o[23+phase]!=1:
                raise ValueError('Pulse history/observation mismatch')
            ids=self.candidates.copy();horizon=120-5*k
            self.reserve(len(ids)*horizon,len(ids));started=time.monotonic()
            f=forecast(observation,self.bank.deficits[ids,5*k:],5*k)
            finite=[v for v in f['utility'] if v is not None];fallback=not finite
            best=max(finite) if finite else None
            j=next(i for i,v in enumerate(f['utility']) if best is None or (v is not None and v>=best-1e-12))
            self.selected=int(ids[j]);self.fallbacks+=int(fallback)
            end=int(f['end'][j]);count=horizon if end<0 else end
            self.record(dict(decision_index=k,tick=5*k,observation=np.asarray(observation).tolist(),prefix=list(self.prefix),
                pulse=asdict(ps),candidate_ids=ids.tolist(),predicted_peak=f['peak'].tolist(),predicted_contact=f['contact'].tolist(),
                predicted_utility=f['utility'],domain_errors=f['errors'],selected_id=self.selected,
                all_unscorable_fallback=fallback,selected_states=f['states'][j,:count+1].tolist(),selected_peaks=f['peaks'][j,:count+1].tolist(),
                selected_model_end=end,model_tick_slots=len(ids)*horizon,evaluated_model_steps=f['evaluated_steps'],
                q_reconstruction_residual=f['q_reconstruction_residual'],snaps=f['snaps'],elapsed_s=time.monotonic()-started))
        elif self.mode=='initial_context' and observation is not None:
            raise ValueError('Initial-context policy must not receive subsequent scene observation')
        elif len(self.candidates)==1:self.selected=int(self.candidates[0])
        action=int(self.bank.actions[self.selected,k]);self.prefix.append(action)
        self.candidates=self.candidates[self.bank.actions[self.candidates,k]==action]
        self.pulse.request(action)
        for _ in range(5):self.pulse.advance()
        return action
