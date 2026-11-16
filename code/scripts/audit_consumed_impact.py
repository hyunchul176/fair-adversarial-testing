"""Independent center-projection audit of native consumed pair impacts."""
import math
import numpy as np


def native_steering(command):
    # The installed ContinuousAction affine map uses NumPy float32 operations.
    return float(np.float32(-.3)+(np.float32(command)+np.float32(1))*np.float32(.6)/np.float32(2))


def axes(s):
    c=math.cos(s['heading']);q=math.sin(s['heading']);return [(c,q),(-q,c)]
def dot(a,b):return sum(x*y for x,y in zip(a,b))
def radius(s,u):
    v,w=axes(s);return s['length']/2*abs(dot(v,u))+s['width']/2*abs(dot(w,u))
def swept(a,b,dt=.1):
    delta=(b['x']-a['x'],b['y']-a['y']);dv=(b['vx']-a['vx'],b['vy']-a['vy']);entry=0.;leave=math.inf;current=True;expanded=True;depth=math.inf;transition=None
    for u in axes(a)+axes(b):
        ra=radius(a,u);rb=radius(b,u);pos=dot(delta,u);vel=dot(dv,u);r=ra+rb
        if abs(pos)>r:current=False
        if abs(vel)<1e-12:
            if abs(pos)>r:leave=-math.inf
        else:
            interval=sorted(((-r-pos)/vel,(r-pos)/vel));entry=max(entry,interval[0]);leave=min(leave,interval[1])
        low=-ra+min(0.,-vel*dt);high=ra+max(0.,-vel*dt);blow=pos-rb;bhigh=pos+rb
        distance=blow-high if low<blow else low-bhigh
        if distance>0:expanded=False
        if abs(distance)<depth:
            depth=abs(distance);sign=1 if -pos>0 else -1;transition=tuple(sign*depth*x for x in u)
    t=entry if leave>=entry else None
    return dict(current_overlap=current,ttc=t,scheduled=expanded,transition=transition if expanded else None)


def minimum_translation_candidates(a, b, dt=.1):
    """Independent vertex SAT with roundoff-bounded minimum-axis ties.

    Native SAT builds normals from world-coordinate corner differences. Near
    parallel axes can exchange their depth ordering under binary64 rounding;
    a unique center-projection argmin is then not an identifiable invariant.
    Only minimum-depth candidates are returned, not arbitrary separating axes.
    The caller still checks each consumed displacement to its original 1e-8 m.
    """
    def corners(s):
        c,q=math.cos(s['heading']),math.sin(s['heading'])
        l,w=s['length']/2,s['width']/2
        return np.array([(s['x']+c*x-q*y,s['y']+q*x+c*y)
                         for x,y in ((-l,-w),(-l,w),(l,w),(l,-w))])
    pa,pb=corners(a),corners(b)
    relative=np.array([a['vx']-b['vx'],a['vy']-b['vy']])*dt
    center=np.array([a['x']-b['x'],a['y']-b['y']])
    # Coordinate subtraction/projection error plus normalized-edge error:
    # O(eps*world_scale) * (1 + relative_extent/min_edge). 64 covers the
    # bounded arithmetic chain; this is an axis-order bound, not a pose tolerance.
    scale=max(1.,float(np.abs(pa).max()),float(np.abs(pb).max()))
    edge=min(a['length'],a['width'],b['length'],b['width'])
    assert edge>0
    extent=float(np.linalg.norm(center)+np.linalg.norm(relative))+max(a['length'],a['width'],b['length'],b['width'])
    bound=64*np.finfo(float).eps*scale*(1+extent/edge)
    candidates=[]
    for polygon in (pa,pb):
        for i in range(4):
            e=polygon[(i+1)%4]-polygon[i]
            u=np.array([-e[1],e[0]]);u/=np.linalg.norm(u)
            ap,bp=pa@u,pb@u
            displacement=float(relative@u)
            lo=float(ap.min())+min(0.,displacement)
            hi=float(ap.max())+max(0.,displacement)
            distance=float(bp.min())-hi if lo<float(bp.min()) else lo-float(bp.max())
            assert distance<=bound, 'Saved scheduled contact has a separating axis'
            depth=abs(distance)
            candidates.append((depth,(1 if center@u>0 else -1)*depth*u))
    minimum=min(depth for depth,_ in candidates)
    return [v for depth,v in candidates if depth-minimum<=bound]


def consumed_impact(row):
    out=[];assert row['pair_contact'] and row['termination_reason']=='collision' and set(row['crashed_ids'])=={0,1}
    for e in row['confirmed_contacts']:
        assert set(e['actors'])=={0,1} and e['kind']=='scheduled_impact'
        step=e['step'];assert step<row['steps']-1
        a,b=e['states'];before=row['trace'][step];after=row['trace'][step+1]
        assert e['actors']==[0,1] and a==before['generator'] and b==before['target']
        geom=swept(a,b);assert geom['scheduled'] and not geom['current_overlap'] and geom['ttc'] is not None and 0<geom['ttc']<=.1
        beta=math.atan(.5*math.tan(native_steering(after['applied_action'][1])));speed=math.hypot(a['vx'],a['vy'])
        unimpacted_g=(a['x']+.1*speed*math.cos(a['heading']+beta),a['y']+.1*speed*math.sin(a['heading']+beta));unimpacted_t=(b['x']+.1*b['vx'],b['y']+.1*b['vy'])
        shift_g=(after['generator']['x']-unimpacted_g[0],after['generator']['y']-unimpacted_g[1]);shift_t=(after['target']['x']-unimpacted_t[0],after['target']['y']-unimpacted_t[1]);tr=geom['transition']
        assert max(abs(x-y/2) for x,y in zip(shift_g,tr))<1e-8 and max(abs(x+y/2) for x,y in zip(shift_t,tr))<1e-8
        out.append(dict(schedule_tick=step,schedule_time_s=(step+1)*.1,consumed_tick=step+1,consumed_time_s=(step+2)*.1,predicted_contact_after_schedule_s=geom['ttc'],native_transition=tr,measured_generator_shift=shift_g,measured_target_shift=shift_t,post_consumption_ttc=swept(after['generator'],after['target'])['ttc']))
    assert out
    # This first consumed impact changes position, but the speed update still
    # obeys the last pre-crash action. Check all saved ticks, including it.
    for actor,limit,jerk in [(0,3.,2.),(1,5.,2.5)]:
        prior=0.
        for r in row['trace']:
            accel=r['acceleration'][actor];command=5*r['applied_action'][0] if actor==0 else r['target_control']['applied_mps2']
            assert abs(accel-command)<1e-5 and abs(accel)<=limit+1e-5 and abs((accel-prior)/.1)<=jerk+1e-5;prior=accel
    return dict(input_id=row['input_id'],case_id=row['case_id'],family=row['family'],target_id=row['target_id'],horizon_s=row['horizon_s'],gap_delta_m=row['gap_delta_m'],speed_delta_mps=row['speed_delta_mps'],steps=row['steps'],crashed_ids=row['crashed_ids'],termination_reason=row['termination_reason'],sampled_overlap_frames=sum(swept(r['generator'],r['target'])['current_overlap'] for r in row['trace']),all_saved_acceleration_and_jerk_pass=True,impacts=out)
