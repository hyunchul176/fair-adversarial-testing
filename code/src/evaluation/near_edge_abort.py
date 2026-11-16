"""C2 near-edge lateral stop probes; synthetic controls, not observed aborts."""
from functools import lru_cache
import numpy as np
from numpy.polynomial import Polynomial as P
from src.evaluation.observed_lateral_timing import extrema

@lru_cache(maxsize=256)
def coefficients(duration,fraction,endpoint):
    if duration<=0 or not 0<fraction<1 or not .4<endpoint<2:raise ValueError('Invalid near-edge stop specification')
    roots=np.roots([6.,-15.,10.,0.,0.,-.1]);u=float(next(r.real for r in roots if abs(r.imag)<1e-8 and 0<r.real<.5))
    scale=(1-2*u)/duration;v=120*u*u*(1-u)**2*scale;a=4*(60*u-180*u*u+120*u**3)*scale**2
    stop=duration*fraction;c0=.4;c1=v*stop;c2=.5*a*stop**2
    tail=np.linalg.solve([[1,1,1],[3,4,5],[6,12,20]],[endpoint-c0-c1-c2,-c1-2*c2,-2*c2])
    return P([c0,c1,c2,*tail]),stop

def preflight(case,spec):
    poly,stop=coefficients(case['event']['metrics']['duration_10_90'],spec['stop_fraction'],spec['endpoint_y'])
    v=extrema(poly.deriv()/stop);a=extrema(poly.deriv(2)/stop**2);j=extrema(poly.deriv(3)/stop**3)
    checks=dict(monotone=v[0]>=-1e-8,acceleration=max(map(abs,a))<=2+1e-5,jerk=max(map(abs,j))<=4+1e-5)
    return dict(checks=checks,eligible=all(checks.values()),coefficients=poly.coef.tolist(),stop_s=stop,
        velocity_extrema=v,acceleration_extrema=a,jerk_extrema=j)

def reference(case,spec,time):
    poly,stop=coefficients(case['event']['metrics']['duration_10_90'],spec['stop_fraction'],spec['endpoint_y'])
    return float(poly(max(0,time)/stop)) if time<stop else spec['endpoint_y']

def minimum_lateral_clearance(row):
    import math
    states=[row['physical_initial_state'][0]]+[r['generator'] for r in row['trace']]
    return min(3.-s['y']-(s['length']*abs(math.sin(s['heading']))+s['width']*abs(math.cos(s['heading'])))/2 for s in states)
