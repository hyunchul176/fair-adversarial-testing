"""Minimal C2 perturbation fixing an observed midpoint without changing endpoints."""
import numpy as np
from numpy.polynomial import Polynomial as P


def coefficients(duration,middle):
    if not 0<middle<duration:raise ValueError('Unordered lane-change times')
    lo,hi=0.,.5
    for _ in range(60):
        u=(lo+hi)/2
        if 10*u**3-15*u**4+6*u**5<.1:lo=u
        else:hi=u
    tenth=(lo+hi)/2;phase=P([tenth,1-2*tenth])
    baseline=4*(10*phase**3-15*phase**4+6*phase**5)
    fraction=middle/duration;bump=P([0,0,0,1,-3,3,-1])
    scale=(2-float(baseline(fraction)))/float(bump(fraction))
    return baseline+scale*bump,float(scale)


def extrema(poly):
    points=[0.,1.]+[float(r.real) for r in poly.deriv().roots() if abs(r.imag)<1e-8 and 0<r.real<1]
    values=[float(poly(t)) for t in points]
    return min(values),max(values)


def preflight(case):
    start,middle,end=case['event']['times'];duration=end-start
    poly,scale=coefficients(duration,middle-start)
    velocity=extrema(poly.deriv()/duration);acceleration=extrema(poly.deriv(2)/duration**2);jerk=extrema(poly.deriv(3)/duration**3)
    checks=dict(monotone=velocity[0]>=-1e-8,acceleration=max(map(abs,acceleration))<=2+1e-5,jerk=max(map(abs,jerk))<=4+1e-5)
    return dict(checks=checks,eligible=all(checks.values()),scale=scale,coefficients=poly.coef.tolist(),
        midpoint_fraction=(middle-start)/duration,velocity_extrema=velocity,acceleration_extrema=acceleration,jerk_extrema=jerk)


def correction(case,time):
    start,middle,end=case['event']['times'];duration=end-start
    if time<=0 or time>=duration:return 0.
    _,scale=coefficients(duration,middle-start);u=time/duration
    return scale*u**3*(1-u)**3
