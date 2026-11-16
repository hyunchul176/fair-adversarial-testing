"""Marginal histogram Hellinger in a single standardized reference space.

This is an average of marginal distances, not joint GMM Hellinger. Bins depend
only on the fixed reference, with overflow bins to retain out-of-range mass.
"""
import numpy as np

METRIC_VERSION='fairtest-marginal-hellinger-v2'

def marginal_hellinger_scaled(observed, reference, n_bins=30):
    observed=np.asarray(observed,dtype=float)
    reference=np.asarray(reference,dtype=float)
    if observed.ndim==1: observed=observed[:,None]
    if reference.ndim==1: reference=reference[:,None]
    if observed.ndim!=2 or reference.ndim!=2 or observed.shape[1]!=reference.shape[1]:
        raise ValueError('Feature dimensions do not match')
    if not len(observed) or not len(reference): raise ValueError('Empty feature sample')
    if not np.isfinite(observed).all() or not np.isfinite(reference).all():
        raise ValueError('Non-finite feature sample')
    if n_bins<3: raise ValueError('At least three bins required')
    distances=[]
    for dim in range(observed.shape[1]):
        ref=reference[:,dim]
        lo,hi=float(ref.min()),float(ref.max())
        if hi-lo<1e-9: lo-=0.5;hi+=0.5
        # n_bins total: n_bins-2 interior bins and two overflow bins.
        bins=np.r_[-np.inf,np.linspace(lo,hi,n_bins-1),np.inf]
        p=np.histogram(observed[:,dim],bins=bins)[0].astype(float)/len(observed)
        q=np.histogram(ref,bins=bins)[0].astype(float)/len(reference)
        distances.append(float(np.sqrt(np.sum((np.sqrt(p)-np.sqrt(q))**2)/2)))
    return float(np.mean(distances))

def marginal_hellinger_from_features(features, scaler, reference_scaled, n_bins=30):
    """Transform raw observations once; reference comes directly from a fitted GMM."""
    return marginal_hellinger_scaled(scaler.transform(features),reference_scaled,n_bins)
