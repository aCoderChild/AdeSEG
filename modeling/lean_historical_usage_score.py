"""Tie-aware normalized ranking of a source's previously measured usage."""
import numpy as np

def percentile_ranks(values):
    x=np.asarray(values,dtype=np.float64)
    if x.ndim!=1 or x.size<2 or not np.isfinite(x).all():raise ValueError('Expected finite one-dimensional usage')
    unique,inverse,counts=np.unique(x,return_inverse=True,return_counts=True)
    starts=np.cumsum(counts)-counts
    average=starts+(counts-1)/2
    return average[inverse]/(x.size-1)
