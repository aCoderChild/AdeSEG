"""Make author SAMURAI's NumPy Kalman-IoU outputs compatible with MPS.

The author passes a list of NumPy float64 scalars to ``torch.tensor(...,
device='mps')``, which fails once stable-frame Kalman scoring is reached.
Only on MPS, return the same IoUs rounded to float32; CUDA/CPU keep the author
values. Report calls and rounding error, without changing mask selection rules.
"""
from contextlib import contextmanager
import numpy as np


@contextmanager
def portable_kalman_iou(model, device, observations):
    if device not in ('cpu', 'mps'):
        raise ValueError('Unsupported SAMURAI device')
    kalman=model.kf
    if 'compute_iou' in vars(kalman):
        raise RuntimeError('Cannot compose with an existing Kalman-IoU hook')
    original=kalman.compute_iou
    def compute(*args, **kwargs):
        result=original(*args, **kwargs)
        reference=np.asarray(result, dtype=np.float64)
        if reference.ndim != 1 or not np.isfinite(reference).all():
            raise ValueError('Invalid author Kalman-IoU result')
        cast=reference.astype(np.float32)
        observations.append({'count':int(reference.size),
                             'max_rounding_error':float(np.max(np.abs(reference-cast.astype(np.float64)))) if reference.size else 0.})
        return cast if device=='mps' else result
    try:
        kalman.compute_iou=compute
        yield
    finally:
        delattr(kalman, 'compute_iou')
