"""Deterministic points from saved score maps and an automatic target mask."""
import numpy as np

def points_in_mask(score, mask, seed):
    score = np.asarray(score)
    mask = np.asarray(mask, dtype=bool)
    if score.ndim != 2 or mask.ndim != 2 or not np.isfinite(score).all():
        raise ValueError('Invalid map or mask')
    mh, mw = mask.shape
    h, w = score.shape
    yy = np.minimum(((np.arange(h) + .5) * mh / h).astype(int), mh-1)
    xx = np.minimum(((np.arange(w) + .5) * mw / w).astype(int), mw-1)
    cells = np.flatnonzero(mask[yy[:,None], xx[None,:]].reshape(-1))
    if len(cells) < 2: return None
    ordered = cells[np.argsort(score.reshape(-1)[cells], kind='stable')]
    ids = [int(ordered[-1]), int(ordered[0])]
    random_ids = np.random.default_rng(seed).choice(cells, 2, replace=False).tolist()
    def coords(indices):
        return [[float((i % w + .5)*mw/w), float((i // w + .5)*mh/h)] for i in indices]
    return {'feature_points':coords(ids), 'random_points':coords(random_ids), 'valid_cells':int(len(cells))}
