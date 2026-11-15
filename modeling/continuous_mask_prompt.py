"""Fixed, continuous SAM2 mask-prompt diagnostic from saved score maps."""
import numpy as np
import torch
import torch.nn.functional as F

def mask_and_score_prompts(mask,score,seed):
    """Return a fixed binary-mask logit prior and standardized score variants.

    Spatial size is the original SAM2 256x256 mask-input grid. This is a
    diagnostic prompt construction, not a probability calibration or a learned
    module. No threshold is applied to the continuous score.
    """
    mask=np.asarray(mask,dtype=bool);score=np.asarray(score,dtype=np.float32)
    if mask.ndim!=2 or score.shape!=(256,256) or not np.isfinite(score).all():raise ValueError('Invalid mask or score geometry')
    res=F.interpolate(torch.from_numpy(mask.astype(np.float32))[None,None],size=(256,256),mode='area')[0,0].numpy()
    prior=(8.0*res-4.0).astype(np.float32)
    std=float(score.std(dtype=np.float64))
    if std<=0 or not np.isfinite(std):raise ValueError('Constant score cannot be standardized')
    z=((score-float(score.mean(dtype=np.float64)))/std).astype(np.float32)
    shuffled=np.random.default_rng(seed).permutation(z.reshape(-1)).reshape(z.shape)
    return {'mask_only':prior[None], 'score':(prior+z)[None], 'shuffled':(prior+shuffled)[None]}
