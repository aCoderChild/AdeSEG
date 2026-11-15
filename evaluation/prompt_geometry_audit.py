"""Post-hoc geometry of a saved box/mask against an evaluation label."""
import numpy as np


def evaluate_prompt_geometry(box,initial,truth):
    pred=np.asarray(initial,dtype=bool);gt=np.asarray(truth,dtype=bool);b=np.asarray(box,dtype=np.float64)
    if pred.ndim!=2 or pred.shape!=gt.shape or b.shape!=(4,) or not np.isfinite(b).all():raise ValueError('Invalid prompt geometry')
    h,w=gt.shape;x0,y0,x1,y1=b
    if not (0<=x0<x1<=w and 0<=y0<y1<=h):raise ValueError('Prompt box outside source image')
    # Rasterize fractional continuous coordinates by pixel-center inclusion.
    xs=np.arange(w,dtype=np.float64)+.5;ys=np.arange(h,dtype=np.float64)+.5
    box_mask=(ys[:,None]>=y0)&(ys[:,None]<y1)&(xs[None,:]>=x0)&(xs[None,:]<x1)
    inter=int((pred & gt).sum());fg=int(gt.sum());predn=int(pred.sum());area=int(box_mask.sum())
    overlap=int((box_mask & gt).sum())
    # GT-empty Dice is not reported; predicted foreground is separately marked.
    return {'present':fg>0,'truth_pixels':fg,'prompt_mask_pixels':predn,'prompt_mask_fraction':predn/(h*w),
      'box_pixels':area,'box_fraction':area/(h*w),'prompt_mask_dice':2*inter/(predn+fg) if fg else None,
      'box_truth_coverage':overlap/fg if fg else None,'box_precision_pixels':overlap/area if fg and area else None,
      'empty_prompt_false_alarm':bool(predn) if not fg else None}
