"""Warm the historical box/point state and always commit the archived mask.

The generated mask is diagnostic only; the archived automatic mask is the
fixed conditioning input for every patient, irrespective of their agreement.
"""
import numpy as np
import torch


def apply_archived_warm_prompt_v2(model,state,frame,box,points,archived_mask,*,before_mask=None):
    box=np.asarray(box,dtype=np.float32);points=np.asarray(points,dtype=np.float32)
    archived=np.asarray(archived_mask,dtype=bool).copy()
    if (type(frame) is not int or frame<0 or box.shape!=(4,) or points.shape!=(2,2) or archived.ndim!=2
            or not np.isfinite(box).all() or not np.isfinite(points).all() or not archived.any()):
        raise ValueError('Invalid archived automatic warm prompt')
    h,w=archived.shape
    if not (0<=box[0]<box[2]<=w and 0<=box[1]<box[3]<=h and
            ((points[:,0]>=0)&(points[:,0]<w)&(points[:,1]>=0)&(points[:,1]<h)).all()):
        raise ValueError('Archived prompt lies outside its viewport')
    _,ids,logits=model.add_new_points_or_box(state,frame_idx=frame,obj_id=1,box=box)
    if ids!=[1] or not bool(torch.isfinite(logits).all()):raise ValueError('Invalid box-prompt output')
    _,ids,logits=model.add_new_points_or_box(state,frame_idx=frame,obj_id=1,points=points,
        labels=np.array([1,0],np.int32),clear_old_points=False)
    if ids!=[1] or logits.ndim!=4 or tuple(logits.shape[:2])!=(1,1) or not bool(torch.isfinite(logits).all()):
        raise ValueError('Invalid point-prompt output')
    generated=logits[0,0].detach().cpu().numpy()>0
    if generated.shape!=archived.shape:raise ValueError('Generated prompt geometry differs')
    chosen=archived.copy()
    if before_mask is not None:before_mask(chosen)
    if not np.array_equal(chosen,archived):raise ValueError('Archived conditioning mask changed before commit')
    _,ids,_=model.add_new_mask(state,frame_idx=frame,obj_id=1,mask=chosen)
    if ids!=[1]:raise ValueError('Invalid warm-mask object')
    return {'conditioning_mask':chosen,'generated_mask':generated,
            'generated_equals_archived':bool(np.array_equal(generated,archived)),
            'different_pixels':int(np.count_nonzero(generated!=archived))}
