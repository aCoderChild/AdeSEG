"""Pure routing rules for output and memory-write refinement ablations."""
import torch


def route_refined_logits(original, refined, arm, *, conditioning):
    if arm not in ('A','B','C','D') or type(conditioning) is not bool:
        raise ValueError('Invalid refinement arm or conditioning flag')
    if not torch.is_tensor(original) or not torch.is_tensor(refined) or original.shape!=refined.shape or original.device!=refined.device or original.dtype!=refined.dtype:
        raise ValueError('Original and refined logits must align')
    if not torch.isfinite(original).all() or not torch.isfinite(refined).all():
        raise ValueError('Nonfinite logits')
    # Preserve the original one-shot conditioning prompt in all arms.
    if conditioning:return original,original
    output=refined if arm in ('B','D') else original
    memory=refined if arm in ('C','D') else original
    return output,memory
