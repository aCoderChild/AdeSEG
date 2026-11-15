"""Read-only summaries of Lean's author-provided per-key attention usage."""
import torch

def summarize_key_usage(usage, conditioning_count, pointer_count):
    if usage.ndim != 1 or not torch.isfinite(usage).all() or (usage < 0).any():
        raise ValueError('Expected finite nonnegative per-key usage')
    if not 0 <= conditioning_count <= len(usage) or not 0 <= pointer_count <= len(usage)-conditioning_count:
        raise ValueError('Invalid memory segment lengths')
    total = usage.sum()
    if total <= 0:
        raise ValueError('Attention usage has zero mass')
    end = len(usage)-pointer_count
    return {'total_usage':float(total.item()),
            'conditioning_mass':float((usage[:conditioning_count].sum()/total).item()),
            'other_spatial_mass':float((usage[conditioning_count:end].sum()/total).item()),
            'pointer_mass':float((usage[end:].sum()/total).item())}
