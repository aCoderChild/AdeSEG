"""Use the author SAM2 multimask decoder only during initial point prompts."""
from contextlib import contextmanager
from types import MethodType

import torch


@contextmanager
def initial_multimask(model, observations):
    """Temporarily choose the existing three-mask, best-predicted-IoU path.

    This does not alter tracking, select from labels, or introduce a new
    confidence cutoff. The video model's ordinary no-object decision remains.
    """
    original_choose = model._use_multimask
    original_head = model._forward_sam_heads

    def choose(self, is_init_cond_frame, point_inputs):
        if not is_init_cond_frame or point_inputs is None:
            raise ValueError('Multimask override is restricted to initial point prompts')
        return True

    def head(self, *args, **kwargs):
        if not kwargs.get('multimask_output', False):
            raise ValueError('Expected the author three-mask path')
        result = original_head(*args, **kwargs)
        low, high, ious, selected_low, selected_high = result[:5]
        if ious.shape != (1, 3) or low.shape[1] != 3 or high.shape[1] != 3:
            raise ValueError('Unexpected author multimask outputs')
        if not torch.isfinite(ious).all():
            raise ValueError('Nonfinite predicted mask IoU')
        chosen = int(ious[0].argmax().item())
        if not torch.equal(selected_low[0, 0], low[0, chosen]) or not torch.equal(selected_high[0, 0], high[0, chosen]):
            raise ValueError('Author selected a different mask than its best predicted IoU')
        observations.append({'predicted_iou': [float(x) for x in ious[0].detach().cpu()], 'selected': chosen})
        return result

    model._use_multimask = MethodType(choose, model)
    model._forward_sam_heads = MethodType(head, model)
    try:
        yield
    finally:
        model._use_multimask = original_choose
        model._forward_sam_heads = original_head
