"""Use pinned native SAM2 mask/encoder routines for same-frame oracle values.

No synthetic neural encoder or encoder-weight change. GT is permitted only in
this labelled TRAIN diagnostic, never a normal inference memory write.
"""
import torch
import torch.nn.functional as F

from scripts.run_token_conditioned_key_adapter import _tree_digest


def same_frame_teachers(model, features, stored, ground_truth):
    if model.training or any(p.requires_grad for p in model.parameters()):
        raise RuntimeError('Oracle encoding requires frozen eval SAM2')
    if ground_truth.ndim != 4 or ground_truth.shape[:2] != (1, 1):
        raise ValueError('GT must be a single-object BCHW mask')
    if not torch.isfinite(ground_truth).all() or not ((ground_truth == 0) | (ground_truth == 1)).all():
        raise ValueError('GT must be binary; do not infer a region threshold')
    vision, sizes = features['vision'], features['sizes']
    with torch.no_grad():
        # Match _forward_sam_heads' native low-to-high interpolation. Fail
        # closed if re-encoding this prediction cannot reproduce stored memory.
        predicted_high = F.interpolate(stored['pred_masks'].float(),
            size=(model.image_size, model.image_size), mode='bilinear', align_corners=False)
        native, native_pos = model._encode_new_memory(vision, sizes, predicted_high,
            stored['object_score_logits'], is_mask_from_pts=False)
        native = native.to(dtype=stored['maskmem_features'].dtype)
        if _tree_digest(native) != _tree_digest(stored['maskmem_features']):
            raise RuntimeError('Native predicted-mask re-encoding differs from stored source memory')
        if _tree_digest(native_pos) != _tree_digest(stored['maskmem_pos_enc']):
            raise RuntimeError('Native source position encoding differs on re-encoding')

        mask = ground_truth.float().to(vision[-1].device)
        if mask.shape[-2:] != (model.image_size, model.image_size):
            # This is SAM2VideoPredictor.add_new_mask's supplied-mask resize,
            # not a learned decision rule or a new inference threshold.
            mask = F.interpolate(mask, size=(model.image_size, model.image_size),
                align_corners=False, mode='bilinear', antialias=True) >= .5
        backbone = vision[-1].permute(1, 2, 0).reshape(1, model.hidden_dim, *sizes[-1])
        high_res = [v.permute(1, 2, 0).reshape(1, -1, *s)
                    for v, s in zip(vision[:-1], sizes[:-1])] if model.use_high_res_features_in_sam else None
        # Reuse the author's +/-10 mask-logit and object-presence convention;
        # ignore the GT-derived pointer. Native pointers are never replaced.
        outputs = model._use_mask_as_output(backbone, high_res, mask)
        teacher_high, teacher_score = outputs[4], outputs[6]
        teachers = {}
        for name, score in [('mask_only', stored['object_score_logits']),
                            ('mask_and_presence', teacher_score)]:
            value, positions = model._encode_new_memory(vision, sizes, teacher_high, score,
                                                        is_mask_from_pts=False)
            if _tree_digest(positions) != _tree_digest(native_pos):
                raise RuntimeError('Mask teacher changed positional encoding')
            value = value.to(dtype=stored['maskmem_features'].dtype)
            if not torch.isfinite(value).all():
                raise RuntimeError('Nonfinite GT-encoded oracle values')
            teachers[name] = value.detach()
    return {'native_reencoded': native.detach(), **teachers,
            'native_reencoding_exact': True,
            'native_presence_logit': float(stored['object_score_logits'].item()),
            'gt_presence_logit': float(teacher_score.item()),
            'gt_used_only_for_oracle_encoding': True}
