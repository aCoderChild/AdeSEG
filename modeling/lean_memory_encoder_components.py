"""Combine frozen encoder inputs for local, off-manifold mechanism tests."""
from itertools import product

COMPONENTS=('vision','mask','object')
ARMS=tuple(''.join(bits) for bits in product('rd',repeat=3))


def mix_encoder_inputs(recipient,donor,arm):
    """Use r=recipient or d=donor for vision, mask and native object logit.

    This is diagnostic only: combinations may be inconsistent with a natural
    trajectory. No prediction or thresholding is done by this helper.
    """
    required={'current_vision_feats','feat_sizes','pred_masks_high_res','object_score_logits','is_mask_from_pts'}
    if set(recipient)!=required or set(donor)!=required or arm not in ARMS:
        raise ValueError('Unexpected encoder inputs or arm')
    if recipient['is_mask_from_pts']!=donor['is_mask_from_pts'] or recipient['feat_sizes']!=donor['feat_sizes']:
        raise ValueError('Encoder geometry or mask-prompt status differs')
    v,m,o=(donor if bit=='d' else recipient for bit in arm)
    return {'current_vision_feats':v['current_vision_feats'],'feat_sizes':v['feat_sizes'],
            'pred_masks_high_res':m['pred_masks_high_res'],'object_score_logits':o['object_score_logits'],
            'is_mask_from_pts':recipient['is_mask_from_pts']}
