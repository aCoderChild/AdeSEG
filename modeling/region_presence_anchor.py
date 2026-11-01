"""Train Cutie regional memory messages while preserving SAM2 presence scores."""
import torch
import torch.nn.functional as F

from modeling.cfbi_training import clone_tensors, object_presence_loss
from modeling.decoder_training_mask import decoder_training_mask, present_mask_loss, select_decoder_mask
from modeling.native_region_readout import NativeRegionalReadout
from modeling.regional_refinement import refinement_loss


def clip_loss_with_presence_anchor(model, heads, samples, sequence, *, anchor_weight=1.0,
                                   check_native=False, return_trace=False):
    if len(samples) < 2 or any(b['frame'] != a['frame'] + 1
                               for a, b in zip(samples, samples[1:])):
        raise ValueError('Expected consecutive frames')
    if any('native_object_score' not in sample for sample in samples):
        raise ValueError('Each clip sample needs its archived native object score')

    bank = clone_tensors(samples[0]['output_dict'])
    losses, anchors = [], []
    with NativeRegionalReadout(model, heads, 'regional', sequence) as observer:
        for index, sample in enumerate(samples):
            with decoder_training_mask(model.sam_mask_decoder) as masks:
                output = model.track_step(
                    frame_idx=sample['frame'], is_init_cond_frame=False,
                    current_vision_feats=sample['vision'],
                    current_vision_pos_embeds=sample['positions'],
                    feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None,
                    output_dict=bank, num_frames=sample['num_frames'],
                    track_in_reverse=False, run_mem_encoder=index + 1 < len(samples))
            logits = select_decoder_mask(masks)
            target = F.interpolate(sample['target'], size=logits.shape[-2:], mode='nearest')
            score = output['object_score_logits']
            native_score = sample['native_object_score'].to(score)
            present = sample['target'].flatten(1).gt(0).any(dim=1)
            anchor = (F.smooth_l1_loss(score[present], native_score[present])
                      if present.any() else score.new_zeros(()))
            anchors.append(anchor)
            losses.append(present_mask_loss(logits, target, refinement_loss)
                          + object_presence_loss(score, target)
                          + anchor_weight * anchor)

            if check_native:
                if not torch.equal(output['pred_masks'].detach(), sample['native_logits']):
                    raise RuntimeError(f"Native mask mismatch at frame {sample['frame']}")
                if not torch.equal(score.detach(), sample['native_object_score']):
                    raise RuntimeError(f"Native object score mismatch at frame {sample['frame']}")
            if index + 1 < len(samples):
                stored = {key: output[key] for key in (
                    'maskmem_pos_enc', 'pred_masks', 'obj_ptr', 'object_score_logits')}
                stored['maskmem_features'] = output['maskmem_features'].to(torch.bfloat16)
                if check_native:
                    native = samples[index + 1]['output_dict']['non_cond_frame_outputs'][sample['frame']]
                    for key in ('maskmem_features', 'pred_masks', 'obj_ptr', 'object_score_logits'):
                        if not torch.equal(stored[key].detach(), native[key]):
                            raise RuntimeError(f'Native stored {key} mismatch')
                bank['non_cond_frame_outputs'][sample['frame']] = stored

    if [call['layer'] for call in observer.calls] != [0, 1, 2, 3] * len(samples):
        raise RuntimeError('Expected all four attention layers for each read')
    loss = torch.stack(losses).mean()
    anchor_mean = torch.stack(anchors).mean()
    return (loss, observer.calls, anchor_mean) if return_trace else loss
