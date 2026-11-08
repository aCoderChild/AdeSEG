"""Train readout heads across one predicted SAM2 memory write."""
import torch
import torch.nn.functional as F

from modeling.cfbi_training import clone_tensors, object_presence_loss
from modeling.decoder_training_mask import decoder_training_mask, present_mask_loss, select_decoder_mask
from modeling.native_region_kv_gate import NativeRegionalKVGate
from modeling.native_region_readout import NativeRegionalReadout
from modeling.regional_refinement import refinement_loss


def _step(model, sample, bank, encode):
    return model.track_step(frame_idx=sample['frame'], is_init_cond_frame=False,
        current_vision_feats=sample['vision'], current_vision_pos_embeds=sample['positions'],
        feat_sizes=sample['sizes'], point_inputs=None, mask_inputs=None, output_dict=bank,
        num_frames=sample['num_frames'], track_in_reverse=False, run_mem_encoder=encode)


def clip_loss(model, heads, samples, mode, sequence, *, check_native=False,
              shuffle_seed=20261031, return_trace=False, region_encoder=None,
              attention_context_factory=None):
    if len(samples) < 2 or any(b['frame'] != a['frame'] + 1 for a, b in zip(samples, samples[1:])):
        raise ValueError('Expected consecutive frames')
    bank = clone_tensors(samples[0]['output_dict'])
    losses = []
    if attention_context_factory is not None:
        if not callable(attention_context_factory):
            raise TypeError('attention_context_factory must be callable')
        context = attention_context_factory(model, heads)
    elif mode == 'regional_kv_gain':
        context = NativeRegionalKVGate(model, heads, sequence)
    elif mode == 'cutie_region_summary':
        context = NativeRegionalReadout(model, heads, 'regional', sequence)
    else:
        context = NativeRegionalReadout(model, heads, mode, sequence, shuffle_seed,
                                        region_encoder=region_encoder)
    with context as observer:
        for index, sample in enumerate(samples):
            with decoder_training_mask(model.sam_mask_decoder) as masks:
                output = _step(model, sample, bank, index + 1 < len(samples))
            logits = select_decoder_mask(masks)
            target = F.interpolate(sample['target'], size=logits.shape[-2:], mode='nearest')
            losses.append(present_mask_loss(logits, target, refinement_loss)
                          + object_presence_loss(output['object_score_logits'], target))
            if check_native and not torch.equal(output['pred_masks'].detach(), sample['native_logits']):
                raise RuntimeError(f"Native output mismatch at frame {sample['frame']}")
            if index + 1 < len(samples):
                stored = {key: output[key] for key in ('maskmem_pos_enc', 'pred_masks', 'obj_ptr', 'object_score_logits')}
                stored['maskmem_features'] = output['maskmem_features'].to(torch.bfloat16)
                if check_native:
                    native = samples[index + 1]['output_dict']['non_cond_frame_outputs'][sample['frame']]
                    for key in ('maskmem_features', 'pred_masks', 'obj_ptr', 'object_score_logits'):
                        if not torch.equal(stored[key].detach(), native[key]):
                            raise RuntimeError(f'Native stored {key} mismatch')
                bank['non_cond_frame_outputs'][sample['frame']] = stored
    if [call['layer'] for call in observer.calls] != [0, 1, 2, 3] * len(samples):
        raise RuntimeError('Expected four attention layers per frame')
    loss = torch.stack(losses).mean()
    return (loss, observer.calls) if return_trace else loss


def pair_loss(model, heads, samples, mode, sequence, *, check_native=False, shuffle_seed=20261031):
    if len(samples) != 2:
        raise ValueError('Expected exactly two frames')
    return clip_loss(model, heads, samples, mode, sequence, check_native=check_native, shuffle_seed=shuffle_seed)
