"""Native predicted-memory clip training for the ordinary decoder baseline."""
import torch

from modeling.cfbi_training import clone_tensors
from modeling.decoder_adaptation import assert_feature_cache_safe, decoder_loss
from modeling.decoder_training_mask import decoder_training_mask


def decoder_clip_loss(model, samples, *, check_native=False, retain_first_write=False):
    assert_feature_cache_safe(model)
    if len(samples)<2 or any(b['frame']!=a['frame']+1 for a,b in zip(samples,samples[1:])):
        raise ValueError('Expected consecutive post-prompt native reads')
    bank = clone_tensors(samples[0]['output_dict'])
    losses, terms, first_write = [], [], None
    for index, sample in enumerate(samples):
        if sample['num_frames'] != samples[0]['num_frames']:
            raise ValueError('Full-video timing differs within a clip')
        with decoder_training_mask(model.sam_mask_decoder) as captured:
            output = model.track_step(frame_idx=sample['frame'],is_init_cond_frame=False,
                current_vision_feats=sample['vision'],current_vision_pos_embeds=sample['positions'],
                feat_sizes=sample['sizes'],point_inputs=None,mask_inputs=None,
                output_dict=bank,num_frames=sample['num_frames'],track_in_reverse=False,
                run_mem_encoder=index+1<len(samples))
        loss, components = decoder_loss(captured,sample['target'])
        losses.append(loss)
        terms.append({name:float(value.detach()) for name,value in components.items()})
        if check_native and (not torch.equal(output['pred_masks'],sample['native_logits']) or
                not torch.equal(output['object_score_logits'],sample['native_object_score'])):
            raise RuntimeError('Initial decoder clip differs from native masks/scores')
        if index+1<len(samples):
            stored = {key:output[key] for key in
                ('maskmem_pos_enc','pred_masks','obj_ptr','object_score_logits')}
            stored['maskmem_features'] = output['maskmem_features'].to(torch.bfloat16)
            if check_native:
                reference = samples[index+1]['output_dict']['non_cond_frame_outputs'][sample['frame']]
                for key in ('maskmem_features','pred_masks','obj_ptr','object_score_logits'):
                    if not torch.equal(stored[key],reference[key]):
                        raise RuntimeError(f'Initial predicted memory differs: {key}')
            if index==0 and retain_first_write:
                first_write = stored['maskmem_features']
                first_write.retain_grad()
            bank['non_cond_frame_outputs'][sample['frame']] = stored
    return torch.stack(losses).sum(), {'terms':terms,'first_write':first_write,'last_loss':losses[-1]}
