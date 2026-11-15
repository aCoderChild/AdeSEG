"""Connect a trained prompt-conditioned channel gate to Lean score capture.

The original author memory selection and the region-budget runtime are not
changed. This replaces only the captured early-feature contrast score used by
an explicitly selected experimental pool policy. The gate requires a matching
trained checkpoint; inference never fits its weights or reads GT.
"""
from contextlib import contextmanager
from types import MethodType
import hashlib
import torch
from modeling.lean_channel_attention import PromptContrastChannelAttention
from modeling.lean_region_token_pool import capture_region_scores
from modeling.prompt_feature_refinement import reference_prototypes


@contextmanager
def capture_channel_attention_scores(model,checkpoint,backbone_sha256,feature_pipeline_sha256,observations):
    # The caller must compute both hashes from the loaded model and runtime;
    # matching metadata alone cannot independently attest the running model.
    gate=PromptContrastChannelAttention(checkpoint=checkpoint,expected_backbone_sha256=backbone_sha256,
        expected_feature_pipeline_sha256=feature_pipeline_sha256).eval().requires_grad_(False)
    with capture_region_scores(model,observations):
        get_feature=model._get_image_feature;add_mask=model.add_new_mask
        cache=model._cache_tarr_affinity_from_new_memory;dump=model._dump_retained_tokens_record
        state={'frame':None,'feature':None,'prototypes':None}
        def feature(self,inference_state,frame_idx,batch_size):
            out=get_feature(inference_state,frame_idx,batch_size);f=out[1]['backbone_fpn'][0]
            if batch_size!=1 or f.ndim!=4 or f.shape[0]!=1 or f.shape[1]!=32 or f.device.type!='cpu':
                raise ValueError('Expected the verified single-object CPU 32-channel early feature')
            state['frame']=int(frame_idx);state['feature']=f[0].detach().float()
            return out
        def mask(self,inference_state,frame_idx,obj_id,mask):
            if state['frame']!=int(frame_idx) or state['prototypes'] is not None:raise ValueError('Initial automatic prompt source differs')
            state['prototypes']=reference_prototypes(state['feature'],mask)
            return add_mask(inference_state,frame_idx=frame_idx,obj_id=obj_id,mask=mask)
        def cached(self,current_out,output_dict,frame_idx,obj_idx,is_init_cond_frame,run_mem_encoder,reverse):
            result=cache(current_out,output_dict,frame_idx,obj_idx,is_init_cond_frame,run_mem_encoder,reverse)
            if not is_init_cond_frame and isinstance(current_out,dict) and torch.is_tensor(current_out.get('maskmem_features')):
                if state['prototypes'] is None or state['frame']!=int(frame_idx):raise ValueError('Early feature source differs')
                with torch.inference_mode():score,weights=gate(state['feature'],*state['prototypes'])
                if score.shape!=(4096,) or not torch.isfinite(score).all():raise ValueError('Invalid channel-weighted contrast')
                current_out['pool_contrast_full']=score.clamp(-1,1).cpu().clone()
                current_out['channel_weights_sha256']=hashlib.sha256(weights.cpu().contiguous().numpy().tobytes()).hexdigest()
            return result
        def selected(self,**kwargs):
            result=dump(**kwargs);entry=kwargs['latest_prev'];contrast=entry.get('pool_contrast');weights_hash=entry.pop('channel_weights_sha256',None)
            if not torch.is_tensor(contrast) or not isinstance(weights_hash,str):raise ValueError('Missing aligned channel scores')
            observations[-1]['channel_contrast_sha256']=hashlib.sha256(contrast.cpu().contiguous().numpy().tobytes()).hexdigest()
            observations[-1]['channel_weights_sha256']=weights_hash
            return result
        model._get_image_feature=MethodType(feature,model);model.add_new_mask=MethodType(mask,model)
        model._cache_tarr_affinity_from_new_memory=MethodType(cached,model);model._dump_retained_tokens_record=MethodType(selected,model)
        try:yield
        finally:
            model._get_image_feature=get_feature;model.add_new_mask=add_mask
            model._cache_tarr_affinity_from_new_memory=cache;model._dump_retained_tokens_record=dump
