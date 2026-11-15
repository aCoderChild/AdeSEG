"""Experimental cross-frame Lean token-pool adapter, label-free at inference.

Reuses the project's prompt prototypes, pooled early-FPN contrast and Lean's
own pre-pruning usage/indices. It does not add a trained channel-attention
module and is not the CMSA-Net CMA implementation.
"""
from contextlib import contextmanager
from types import MethodType
import hashlib
import torch
from modeling.prompt_feature_refinement import reference_prototypes
from modeling.lean_early_contrast_anchor import blend_anchor_with_early_contrast
from modeling.lean_token_storage_observer import representation
from modeling.lean_token_window_storage import TokenWindowRuntime
from modeling.cross_frame_token_pool import source_score_metadata,compact_stored_pool,apply_compacted_pool

class CrossFramePoolRuntime(TokenWindowRuntime):
    def __init__(self,budget,insurance_reserve,strategy):
        super().__init__(budget,insurance_reserve)
        if strategy not in ('usage','contrast','distributed'):raise ValueError('Unknown pool strategy')
        self.strategy=strategy

    def admit(self,model,output_dict,inference_state,obj_idx,frame,current_out):
        self.verify(output_dict,inference_state,obj_idx,'before_admission',frame)
        noncond=output_dict.get('non_cond_frame_outputs',{})
        for entry in noncond.values():
            if entry.get('smr_state')=='pending':
                ids=entry.get('pool_token_ids');keep=entry.get('pending_keep_idx')
                feat=entry.get('maskmem_features');pos=entry.get('maskmem_pos_enc')
                if not all(torch.is_tensor(x) for x in (ids,keep,feat)) or pos is None:
                    raise ValueError('Pending memory lacks token identity or dense tensors')
                if not torch.equal(ids.cpu().long(),keep.cpu().long()):
                    raise ValueError('Author selection and token IDs differ')
                pos=pos[-1] if isinstance(pos,(list,tuple)) else pos
                expected_feat=feat.flatten(2).permute(2,0,1).index_select(0,keep.to(feat.device).long())
                expected_pos=pos.flatten(2).permute(2,0,1).index_select(0,keep.to(pos.device).long())
                model._compress_memory_entry(entry)
                if not torch.equal(entry['maskmem_features_seq'],expected_feat) or not torch.equal(entry['maskmem_pos_seq'],expected_pos):
                    raise ValueError('Author compression misaligned features or positions with token IDs')
        self.verify(output_dict,inference_state,obj_idx,'after_compression',frame)
        cond=sum(representation(e)['feature_tokens'] for e in output_dict.get('cond_frame_outputs',{}).values())
        current=representation(current_out)['feature_tokens']
        if cond!=4096 or current!=4096 or frame in noncond:raise ValueError('Protected memory geometry changed')
        capacity=self.budget-self.insurance_reserve-cond-current
        if capacity<0:raise ValueError('Protected memories do not fit')
        history=[]
        for idx,entry in sorted(noncond.items()):
            if type(idx) is not int or idx>=frame:raise ValueError('Expected forward history')
            if representation(entry)['feature_tokens']:
                if entry.get('maskmem_features') is not None or entry.get('maskmem_pos_enc') is not None:
                    raise ValueError('History must be physically compressed before pooling')
                history.append((idx,entry))
        candidates=sum(representation(entry)['feature_tokens'] for _,entry in history)
        replacement=compact_stored_pool(history,capacity,self.strategy)
        apply_compacted_pool(history,replacement)
        kept=[x for x in replacement if len(x['indices'])];evicted=[x['frame'] for x in replacement if not len(x['indices'])]
        counts={str(x['frame']):len(x['indices']) for x in kept}
        id_hashes={str(idx):hashlib.sha256(entry['pool_token_ids'].cpu().contiguous().numpy().tobytes()).hexdigest()
                   for idx,entry in history if entry.get('pool_token_ids') is not None}
        score_bytes=sum(entry[k].numel()*entry[k].element_size() for _,entry in history
                        for k in ('pool_token_ids','pool_usage_rank','pool_contrast') if torch.is_tensor(entry.get(k)))
        total=cond+current+self.insurance_reserve+sum(counts.values())
        if total>self.budget:raise ValueError('Planned token budget exceeded')
        self.events.append({'stage':'admission_plan','frame':int(frame),'condition_tokens':cond,'insurance_reserve':self.insurance_reserve,
          'new_tokens':current,'selected_frames':[x['frame'] for x in kept]+[frame],'evict_frames':evicted,
          'planned_reserved_tokens':total,'budget':self.budget,'pool_budget':capacity,'pool_counts':counts,'pool_ids_sha256':id_hashes,
          'pool_candidates':candidates,'pool_kept':sum(counts.values()),
          'score_metadata_bytes':int(score_bytes),'strategy':self.strategy})

@contextmanager
def capture_pool_scores(model, observations):
    """Attach source-local scores before author compression; no GT input."""
    get_feature=model._get_image_feature;add_mask=model.add_new_mask;cache=model._cache_tarr_affinity_from_new_memory;dump=model._dump_retained_tokens_record
    state={'frame':None,'feature':None,'prototypes':None}
    def feature(self,inference_state,frame_idx,batch_size):
        out=get_feature(inference_state,frame_idx,batch_size)
        f=out[1]['backbone_fpn'][0]
        if batch_size!=1 or f.ndim!=4 or f.shape[0]!=1:raise ValueError('Unexpected early feature')
        state['frame']=int(frame_idx);state['feature']=f[0].detach().float()
        return out
    def mask(self,inference_state,frame_idx,obj_id,mask):
        if state['frame']!=int(frame_idx) or state['prototypes'] is not None:raise ValueError('Initial prompt state differs')
        state['prototypes']=reference_prototypes(state['feature'],mask)
        return add_mask(inference_state,frame_idx=frame_idx,obj_id=obj_id,mask=mask)
    def cache_scores(self,current_out,output_dict,frame_idx,obj_idx,is_init_cond_frame,run_mem_encoder,reverse):
        result=cache(current_out,output_dict,frame_idx,obj_idx,is_init_cond_frame,run_mem_encoder,reverse)
        memory=current_out.get('maskmem_features') if isinstance(current_out,dict) else None
        if not is_init_cond_frame and torch.is_tensor(memory):
            if state['prototypes'] is None or state['frame']!=int(frame_idx) or memory.shape[-2:]!=(64,64):raise ValueError('Pool score source is unavailable or misaligned')
            pos,neg=state['prototypes']
            with torch.no_grad():
                score=blend_anchor_with_early_contrast(torch.zeros(4096,device=state['feature'].device),state['feature'],pos,neg,(64,64),1.)
            current_out['pool_contrast_full']=score.detach().clamp(-1,1).cpu().clone()
        return result
    def source_selection(self,**kwargs):
        result=dump(**kwargs);entry=kwargs['latest_prev'];contrast=entry.get('pool_contrast_full')
        if not torch.is_tensor(contrast):raise ValueError('Selected source lacks early contrast')
        metadata=source_score_metadata(kwargs['usage_scores'],contrast,kwargs['keep_idx'])
        entry.update(metadata);entry['pool_contrast_full']=None
        observations.append({'frame':int(kwargs['frame_idx']),'source_frame':int(entry['frame_idx']),
          'tokens':int(metadata['pool_token_ids'].numel()),'ids_sha256':hashlib.sha256(metadata['pool_token_ids'].numpy().tobytes()).hexdigest()})
        return result
    model._get_image_feature=MethodType(feature,model);model.add_new_mask=MethodType(mask,model)
    model._cache_tarr_affinity_from_new_memory=MethodType(cache_scores,model);model._dump_retained_tokens_record=MethodType(source_selection,model)
    try:yield
    finally:
        model._get_image_feature=get_feature;model.add_new_mask=add_mask
        model._cache_tarr_affinity_from_new_memory=cache;model._dump_retained_tokens_record=dump
