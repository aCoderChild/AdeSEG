"""Three-region memory selection for Lean; experimental, inference-only.

This reuses the existing automatic-prompt contrast and author selection hooks.
The boundary prototype is estimated from the initial predicted mask, not labels.
It is a fixed cosine descriptor, not FBB-HFNet's trained attention module.
"""
from contextlib import contextmanager
from types import MethodType
import hashlib
import numpy as np
import torch
import torch.nn.functional as F

from modeling.cross_frame_token_pool import TokenGroup, select_global_tokens, apply_compacted_pool
from modeling.lean_cross_frame_pool import capture_pool_scores
from modeling.lean_token_storage_observer import representation
from modeling.lean_token_window_storage import TokenWindowRuntime


def boundary_prototype(features, mask):
    """Describe a two-cell soft morphological band around the prompt mask."""
    if features.ndim != 3 or not torch.isfinite(features).all():
        raise ValueError('Expected finite early feature grid')
    mask=torch.as_tensor(mask,device=features.device,dtype=torch.float32)
    if mask.ndim!=2 or not torch.isfinite(mask).all() or (mask<0).any() or (mask>1).any():
        raise ValueError('Expected a finite automatic mask in [0,1]')
    weights=F.interpolate(mask[None,None],size=features.shape[-2:],mode='area')
    dilated=F.max_pool2d(weights,5,stride=1,padding=2)
    eroded=-F.max_pool2d(-weights,5,stride=1,padding=2)
    band=(dilated-eroded)[0,0]
    if band.sum()<=0:raise ValueError('Prompt has no representable boundary')
    prototype=(features.float()*band).sum((1,2))/band.sum()
    if not torch.isfinite(prototype).all() or prototype.norm()==0:raise ValueError('Degenerate boundary prototype')
    return F.normalize(prototype,dim=0)


def boundary_scores(features, prototype, size=(64,64)):
    if features.ndim!=3 or prototype.shape!=(features.shape[0],) or not torch.isfinite(features).all() or not torch.isfinite(prototype).all():
        raise ValueError('Invalid boundary features')
    feature=F.normalize(features.float(),dim=0)
    score=torch.einsum('chw,c->hw',feature,prototype.float())[None,None]
    return F.interpolate(score,size=size,mode='area').reshape(-1).clamp(-1,1)


def region_indices(groups, boundaries, budget, strategy):
    """Select exact capacity; three fixed quotas use disjoint token sets."""
    if strategy!='three_region':
        return select_global_tokens(groups,budget,strategy)
    # Base selector performs all geometry, ordering and score validation.
    select_global_tokens(groups,0,'usage')
    if len(groups)!=len(boundaries) or type(budget) is not int or budget<0:
        raise ValueError('Invalid region pool')
    rankings=[[],[],[]]
    for gi,(group,b) in enumerate(zip(groups,boundaries)):
        b=np.asarray(b);u=np.asarray(group.usage_rank);c=np.asarray(group.contrast)
        if b.shape!=u.shape or not np.isfinite(b).all() or (b< -1).any() or (b>1).any():raise ValueError('Invalid boundary score')
        for j,(usage,contrast,boundary) in enumerate(zip(u,c,b)):
            for ranking,value in zip(rankings,(contrast,-contrast,boundary)):
                ranking.append((float(.5*usage+.25*(value+1)),gi,j))
    for ranking in rankings:ranking.sort(key=lambda x:-x[0])
    count=min(budget,sum(len(g.token_ids) for g in groups))
    quotas=[count//3+(i<count%3) for i in range(3)]
    used=set();chosen=[]
    for ranking,quota in zip(rankings,quotas):
        added=0
        for _,gi,j in ranking:
            if added==quota:break
            item=(gi,j)
            if item not in used:
                used.add(item);chosen.append(item);added+=1
        if added!=quota:raise ValueError('Region quota could not be filled')
    per=[[] for _ in groups]
    for gi,j in chosen:per[gi].append(j)
    return [np.asarray(sorted(x),dtype=np.int64) for x in per]


def compact_region_pool(entries,budget,strategy):
    groups=[];boundaries=[]
    for frame,e in entries:
        f=e.get('maskmem_features_seq');p=e.get('maskmem_pos_seq')
        ids=e.get('pool_token_ids');u=e.get('pool_usage_rank');c=e.get('pool_contrast');b=e.get('pool_boundary')
        if not all(torch.is_tensor(x) for x in (f,p,ids,u,c,b)) or f.ndim!=3 or p.ndim!=3 or f.shape[1]!=1 or p.shape[1]!=1:
            raise ValueError('Invalid compressed memory')
        n=f.shape[0]
        if p.shape[0]!=n or ids.dtype!=torch.long or any(x.ndim!=1 or x.numel()!=n for x in (ids,u,c,b)) or not torch.isfinite(f).all() or not torch.isfinite(p).all():
            raise ValueError('Memory and metadata are not aligned')
        groups.append(TokenGroup(frame,ids.cpu().numpy(),u.cpu().numpy(),c.cpu().numpy()));boundaries.append(b.cpu().numpy())
    indices=region_indices(groups,boundaries,budget,strategy)
    replacements=[];new_boundary=[]
    for (frame,e),idx in zip(entries,indices):
        tensors={}
        for k in ('maskmem_features_seq','maskmem_pos_seq','pool_token_ids','pool_usage_rank','pool_contrast'):
            tensors[k]=e[k].index_select(0,torch.as_tensor(idx,device=e[k].device))
        replacements.append({'frame':frame,'indices':idx,'tensors':tensors})
        new_boundary.append(e['pool_boundary'].index_select(0,torch.as_tensor(idx,device=e['pool_boundary'].device)))
    # Base commit independently validates all features, positions, IDs and scores.
    apply_compacted_pool(entries,replacements)
    for (_,e),idx,b in zip(entries,indices,new_boundary):e['pool_boundary']=b if len(idx) else None
    return replacements


@contextmanager
def capture_region_scores(model,observations):
    """Compose with the previously verified source-score observer."""
    with capture_pool_scores(model,observations):
        get_feature=model._get_image_feature;add_mask=model.add_new_mask
        cache=model._cache_tarr_affinity_from_new_memory;dump=model._dump_retained_tokens_record
        state={'frame':None,'feature':None,'boundary':None}
        def feature(self,inference_state,frame_idx,batch_size):
            out=get_feature(inference_state,frame_idx,batch_size)
            state['frame']=int(frame_idx);state['feature']=out[1]['backbone_fpn'][0][0].detach().float()
            return out
        def mask(self,inference_state,frame_idx,obj_id,mask):
            if state['frame']!=int(frame_idx) or state['boundary'] is not None:raise ValueError('Prompt and boundary source differ')
            state['boundary']=boundary_prototype(state['feature'],mask)
            return add_mask(inference_state,frame_idx=frame_idx,obj_id=obj_id,mask=mask)
        def cached(self,current_out,output_dict,frame_idx,obj_idx,is_init_cond_frame,run_mem_encoder,reverse):
            result=cache(current_out,output_dict,frame_idx,obj_idx,is_init_cond_frame,run_mem_encoder,reverse)
            if not is_init_cond_frame and isinstance(current_out,dict) and torch.is_tensor(current_out.get('maskmem_features')):
                if state['boundary'] is None or state['frame']!=int(frame_idx):raise ValueError('Boundary source differs')
                current_out['pool_boundary_full']=boundary_scores(state['feature'],state['boundary']).detach().cpu().clone()
            return result
        def selected(self,**kwargs):
            result=dump(**kwargs);entry=kwargs['latest_prev'];full=entry.get('pool_boundary_full');ids=entry['pool_token_ids']
            if not torch.is_tensor(full) or full.shape!=(4096,):raise ValueError('Missing boundary score grid')
            entry['pool_boundary']=full[ids].clone();entry['pool_boundary_full']=None
            observations[-1]['boundary_sha256']=hashlib.sha256(entry['pool_boundary'].numpy().tobytes()).hexdigest()
            return result
        model._get_image_feature=MethodType(feature,model);model.add_new_mask=MethodType(mask,model)
        model._cache_tarr_affinity_from_new_memory=MethodType(cached,model);model._dump_retained_tokens_record=MethodType(selected,model)
        try:yield
        finally:
            model._get_image_feature=get_feature;model.add_new_mask=add_mask
            model._cache_tarr_affinity_from_new_memory=cache;model._dump_retained_tokens_record=dump


class RegionPoolRuntime(TokenWindowRuntime):
    def __init__(self,budget,insurance_reserve,strategy):
        super().__init__(budget,insurance_reserve)
        if strategy not in ('three_region','usage','contrast','distributed'):raise ValueError('Unknown region strategy')
        self.strategy=strategy

    def admit(self,model,output_dict,inference_state,obj_idx,frame,current_out):
        self.verify(output_dict,inference_state,obj_idx,'before_admission',frame)
        noncond=output_dict.get('non_cond_frame_outputs',{})
        for entry in noncond.values():
            if entry.get('smr_state')=='pending':
                ids=entry.get('pool_token_ids');keep=entry.get('pending_keep_idx');feat=entry.get('maskmem_features');pos=entry.get('maskmem_pos_enc')
                if not all(torch.is_tensor(x) for x in (ids,keep,feat,entry.get('pool_boundary'))) or pos is None or not torch.equal(ids.cpu().long(),keep.cpu().long()):raise ValueError('Pending identity differs')
                pos=pos[-1] if isinstance(pos,(list,tuple)) else pos
                f=feat.flatten(2).permute(2,0,1).index_select(0,keep.to(feat.device).long())
                p=pos.flatten(2).permute(2,0,1).index_select(0,keep.to(pos.device).long())
                model._compress_memory_entry(entry)
                if not torch.equal(entry['maskmem_features_seq'],f) or not torch.equal(entry['maskmem_pos_seq'],p):raise ValueError('Author compression misaligned')
        self.verify(output_dict,inference_state,obj_idx,'after_compression',frame)
        cond=sum(representation(e)['feature_tokens'] for e in output_dict.get('cond_frame_outputs',{}).values())
        current=representation(current_out)['feature_tokens']
        if cond!=4096 or current!=4096 or frame in noncond:raise ValueError('Protected memory changed')
        cap=self.budget-self.insurance_reserve-cond-current
        if cap<0:raise ValueError('Protected memories do not fit')
        history=[]
        for idx,e in sorted(noncond.items()):
            if type(idx) is not int or idx>=frame:raise ValueError('Expected forward history')
            if representation(e)['feature_tokens']:
                if e.get('maskmem_features') is not None or e.get('maskmem_pos_enc') is not None:raise ValueError('History not compressed')
                history.append((idx,e))
        candidates=sum(representation(e)['feature_tokens'] for _,e in history)
        replacements=compact_region_pool(history,cap,self.strategy)
        counts={str(x['frame']):len(x['indices']) for x in replacements if len(x['indices'])}
        id_hashes={str(idx):hashlib.sha256(e['pool_token_ids'].cpu().contiguous().numpy().tobytes()).hexdigest() for idx,e in history if e.get('pool_token_ids') is not None}
        metadata=sum(e[k].numel()*e[k].element_size() for _,e in history for k in ('pool_token_ids','pool_usage_rank','pool_contrast','pool_boundary') if torch.is_tensor(e.get(k)))
        kept=sum(counts.values());total=cond+current+self.insurance_reserve+kept
        if total>self.budget or kept!=min(cap,candidates):raise ValueError('Pool capacity differs')
        self.events.append({'stage':'admission_plan','frame':int(frame),'condition_tokens':cond,'insurance_reserve':self.insurance_reserve,'new_tokens':current,
          'selected_frames':[x['frame'] for x in replacements if len(x['indices'])]+[frame], 'evict_frames':[x['frame'] for x in replacements if not len(x['indices'])],
          'planned_reserved_tokens':total,'budget':self.budget,'pool_budget':cap,'pool_counts':counts,'pool_ids_sha256':id_hashes,'pool_candidates':candidates,'pool_kept':kept,'score_metadata_bytes':int(metadata),'strategy':self.strategy})
