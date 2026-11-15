"""Experimental spatial-token admission for Lean's forward-only memory.

This module does not itself install hooks or modify author code. It counts
feature representations and reserves insurance capacity; positional tensors,
pointer records, masks, caches, auxiliary state and transient allocations are
outside its budget. It preserves prediction outputs and object pointers.
"""
from modeling.lean_token_storage_observer import representation
from modeling.token_memory_window import SpatialMemory, select_token_window

SPATIAL_FIELDS=('maskmem_features','maskmem_pos_enc','maskmem_features_seq','maskmem_pos_seq')

def temporal_slot(current_frame, source_frame, num_slots):
    """Use the author's learned slot by actual age; clamp older ages to slot 1."""
    if any(type(x) is not int for x in (current_frame,source_frame,num_slots)) or num_slots<2 or source_frame>=current_frame:
        raise ValueError('Invalid forward temporal slot')
    return num_slots-min(current_frame-source_frame,num_slots-1)

def plan_admission(model, output_dict, current_index, current_out, budget, insurance_reserve):
    """Plan an admission using actual resident tensors without mutating them.

    The current output must not yet be in the dictionary. Pending entries are
    accounted as their current dense representation; a caller may separately
    invoke the author's compression before constructing a new plan.
    """
    if type(current_index) is not int: raise ValueError('Invalid frame index')
    cond=output_dict.get('cond_frame_outputs',{})
    noncond=output_dict.get('non_cond_frame_outputs',{})
    if current_index in noncond or current_index in cond:raise ValueError('Frame already stored')
    if not cond:raise ValueError('Missing protected condition memory')
    cond_count=sum(representation(entry)['feature_tokens'] for entry in cond.values())
    current_count=representation(current_out)['feature_tokens']
    if current_count<=0:raise ValueError('Current memory must contain feature-position pairs')
    history=[]
    for idx,entry in sorted(noncond.items()):
        if type(idx) is not int or idx>=current_index:raise ValueError('Only forward chronological state is supported')
        count=representation(entry)['feature_tokens']
        if count:history.append(SpatialMemory(idx,count))
    history.append(SpatialMemory(current_index,current_count))
    selected,used=select_token_window(cond_count,history,budget,reserved_tokens=insurance_reserve)
    selected_ids={entry.frame_index for entry in selected}
    evict=[entry.frame_index for entry in history[:-1] if entry.frame_index not in selected_ids]
    return {'condition_tokens':cond_count,'insurance_reserve':insurance_reserve,'new_tokens':current_count,
            'selected_frames':[entry.frame_index for entry in selected],'evict_frames':evict,
            'planned_reserved_tokens':used,'budget':budget}

def apply_eviction(output_dict, plan):
    """Remove only spatial memory for planned old entries; keep predictions and pointers."""
    noncond=output_dict['non_cond_frame_outputs']
    for idx in plan['evict_frames']:
        entry=noncond[idx]
        if entry.get('smr_protected',False):raise ValueError('Cannot evict protected memory')
        for field in SPATIAL_FIELDS:entry[field]=None
        entry['pending_keep_idx']=None
        entry['smr_state']='budget_evicted'
        # These caches are derived from the removed spatial memory.
        entry['anchor_similarity_cache']=None
        entry['token_window_evicted']=True

def count_with_insurance(output_dict, bank):
    spatial=sum(representation(entry)['feature_tokens'] for group in ('cond_frame_outputs','non_cond_frame_outputs') for entry in output_dict.get(group,{}).values())
    insurance=0
    for entry in bank:
        feat=entry.get('mem_seq');pos=entry.get('pos_seq')
        if feat is None or pos is None or feat.ndim!=3 or pos.ndim!=3 or feat.shape[0]!=pos.shape[0]:raise ValueError('Invalid insurance entry')
        insurance+=int(feat.shape[0])
    return {'spatial_tokens':spatial,'insurance_tokens':insurance,'total_tokens':spatial+insurance}

class TokenWindowRuntime:
    """Forward-only experimental admission, called explicitly by patched Lean.

    The reserve must bound the entire possible insurance bank, not only its
    present contents; otherwise an author bank write could exceed the budget.
    """
    def __init__(self,budget,insurance_reserve):
        if type(budget) is not int or type(insurance_reserve) is not int or budget<=0 or insurance_reserve<0:
            raise ValueError('Invalid token window capacity')
        self.budget=budget;self.insurance_reserve=insurance_reserve;self.events=[]

    def verify(self,output_dict,inference_state,obj_idx,stage,frame):
        bank=inference_state.get('insurance_mem_bank',{}).get(obj_idx,[]) or []
        count=count_with_insurance(output_dict,bank)
        if count['insurance_tokens']>self.insurance_reserve:
            raise ValueError('Insurance bank exceeds reserved token capacity')
        if count['spatial_tokens']+self.insurance_reserve>self.budget:
            raise ValueError('Spatial memory consumes protected insurance reserve')
        if count['total_tokens']>self.budget:
            raise ValueError('Persistent spatial-token budget exceeded')
        self.events.append({'stage':stage,'frame':int(frame),**count})
        return count

    def admit(self,model,output_dict,inference_state,obj_idx,frame,current_out):
        """Compress pending history using the author method, then evict before insertion."""
        self.verify(output_dict,inference_state,obj_idx,'before_admission',frame)
        for entry in output_dict.get('non_cond_frame_outputs',{}).values():
            if entry.get('smr_state')=='pending':model._compress_memory_entry(entry)
        self.verify(output_dict,inference_state,obj_idx,'after_compression',frame)
        plan=plan_admission(model,output_dict,frame,current_out,self.budget,self.insurance_reserve)
        apply_eviction(output_dict,plan)
        self.events.append({'stage':'admission_plan','frame':int(frame),**plan})
        # The caller must insert the current output immediately and then verify.
        return plan
