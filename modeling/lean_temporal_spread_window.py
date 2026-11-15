"""Budgeted temporal-coverage control using the existing Lean storage runtime."""
from modeling.lean_token_window_storage import TokenWindowRuntime, apply_eviction
from modeling.lean_token_storage_observer import representation


def select_spread(condition_frame, condition_tokens, history, budget, reserve):
    """Select whole resident entries; preserve condition and newest entry.

    Evict the interior entry whose neighbors span the shortest time interval.
    On equal spans evict the more recent entry. The retained set is online:
    previously evicted entries cannot be recovered.
    """
    if any(type(x) is not int for x in (condition_frame,condition_tokens,budget,reserve)) or condition_tokens<=0 or reserve<0:
        raise ValueError('Invalid protected memory or budget')
    if not history or any(type(i) is not int or type(n) is not int or n<=0 for i,n in history):
        raise ValueError('Invalid history')
    if history[0][0]<=condition_frame or any(x[0]>=y[0] for x,y in zip(history,history[1:])):
        raise ValueError('History must follow the condition in strict order')
    if condition_tokens+reserve+history[-1][1]>budget:
        raise ValueError('Budget cannot hold condition, reserve and newest')
    kept=list(history);removed=[]
    used=condition_tokens+reserve+sum(n for _,n in kept)
    while used>budget:
        # Newest is protected; condition is the immutable left boundary.
        candidates=[]
        for j,(frame,_) in enumerate(kept[:-1]):
            left=condition_frame if j==0 else kept[j-1][0]
            right=kept[j+1][0]
            candidates.append((right-left,-frame,j))
        _,_,index=min(candidates)
        frame,count=kept.pop(index);removed.append(frame);used-=count
    return kept,removed,used


def plan_spread(output_dict,current_index,current_out,budget,reserve):
    cond=output_dict.get('cond_frame_outputs',{});noncond=output_dict.get('non_cond_frame_outputs',{})
    if len(cond)!=1 or current_index in cond or current_index in noncond:
        raise ValueError('Exactly one separate condition is required')
    condition_frame,condition=next(iter(cond.items()))
    condition_tokens=representation(condition)['feature_tokens'];current_count=representation(current_out)['feature_tokens']
    history=[]
    for idx,entry in sorted(noncond.items()):
        if type(idx) is not int or idx>=current_index:raise ValueError('Non-forward memory')
        count=representation(entry)['feature_tokens']
        if count:history.append((idx,count))
    history.append((current_index,current_count))
    selected,evicted,used=select_spread(condition_frame,condition_tokens,history,budget,reserve)
    return {'condition_frame':condition_frame,'condition_tokens':condition_tokens,'insurance_reserve':reserve,'new_tokens':current_count,
            'candidate_entries':[{'frame':i,'tokens':n} for i,n in history], 'selected_frames':[i for i,_ in selected],
            'evict_frames':evicted,'planned_reserved_tokens':used,'budget':budget}


class TemporalSpreadRuntime(TokenWindowRuntime):
    def admit(self,model,output_dict,inference_state,obj_idx,frame,current_out):
        self.verify(output_dict,inference_state,obj_idx,'before_admission',frame)
        for entry in output_dict.get('non_cond_frame_outputs',{}).values():
            if entry.get('smr_state')=='pending':model._compress_memory_entry(entry)
        self.verify(output_dict,inference_state,obj_idx,'after_compression',frame)
        plan=plan_spread(output_dict,frame,current_out,self.budget,self.insurance_reserve)
        apply_eviction(output_dict,plan)
        self.events.append({'stage':'admission_plan','frame':int(frame),**plan})
        return plan
