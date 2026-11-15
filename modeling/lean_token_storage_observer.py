"""Read-only accounting of Lean's stored spatial memory representations."""
from types import MethodType
import torch

def representation(entry):
    result={'feature_tokens':0,'position_tokens':0,'representations':[],'state':entry.get('smr_state'),
            'pending_indices':int(entry['pending_keep_idx'].numel()) if torch.is_tensor(entry.get('pending_keep_idx')) else 0,
            'object_pointer_elements':int(entry['obj_ptr'].numel()) if torch.is_tensor(entry.get('obj_ptr')) else 0}
    dense=entry.get('maskmem_features'); densepos=entry.get('maskmem_pos_enc')
    seq=entry.get('maskmem_features_seq');seqpos=entry.get('maskmem_pos_seq')
    if dense is not None:
        if not torch.is_tensor(dense) or dense.ndim!=4:raise ValueError('Unexpected dense memory')
        result['feature_tokens']+=int(dense.shape[-2]*dense.shape[-1]);result['representations'].append('dense')
    if densepos is not None:
        pos=densepos[-1] if isinstance(densepos,(list,tuple)) else densepos
        if not torch.is_tensor(pos) or pos.ndim!=4:raise ValueError('Unexpected dense positions')
        result['position_tokens']+=int(pos.shape[-2]*pos.shape[-1])
    if seq is not None:
        if not torch.is_tensor(seq) or seq.ndim!=3:raise ValueError('Unexpected compressed memory')
        result['feature_tokens']+=int(seq.shape[0]);result['representations'].append('compressed')
    if seqpos is not None:
        if not torch.is_tensor(seqpos) or seqpos.ndim!=3:raise ValueError('Unexpected compressed positions')
        result['position_tokens']+=int(seqpos.shape[0])
    if result['feature_tokens']!=result['position_tokens']:raise ValueError('Memory feature/position count differs')
    return result

def snapshot(model,output_dict,inference_state,obj_idx):
    entries=[]
    for group in ('cond_frame_outputs','non_cond_frame_outputs'):
        for index,entry in sorted(output_dict.get(group,{}).items()):
            item={'frame':int(index),'group':group,**representation(entry)}
            item['author_logical_tokens']=int(model._profile_memory_entry(entry)['actual_tokens'])
            entries.append(item)
    bank=[] if inference_state is None or obj_idx is None else inference_state.get('insurance_mem_bank',{}).get(obj_idx,[]) or []
    insurance=[]
    for item in bank:
        feat=item.get('mem_seq');pos=item.get('pos_seq')
        if not torch.is_tensor(feat) or not torch.is_tensor(pos) or feat.shape[0]!=pos.shape[0]:raise ValueError('Insurance geometry differs')
        insurance.append(int(feat.shape[0]))
    return {'entries':entries,'spatial_feature_tokens':sum(x['feature_tokens'] for x in entries),
            'logical_spatial_tokens':sum(x['author_logical_tokens'] for x in entries),
            'insurance_feature_tokens':sum(insurance),'insurance_entries':insurance,
            'object_pointer_elements':sum(x['object_pointer_elements'] for x in entries)}

def install_storage_observer(model,events):
    original=model._prepare_memory_conditioned_features
    def observed(self,*args,**kwargs):
        if args:raise ValueError('Unexpected positional memory call')
        before=snapshot(self,kwargs['output_dict'],kwargs.get('inference_state'),kwargs.get('obj_idx'))
        result=original(*args,**kwargs)
        after=snapshot(self,kwargs['output_dict'],kwargs.get('inference_state'),kwargs.get('obj_idx'))
        events.append({'frame':int(kwargs['frame_idx']),'initial':bool(kwargs['is_init_cond_frame']),'before':before,'after':after})
        return result
    model._prepare_memory_conditioned_features=MethodType(observed,model)
    return original
