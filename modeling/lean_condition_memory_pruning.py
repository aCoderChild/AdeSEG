"""Fixed-budget read-time selection of Lean's original conditioning grid.

The saved conditioning memory remains intact. This adapter uses the author's
already-computed first-read attention usage or a fixed spatial control, and
keeps features, positions, and indexed rotary coordinates aligned.
"""
from contextlib import contextmanager
from types import MethodType
import hashlib
import torch


def choose_condition_indices(usage, strategy):
    if usage.ndim != 1 or usage.numel() != 4096 or not torch.isfinite(usage).all() or (usage < 0).any():
        raise ValueError('Expected finite nonnegative usage on a 64x64 grid')
    if strategy == 'full':
        return torch.arange(4096, device=usage.device)
    if strategy == 'usage':
        return torch.topk(usage, 1024).indices
    if strategy == 'uniform':
        return torch.arange(4096, device=usage.device).reshape(64,64)[::2,::2].reshape(-1)
    raise ValueError('Unknown conditioning selection strategy')


@contextmanager
def condition_memory_pruning(model, strategy, observations):
    if strategy not in ('full','usage','uniform'):
        raise ValueError('Unknown conditioning selection strategy')
    get_feature=model._get_image_feature
    materialize=model._materialize_memory_entry
    attention=model.memory_attention.forward
    with torch.no_grad():
        grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64)).flatten(2).permute(2,0,1).contiguous()
        expected_pos=grid+model.maskmem_tpos_enc[model.num_maskmem-1].detach()
    state={'frame':None,'indices':None,'stored_features':None,'stored_positions':None}

    def feature(self,inference_state,frame_idx,batch_size):
        result=get_feature(inference_state,frame_idx,batch_size)
        state['frame']=int(frame_idx)
        return result

    def read_memory(self,prev,device,*,is_cond=False):
        features,positions=materialize(prev,device,is_cond=is_cond)
        if not is_cond or features is None or positions is None:
            return features,positions
        if features.shape[0]!=4096 or positions.shape[0]!=4096 or features.shape[1]!=1:
            raise ValueError('Expected a single full conditioning memory grid')
        if state['stored_features'] is None:
            state['stored_features']=features.detach().clone()
            state['stored_positions']=positions.detach().clone()
        elif not torch.equal(features,state['stored_features']) or not torch.equal(positions,state['stored_positions']):
            raise ValueError('Stored conditioning features or positions changed')
        if state['indices'] is None or strategy=='full':
            observations.append({'event':'conditioning_read','frame':state['frame'],'tokens':4096})
            return features,positions
        idx=state['indices'].to(features.device)
        if idx.numel()!=1024 or torch.unique(idx).numel()!=1024:
            raise ValueError('Invalid selected conditioning coordinates')
        # No tensor on the model's storage is mutated.
        observations.append({'event':'conditioning_read','frame':state['frame'],'tokens':1024})
        return features[idx],positions[idx]

    def observe_attention(self,*args,**kwargs):
        result=attention(*args,**kwargs)
        if state['indices'] is None:
            if not kwargs.get('return_attn_stat') or not isinstance(result,tuple) or result[1] is None:
                raise ValueError('First-read author attention usage unavailable')
            usage=result[1].detach();pos=kwargs['memory_pos']
            if state['frame'] is None or pos.shape[1]!=1 or pos.shape[0]!=usage.numel() or not torch.equal(pos[:4096],expected_pos):
                raise ValueError('First-read conditioning grid differs')
            indices=choose_condition_indices(usage[:4096],strategy).detach().clone()
            state['indices']=indices
            selected=indices.cpu().contiguous().numpy()
            observations.append({'event':'selection','frame':state['frame'],'strategy':strategy,'original_tokens':4096,'retained_tokens':int(indices.numel()),
              'selected_indices_sha256':hashlib.sha256(selected.tobytes()).hexdigest(),
              'selected_indices':selected.tolist(), 'first_read_usage':usage[:4096].cpu().tolist(),
              'selected_usage_fraction_of_conditioning':float((usage[indices].sum()/usage[:4096].sum()).item()),
              'pointer_tokens':int(kwargs['num_obj_ptr_tokens'])})
        return result

    model._get_image_feature=MethodType(feature,model)
    model._materialize_memory_entry=MethodType(read_memory,model)
    model.memory_attention.forward=MethodType(observe_attention,model.memory_attention)
    try:
        yield
    finally:
        model._get_image_feature=get_feature
        model._materialize_memory_entry=materialize
        model.memory_attention.forward=attention
