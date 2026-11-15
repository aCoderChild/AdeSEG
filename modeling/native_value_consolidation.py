"""Training-free diagnostic adapter around official ToMe merge/unmerge.

Keep the native source schedule and dense queries/keys. Consolidate values
inside each past frame's local windows, then restore original member slots.
This is an accuracy hypothesis, not a token-count or speed improvement.
"""
import math
import torch
import torch.nn.functional as F
from modeling.region_constrained_tome import NativeRegionConstrainedToMe, author_tome
from modeling.attention_distortion_observer import _sha256_tensor


def consolidate_values(keys, values, spans, *, pointers, mode="value", window_size=4):
    if mode not in ("noop", "value", "joint"):
        raise ValueError("Expected noop, value or joint consolidation")
    if keys.shape != values.shape or keys.ndim != 4 or keys.shape[0] != 1:
        raise ValueError("Expected one-object aligned [B,H,N,D] K/V")
    if not 0 <= pointers < keys.shape[-2]:
        raise ValueError("Invalid pointer tail")
    if not torch.isfinite(keys).all() or not torch.isfinite(values).all():
        raise ValueError("K/V must be finite")
    if mode == "noop":
        return keys, values, {"groups":0,"mode":mode,"token_count":keys.shape[-2]}
    changed_key, changed_value = keys.clone(), values.clone()
    tome = author_tome()
    cursor, groups = 0, 0
    for span in spans:
        start, stop = int(span['start']), int(span['stop'])
        side = math.isqrt(stop-start)
        if start != cursor or side*side != stop-start or side % window_size:
            raise ValueError("Expected contiguous square tiled memory maps")
        cursor = stop
        if int(span['t_pos']) == 0:
            continue
        grid = torch.arange(start,stop,device=keys.device).reshape(side,side)
        for top in range(0,side,window_size):
            for left in range(0,side,window_size):
                indices = grid[top:top+window_size,left:left+window_size].flatten()
                local_key,local_value = keys.index_select(-2,indices),values.index_select(-2,indices)
                metric = F.normalize(local_key.float().mean(dim=1),dim=-1)
                merge,unmerge = tome.bipartite_soft_matching(metric,indices.numel()//4)
                pooled_values,pooled_keys = [],[]
                for head in range(keys.shape[1]):
                    value_mean,_ = tome.merge_wavg(merge,local_value[:,head])
                    pooled_values.append(unmerge(value_mean))
                    if mode == 'joint':
                        key_mean,_ = tome.merge_wavg(merge,local_key[:,head])
                        pooled_keys.append(unmerge(key_mean))
                changed_value.index_copy_(-2,indices,torch.stack(pooled_values,dim=1))
                if mode == 'joint':
                    changed_key.index_copy_(-2,indices,torch.stack(pooled_keys,dim=1))
                groups += 1
    if cursor != keys.shape[-2]-pointers:
        raise ValueError("Memory spans do not cover the spatial tokens")
    if pointers and (not torch.equal(changed_key[...,-pointers:,:],keys[...,-pointers:,:])
                     or not torch.equal(changed_value[...,-pointers:,:],values[...,-pointers:,:])):
        raise RuntimeError("Consolidation changed pointers")
    return changed_key,changed_value,{"groups":groups,"mode":mode,"token_count":keys.shape[-2]}


class NativeValueConsolidation(NativeRegionConstrainedToMe):
    """Read-only ToMe adaptation; no native time-slot extension or writes."""
    def __init__(self,model,*,mode="value"):
        super().__init__(model,mode='global')
        self.consolidation_mode = mode

    def _sdpa(self,query,key,value,*args,**kwargs):
        if self.inside or self.active is None:
            return self.original_sdpa(query,key,value,*args,**kwargs)
        if args or kwargs.get('attn_mask') is not None or kwargs.get('dropout_p',0)!=0 or kwargs.get('is_causal',False) or kwargs.get('scale') is not None or kwargs.get('enable_gqa',False):
            raise RuntimeError("Unsupported consolidation SDPA options")
        layer,pointers = self.active
        before = [_sha256_tensor(x) for x in (query,key,value)]
        k,v,audit = consolidate_values(key,value,self.spans,pointers=pointers,
                                       mode=self.consolidation_mode)
        if self.consolidation_mode in ('noop','value') and not torch.equal(k,key):
            raise RuntimeError("Value consolidation changed native keys")
        output = self.original_sdpa(query,k,v,*args,**kwargs)
        if before != [_sha256_tensor(x) for x in (query,key,value)]:
            raise RuntimeError("Consolidation mutated source Q/K/V")
        self.calls.append({'frame':self.frame,'layer':layer,'pointer_keys':pointers,
                           'qkv_input_sha256':before,'key_unchanged':torch.equal(k,key),
                           'frames':[{'frame':x['frame'],'t_pos':x['t_pos'],
                                      'source_tokens':x['stop']-x['start'],
                                      'target_tokens':x['stop']-x['start'],
                                      'merged_tokens':0} for x in self.spans],
                           'spatial_keys_before':key.shape[-2]-pointers,
                           'spatial_keys_after':key.shape[-2]-pointers,
                           'spatial_key_mass_sum':float(key.shape[-2]-pointers),
                           'key_mass_sum':float(key.shape[-2]),
                           'source_inputs_unchanged':True,**audit})
        return output
