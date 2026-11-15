#!/usr/bin/env python3
"""Observe actual key positions while exactly replaying the corrected Lean run."""
import argparse
from collections import Counter
import hashlib
import json
import math
from pathlib import Path
import sys
from types import MethodType
import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.lean_memory_position_lookup import MemoryPositionLookup
from scripts.memory_token_io import digest
import scripts.run_lean_rope_cpu_fix as runner


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','merged','historical','replay-a','replay-b','fixed-a','fixed-b','reference','output'):
        p.add_argument('--'+name,type=Path,required=True)
    p.add_argument('--sequence',required=True)
    a=p.parse_args()
    if not (a.reference/'completion.json').exists(): raise ValueError('Missing fixed reference')
    observations=[]
    build=runner.build_author_predictor
    def build_observed(args, repo):
        model=build(args,repo)
        if any(layer.toma_drop_ratio_ca_kv != 0 for layer in model.memory_attention.layers):
            raise ValueError('Position diagnostic does not support cross-attention merging')
        if model.device.type != 'cpu': raise ValueError('Position diagnostic requires CPU')
        with torch.no_grad():
            grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64)).flatten(2).permute(2,0,1)[:,0,:].contiguous()
            offsets=model.maskmem_tpos_enc.detach().reshape(-1,model.mem_dim).cpu()
            lookup=MemoryPositionLookup(grid,offsets)
        attn=model.memory_attention.layers[0].cross_attn_image
        correct=attn.compute_cis(end_x=64,end_y=64).cpu()
        original=model.memory_attention.forward
        def observed(self, *args, **kwargs):
            curr=kwargs['curr']; curr=curr[0] if isinstance(curr,list) else curr
            memory_pos=kwargs['memory_pos']; ptr=int(kwargs['num_obj_ptr_tokens'])
            if curr.shape[0] != 4096: raise ValueError('Unexpected query grid')
            positions=lookup.decode(memory_pos,ptr)
            if len(positions) == 0: raise ValueError('No spatial memory')
            layer=self.layers[0]
            base=layer.ca_rope_base_len_override
            base=int(base) if base is not None else int(round(curr.shape[0]*(1.-float(layer.toma_drop_ratio_ca_kv))))
            base=max(1,min(base,curr.shape[0]))
            if curr.shape[0]%base or len(positions)%base:
                from sam2.modeling.memory_attention import _largest_square_divisor_leq
                base=_largest_square_divisor_leq(math.gcd(curr.shape[0],len(positions)),base)
            side=math.isqrt(base)
            if side*side != base: raise ValueError('Unexpected nonsquare fallback')
            freq=attn.compute_cis(end_x=side,end_y=side).cpu()
            q_old=freq.repeat(curr.shape[0]//base,1)
            k_old=freq.repeat(len(positions)//base,1)
            q_matches=int(torch.all(torch.isclose(q_old,correct,atol=1e-6,rtol=1e-6),dim=1).sum())
            k_matches=int(torch.all(torch.isclose(k_old,correct[positions],atol=1e-6,rtol=1e-6),dim=1).sum())
            observations.append({'spatial_keys':len(positions),'pointer_keys':ptr,'base':base,'query_frequency_matches':q_matches,
                'key_frequency_matches':k_matches,'position_indices_sha256':hashlib.sha256(positions.numpy().tobytes()).hexdigest(),
                'unique_spatial_indices':int(torch.unique(positions).numel())})
            return original(*args,**kwargs)
        model.memory_attention.forward=MethodType(observed,model.memory_attention)
        return model
    runner.build_author_predictor=build_observed
    args=['run_lean_rope_cpu_fix.py','--mode','fixed','--sequence',a.sequence]
    for name in ('checkpoint','merged','historical','replay_a','replay_b','fixed_a','fixed_b','output'):
        args.extend(['--'+name.replace('_','-'),str(getattr(a,name))])
    sys.argv=args
    runner.main()
    observed=json.loads((a.output/'records.json').read_text()); reference=json.loads((a.reference/'records.json').read_text())
    for key in ('mask_sha256','selections_sha256','frequency_trace_sha256','memory_events_sha256','frames_sha256'):
        if observed[key] != reference[key]: raise ValueError(f'Observer changed fixed run: {key}')
    summary={'sequence':a.sequence,'passed':True,'input':'CPU FP32, exact position-grid and temporal-offset lookup',
             'source_sha256':digest(Path(__file__)),'lookup_sha256':digest(ROOT/'modeling/lean_memory_position_lookup.py'),
             'reference_records_sha256':digest(a.reference/'records.json'), 'observed_records_sha256':digest(a.output/'records.json'),
             'attention_calls':len(observations),'query_tokens':4096*len(observations),
             'query_frequency_matches':sum(x['query_frequency_matches'] for x in observations),
             'key_tokens':sum(x['spatial_keys'] for x in observations),
             'key_frequency_matches':sum(x['key_frequency_matches'] for x in observations),
             'bases':dict(Counter(str(x['base']) for x in observations)), 'observations':observations}
    if not observations: raise ValueError('No attention calls observed')
    (a.output/'position_observation.json').write_text(json.dumps(summary,indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k!='observations'},indent=2))


if __name__=='__main__': main()
