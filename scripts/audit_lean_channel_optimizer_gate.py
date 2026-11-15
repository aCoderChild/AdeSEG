#!/usr/bin/env python3
"""Verify repeated synthetic-target optimizer checks and block fixture use."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_attention import PromptContrastChannelAttention
from modeling.lean_channel_training import build_trainable_gate,parameter_hash
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for key in ('first','second','output'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();results=[]
    for folder in (a.first,a.second):
        r=read(folder/'verification.json')
        if not r['passed'] or r['scope']!='Synthetic target with one real feature; engineering only and no usable trained checkpoint.' or r['runner_sha256']!=digest(ROOT/'scripts/verify_lean_channel_optimizer_gate.py') or r['adapter_sha256']!=digest(ROOT/'modeling/lean_channel_attention.py') or r['training_sha256']!=digest(ROOT/'modeling/lean_channel_training.py') or r['loss_sha256']!=digest(ROOT/'modeling/lean_channel_training_loss.py') or r['optimization_protocol_sha256']!=digest(ROOT/'protocols/lean_channel_attention_optimization_pilot_v1.json'):
            raise ValueError('Optimizer gate source or protocol changed')
        for arm,record in r['arms'].items():
            path=folder/f'{arm}_fixture.pt'
            if digest(path)!=record['fixture_sha256']:raise ValueError('Optimizer fixture changed')
            payload=torch.load(path,map_location='cpu',weights_only=True)
            if payload.get('engineering_only') is not True:raise ValueError('Synthetic fixture is not marked')
            try:PromptContrastChannelAttention(checkpoint=path,expected_backbone_sha256=payload['backbone_sha256'],expected_feature_pipeline_sha256=payload['feature_pipeline_sha256'])
            except ValueError as e:
                if 'Engineering fixture cannot' not in str(e):raise
            else:raise ValueError('Synthetic fixture accepted as trained')
            model=build_trainable_gate(20261114);model.gate.load_state_dict(payload['gate_state_dict'],strict=True)
            if parameter_hash(model)!=record['final_parameter_sha256'] or record['steps']!=1:raise ValueError('Optimizer fixture weights changed')
        results.append(r)
    if results[0]!=results[1]:raise ValueError('Optimizer gates differ')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'first_sha256':digest(a.first/'verification.json'),'second_sha256':digest(a.second/'verification.json'),
      'adapter_sha256':digest(ROOT/'modeling/lean_channel_attention.py'),'training_sha256':digest(ROOT/'modeling/lean_channel_training.py'),
      'loss_sha256':digest(ROOT/'modeling/lean_channel_training_loss.py'),'optimization_protocol_sha256':digest(ROOT/'protocols/lean_channel_attention_optimization_pilot_v1.json'),
      'verifier_sha256':digest(Path(__file__)),'scope':'Synthetic-target engineering only; fixtures are not valid inference checkpoints.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
