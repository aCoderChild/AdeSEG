#!/usr/bin/env python3
"""Independently audit two fixed-protocol training runs before validation."""
import argparse,json,math,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_attention import PromptContrastChannelAttention,author_sha256
from modeling.lean_channel_training import parameter_hash
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for k in ('first','second','dataset','dataset-audit','optimizer-audit','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();dataset=read(a.dataset/'dataset.json');da=read(a.dataset_audit/'verification.json');oa=read(a.optimizer_audit/'verification.json')
    if not da['passed'] or da['verifier_sha256']!=digest(ROOT/'scripts/audit_lean_channel_training_dataset.py') or da['dataset_sha256']!=digest(a.dataset/'dataset.json') or not oa['passed'] or oa['verifier_sha256']!=digest(ROOT/'scripts/audit_lean_channel_optimizer_gate.py'):
        raise ValueError('Prerequisite audit changed')
    results=[];states=[]
    for folder in (a.first,a.second):
        r=read(folder/'training_receipt.json')
        if not r['passed'] or r['runner_sha256']!=digest(ROOT/'scripts/train_lean_channel_attention_pilot.py') or r['adapter_sha256']!=digest(ROOT/'modeling/lean_channel_attention.py') or r['training_sha256']!=digest(ROOT/'modeling/lean_channel_training.py') or r['loss_sha256']!=digest(ROOT/'modeling/lean_channel_training_loss.py') or r['optimization_protocol_sha256']!=digest(ROOT/'protocols/lean_channel_attention_optimization_pilot_v1.json'):
            raise ValueError('Training source changed')
        if r['dataset_sha256']!=digest(a.dataset/'dataset.json') or r['train_manifest_sha256']!=digest(a.dataset/'train_manifest.json') or r['dataset_audit_sha256']!=digest(a.dataset_audit/'verification.json') or r['optimizer_audit_sha256']!=digest(a.optimizer_audit/'verification.json'):
            raise ValueError('Training data identity differs')
        if set(r['arms'])!={'prompt_conditioned','descriptor_independent','mismatched_prompt'}:raise ValueError('Missing training arms')
        this={};initial=set()
        for arm,record in r['arms'].items():
            path=folder/f'{arm}.pt';payload=torch.load(path,map_location='cpu',weights_only=True)
            if digest(path)!=record['checkpoint_sha256'] or payload.get('engineering_only') is not False or payload['arm']!=arm or payload['training_steps']!=1200 or payload['train_manifest_sha256']!=digest(a.dataset/'train_manifest.json') or payload['optimization_protocol_sha256']!=digest(ROOT/'protocols/lean_channel_attention_optimization_pilot_v1.json'):
                raise ValueError('Checkpoint provenance differs')
            if len(record['epoch_losses'])!=80 or any(not math.isfinite(x) for x in record['epoch_losses']) or record['steps']!=1200 or record['initial_parameter_sha256']==record['final_parameter_sha256']:
                raise ValueError('Training did not execute the fixed schedule')
            model=PromptContrastChannelAttention(checkpoint=path,expected_backbone_sha256=dataset['backbone_sha256'],expected_feature_pipeline_sha256=dataset['feature_pipeline_sha256']).eval()
            if parameter_hash(model)!=record['final_parameter_sha256'] or payload['author_sha256']!=author_sha256():raise ValueError('Checkpoint parameters differ')
            initial.add(record['initial_parameter_sha256']);this[arm]=payload['gate_state_dict']
        if len(initial)!=1:raise ValueError('Arms did not use identical initialization')
        states.append(this);results.append(r)
    for arm in results[0]['arms']:
        a0=results[0]['arms'][arm];a1=results[1]['arms'][arm]
        for k in ('initial_parameter_sha256','final_parameter_sha256','steps','epoch_losses'):
            if a0[k]!=a1[k]:raise ValueError('Training repeats differ: '+arm+' '+k)
        if any(not torch.equal(states[0][arm][k],states[1][arm][k]) for k in states[0][arm]):raise ValueError('Checkpoint tensors differ')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'first_receipt_sha256':digest(a.first/'training_receipt.json'),'second_receipt_sha256':digest(a.second/'training_receipt.json'),
      'dataset_sha256':digest(a.dataset/'dataset.json'),'train_manifest_sha256':digest(a.dataset/'train_manifest.json'),'optimizer_protocol_sha256':digest(ROOT/'protocols/lean_channel_attention_optimization_pilot_v1.json'),
      'verifier_sha256':digest(Path(__file__)),'scope':'Matched deterministic training provenance only; validation not yet evaluated.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
