#!/usr/bin/env python3
"""Train predeclared author-CBAM gate arms on training-sequence data only."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_attention import author_sha256
from modeling.lean_channel_training import build_trainable_gate,train_gate
from scripts.memory_token_io import digest,validate_output

PROTOCOL=ROOT/'protocols/lean_channel_attention_optimization_pilot_v1.json'
def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for k in ('dataset','dataset-audit','optimizer-audit','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();dataset=read(a.dataset/'dataset.json');done=read(a.dataset/'completion.json');training=read(a.dataset/'train_manifest.json');proto=read(PROTOCOL);audit=read(a.dataset_audit/'verification.json');opt_audit=read(a.optimizer_audit/'verification.json')
    if not done['passed'] or done['dataset_sha256']!=digest(a.dataset/'dataset.json') or dataset['train_manifest_sha256']!=digest(a.dataset/'train_manifest.json') or dataset['preparation_source_sha256']!=digest(ROOT/'scripts/prepare_lean_channel_training_dataset.py') or dataset['optimization_protocol_sha256']!=digest(PROTOCOL):
        raise ValueError('Dataset or optimization protocol changed')
    if not audit['passed'] or audit['verifier_sha256']!=digest(ROOT/'scripts/audit_lean_channel_training_dataset.py') or audit['dataset_sha256']!=digest(a.dataset/'dataset.json') or audit['train_manifest_sha256']!=digest(a.dataset/'train_manifest.json') or audit['optimization_protocol_sha256']!=digest(PROTOCOL):
        raise ValueError('Independent dataset audit changed')
    if not opt_audit['passed'] or opt_audit['verifier_sha256']!=digest(ROOT/'scripts/audit_lean_channel_optimizer_gate.py') or opt_audit['adapter_sha256']!=digest(ROOT/'modeling/lean_channel_attention.py') or opt_audit['training_sha256']!=digest(ROOT/'modeling/lean_channel_training.py') or opt_audit['loss_sha256']!=digest(ROOT/'modeling/lean_channel_training_loss.py') or opt_audit['optimization_protocol_sha256']!=digest(PROTOCOL):
        raise ValueError('Independent optimizer gate changed')
    if digest(a.checkpoint)!=dataset['backbone_sha256'] or training['feature_pipeline_sha256']!=dataset['feature_pipeline_sha256']:
        raise ValueError('Feature backbone or pipeline changed')
    expected=['seq2','seq3','seq4'];groups=dataset['groups']
    if training['sequences']!=expected or groups['train']!=expected or groups['validation']!=['seq9','seq10'] or set(groups['protected_priority'])!={'seq8','seq11','seq20','seq23'}:
        raise ValueError('Unexpected sequence groups')
    if training['samples']!=[x for x in dataset['samples'] if x['group']=='train'] or len(training['samples'])!=15:
        raise ValueError('Training samples differ')
    cfg=proto['training']
    if cfg['seed']!=20261114 or cfg['device']!='cpu' or cfg['dtype']!='float32' or cfg['epochs']!=80 or cfg['optimizer']!='Adam' or cfg['learning_rate']!=0.0003 or cfg['weight_decay']!=0.0001:
        raise ValueError('Optimization protocol differs')
    samples=[];prototypes={}
    for item in sorted(training['samples'],key=lambda x:(x['sequence'],x['frame'])):
        if item['group']!='train' or item['sequence'] not in expected:raise ValueError('Training leakage')
        feature=Path(item['feature_path']);target=Path(item['target_path'])
        if digest(feature)!=item['feature_file_sha256'] or digest(target)!=item['target_sha256']:raise ValueError('Training tensor changed')
        sample=torch.load(feature,map_location='cpu',weights_only=True);labels=torch.load(target,map_location='cpu',weights_only=True)
        if sample['frame']!=item['frame'] or sample['feature'].shape!=(32,256,256) or labels.shape!=(4096,) or not torch.isfinite(labels).all() or (labels<0).any() or (labels>1).any():raise ValueError('Invalid training sample')
        pair=(sample['foreground'],sample['background']);previous=prototypes.setdefault(item['sequence'],pair)
        if not torch.equal(previous[0],pair[0]) or not torch.equal(previous[1],pair[1]):raise ValueError('Prompt prototypes changed within sequence')
        samples.append({'sequence':item['sequence'],'frame':item['frame'],'feature':sample['feature'],'foreground':pair[0],'background':pair[1],'target':labels})
    if set(prototypes)!=set(expected):raise ValueError('Missing train sequence')
    foreign={'seq2':prototypes['seq3'],'seq3':prototypes['seq4'],'seq4':prototypes['seq2']}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);results={};initial_hash=None
    for arm in ('prompt_conditioned','descriptor_independent','mismatched_prompt'):
        model=build_trainable_gate(cfg['seed']);run=train_gate(model,samples,arm,foreign,cfg['epochs'],cfg['learning_rate'],cfg['weight_decay'])
        if initial_hash is None:initial_hash=run['initial_parameter_sha256']
        if run['initial_parameter_sha256']!=initial_hash:raise ValueError('Training arms have different initial weights')
        payload={'format':'sam2_early_prompt_cbam_v1','channels':32,'author_sha256':author_sha256(),'training_steps':run['steps'],
          'feature_interface':'sam2.1_hiera_base_plus.conv_s0.fpn0','backbone_sha256':dataset['backbone_sha256'],
          'feature_pipeline_sha256':dataset['feature_pipeline_sha256'],'train_manifest_sha256':dataset['train_manifest_sha256'],
          'optimization_protocol_sha256':digest(PROTOCOL),'dataset_sha256':digest(a.dataset/'dataset.json'),'arm':arm,'engineering_only':False,
          'gate_state_dict':{k:v.detach().cpu().clone() for k,v in model.gate.state_dict().items()}}
        checkpoint=out/f'{arm}.pt';torch.save(payload,checkpoint);run['checkpoint_sha256']=digest(checkpoint);results[arm]=run
        (out/'progress.json').write_text(json.dumps({'completed_arms':list(results),'remaining_arms':[x for x in ('prompt_conditioned','descriptor_independent','mismatched_prompt') if x not in results]},indent=2)+'\n')
        print('completed',arm,flush=True)
    result={'passed':True,'scope':'small sequence-separated exploratory supervised feasibility pilot','arms':results,
      'train_manifest_sha256':dataset['train_manifest_sha256'],'dataset_sha256':digest(a.dataset/'dataset.json'),'optimization_protocol_sha256':digest(PROTOCOL),
      'dataset_audit_sha256':digest(a.dataset_audit/'verification.json'),'optimizer_audit_sha256':digest(a.optimizer_audit/'verification.json'),'author_sha256':author_sha256(),'adapter_sha256':digest(ROOT/'modeling/lean_channel_attention.py'),'training_sha256':digest(ROOT/'modeling/lean_channel_training.py'),
      'loss_sha256':digest(ROOT/'modeling/lean_channel_training_loss.py'),'runner_sha256':digest(Path(__file__))}
    (out/'training_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'arms':list(results)}))
if __name__=='__main__':main()
