#!/usr/bin/env python3
"""Freeze audited train/validation features and labels after capture completes."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output
from modeling.lean_channel_feature_capture import sha


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for k in ('root','preflight','pipeline-gate','interface-audit','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();pre=read(a.preflight/'completion.json');manifest=read(a.preflight/'manifest.json');gate=read(a.pipeline_gate/'verification.json');interface=read(a.interface_audit/'verification.json')
    if not pre['passed'] or pre['manifest_sha256']!=digest(a.preflight/'manifest.json') or manifest['protocol_sha256']!=digest(ROOT/'protocols/lean_channel_attention_training_pilot_v1.json') or manifest['source_sha256']!=digest(ROOT/'scripts/preflight_lean_channel_training.py'):
        raise ValueError('Split preflight changed')
    if not gate['passed'] or gate['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_channel_pipeline_gates.py') or gate['runner_sha256']!=digest(ROOT/'scripts/capture_lean_channel_features.py') or gate['preflight_sha256']!=pre['manifest_sha256']:
        raise ValueError('Feature pipeline gate changed')
    if not interface['passed'] or interface['verifier_sha256']!=digest(ROOT/'scripts/audit_lean_channel_attention_interface.py') or interface.get('adapter_sha256')!=digest(ROOT/'modeling/lean_channel_attention.py'):
        raise ValueError('Channel interface audit changed')
    groups=manifest['groups'];train=groups['train'];val=groups['validation'];protected=set(groups['protected_priority'])
    if set(train)&set(val) or (set(train)|set(val))&protected:raise ValueError('Group leakage')
    videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};prepared=[];backbone_sha=None
    # Validate all feature captures first; no ground-truth masks are opened here.
    for seq in train+val:
        folder=a.root/('pipeline_gate_attempt03' if seq=='seq3' else f'{seq}_capture_attempt01')
        receipt=read(folder/'channel_feature_receipt.json');done=read(folder/'completion.json');meta=manifest['sequences'][seq]
        if not receipt['passed'] or receipt['sequence']!=seq or receipt['mode']!=('gate' if seq=='seq3' else 'candidate') or receipt['runner_sha256']!=gate['runner_sha256'] or receipt['feature_pipeline_sha256']!=gate['feature_pipeline_sha256'] or receipt['preflight_sha256']!=pre['manifest_sha256'] or receipt['targets']!=meta['targets']:
            raise ValueError('Feature capture identity differs: '+seq)
        if backbone_sha is None:backbone_sha=receipt['checkpoint_sha256']
        if receipt['checkpoint_sha256']!=backbone_sha:raise ValueError('Backbone checkpoint differs')
        if not done['passed'] or done['records_sha256']!=digest(folder/'records.json') or receipt['records_sha256']!=done['records_sha256'] or receipt['events_sha256']!=digest(folder/'token_window_events.json'):
            raise ValueError('Feature capture traces changed: '+seq)
        integrity=read(folder/'parameter_integrity_after.json')
        if not integrity['unchanged'] or integrity['changed_names'] or integrity['before']!=integrity['after']:raise ValueError('Model parameters changed: '+seq)
        events=receipt['features']
        if {e['frame'] for e in events}!=set(meta['targets']):raise ValueError('Feature coverage differs')
        source=Path(meta['source']);source_done=read(source/'completion.json');record=read(source/'records.json')[0];rows=read(source/seq/'frames.json')
        if digest(source/'records.json')!=source_done['records_sha256'] or digest(source/seq/'frames.json')!=record['frames_sha256'] or record['frames_sha256']!=meta['frames_metadata_sha256']:
            raise ValueError('Archived image metadata changed')
        for e in events:
            idx=e['frame'];path=folder/'features'/f'{idx:06d}.pt'
            if digest(path)!=e['file_sha256'] or digest(videos[seq][idx].image_path)!=rows[idx]['image_sha256']:raise ValueError('Feature or input changed')
            d=torch.load(path,map_location='cpu',weights_only=True)
            if d['frame']!=idx or d['feature'].shape!=(32,256,256) or any(sha(d[k])!=e[k+'_sha256'] for k in ('feature','foreground','background')):
                raise ValueError('Feature content differs')
            prepared.append({'sequence':seq,'frame':idx,'group':'train' if seq in train else 'validation','feature_path':str(path),'feature_file_sha256':e['file_sha256'],
              'image_path':str(videos[seq][idx].image_path),'image_sha256':rows[idx]['image_sha256'],'mask_path':str(videos[seq][idx].mask_path),'label_sha256':rows[idx]['label_sha256']})
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'targets').mkdir()
    for item in prepared:
        path=Path(item['mask_path'])
        if digest(path)!=item['label_sha256']:raise ValueError('Label content differs')
        mask=(load_label_mask(path,[1])==1).astype(np.float32)
        target=F.interpolate(torch.from_numpy(mask)[None,None],size=(64,64),mode='area')[0,0].reshape(-1)
        target_path=out/'targets'/f"{item['sequence']}_{item['frame']:06d}.pt";torch.save(target,target_path)
        item['target_path']=str(target_path);item['target_sha256']=digest(target_path)
    train_manifest={'sequences':train,'samples':[x for x in prepared if x['group']=='train'],'feature_pipeline_sha256':gate['feature_pipeline_sha256']}
    (out/'train_manifest.json').write_text(json.dumps(train_manifest,indent=2)+'\n')
    dataset={'groups':{'train':train,'validation':val,'protected_priority':sorted(protected)},'samples':prepared,
      'feature_pipeline_sha256':gate['feature_pipeline_sha256'],'backbone_sha256':backbone_sha,'preflight_sha256':pre['manifest_sha256'],'pipeline_verification_sha256':digest(a.pipeline_gate/'verification.json'),
      'interface_verification_sha256':digest(a.interface_audit/'verification.json'),'optimization_protocol_sha256':digest(ROOT/'protocols/lean_channel_attention_optimization_pilot_v1.json'),
      'train_manifest_sha256':digest(out/'train_manifest.json'),'preparation_source_sha256':digest(Path(__file__)),'warning':'Sequence-separated exploratory pilot; patient identity is unverified.'}
    (out/'dataset.json').write_text(json.dumps(dataset,indent=2)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'dataset_sha256':digest(out/'dataset.json')},indent=2)+'\n')
    print(json.dumps({'passed':True,'train_samples':sum(x['group']=='train' for x in prepared),'validation_samples':sum(x['group']=='validation' for x in prepared)}))
if __name__=='__main__':main()
