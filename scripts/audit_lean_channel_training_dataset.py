#!/usr/bin/env python3
"""Independently reconstruct masks and verify frozen training data identities."""
import argparse,json,sys
from pathlib import Path
import numpy as np
import torch
import torch.nn.functional as F
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from modeling.lean_channel_feature_capture import sha
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--dataset',type=Path,required=True);p.add_argument('--preflight',type=Path,required=True);p.add_argument('--pipeline-gate',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    d=read(a.dataset/'dataset.json');done=read(a.dataset/'completion.json');m=read(a.preflight/'manifest.json');gate=read(a.pipeline_gate/'verification.json')
    if not done['passed'] or done['dataset_sha256']!=digest(a.dataset/'dataset.json') or d['preparation_source_sha256']!=digest(ROOT/'scripts/prepare_lean_channel_training_dataset.py') or d['train_manifest_sha256']!=digest(a.dataset/'train_manifest.json') or d['preflight_sha256']!=digest(a.preflight/'manifest.json') or d['pipeline_verification_sha256']!=digest(a.pipeline_gate/'verification.json') or not gate['passed']:
        raise ValueError('Dataset provenance changed')
    if d['optimization_protocol_sha256']!=digest(ROOT/'protocols/lean_channel_attention_optimization_pilot_v1.json') or d['feature_pipeline_sha256']!=gate['feature_pipeline_sha256'] or d['groups']!={'train':['seq2','seq3','seq4'],'validation':['seq9','seq10'],'protected_priority':['seq11','seq20','seq23','seq8']}:
        raise ValueError('Training group or fixed protocol changed')
    tm=read(a.dataset/'train_manifest.json');train=[x for x in d['samples'] if x['group']=='train'];val=[x for x in d['samples'] if x['group']=='validation']
    if tm['samples']!=train or len(train)!=15 or len(val)!=10:raise ValueError('Training or validation coverage differs')
    videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};counts={};prototype={}
    for x in d['samples']:
        seq=x['sequence'];i=x['frame'];expected_group='train' if seq in d['groups']['train'] else 'validation'
        if x['group']!=expected_group or i not in m['sequences'][seq]['targets'] or Path(x['image_path'])!=videos[seq][i].image_path or Path(x['mask_path'])!=videos[seq][i].mask_path:
            raise ValueError('Sample assignment changed')
        if digest(Path(x['image_path']))!=x['image_sha256'] or digest(Path(x['mask_path']))!=x['label_sha256'] or digest(Path(x['feature_path']))!=x['feature_file_sha256'] or digest(Path(x['target_path']))!=x['target_sha256']:
            raise ValueError('Sample files changed')
        f=torch.load(x['feature_path'],map_location='cpu',weights_only=True);target=torch.load(x['target_path'],map_location='cpu',weights_only=True)
        if f['frame']!=i or f['feature'].shape!=(32,256,256) or target.shape!=(4096,) or not torch.isfinite(f['feature']).all():raise ValueError('Invalid feature geometry')
        mask=np.asarray(load_label_mask(Path(x['mask_path']),[1])==1,dtype=np.float32)
        expected=F.interpolate(torch.from_numpy(mask)[None,None],size=(64,64),mode='area')[0,0].flatten()
        if not torch.equal(target,expected):raise ValueError('Training target does not reconstruct from label')
        pair=(sha(f['foreground']),sha(f['background']));old=prototype.setdefault(seq,pair)
        if old!=pair:raise ValueError('Prompt prototypes changed within video')
        counts[seq]=counts.get(seq,0)+1
    if counts!={s:5 for s in d['groups']['train']+d['groups']['validation']}:raise ValueError('Sequence sample count differs')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'passed':True,'dataset_sha256':digest(a.dataset/'dataset.json'),'train_manifest_sha256':digest(a.dataset/'train_manifest.json'),
      'optimization_protocol_sha256':d['optimization_protocol_sha256'],'counts':counts,'verifier_sha256':digest(Path(__file__)),'limitation':'Patient identity is not verified; this is exploratory sequence-separated data.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
