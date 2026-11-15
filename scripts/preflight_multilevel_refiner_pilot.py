#!/usr/bin/env python3
"""Verify saved, aligned features and labels for a small refiner feasibility pilot."""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for k in ('captures','preflight','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();read=lambda x:json.loads(x.read_text());pre=read(a.preflight/'manifest.json');done=read(a.preflight/'completion.json')
if not done['passed'] or done['manifest_sha256']!=digest(a.preflight/'manifest.json'):raise ValueError('Split preflight changed')
if pre['groups']['train']!=['seq2','seq3','seq4'] or pre['groups']['validation']!=['seq9','seq10'] or set(pre['groups']['protected_priority'])!={'seq8','seq11','seq20','seq23'}:raise ValueError('Unexpected split')
videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};samples=[];pipelines=set()
for split in ('train','validation'):
 for seq in pre['groups'][split]:
  folder=a.captures/('gate_attempt01' if seq=='seq3' else f'{seq}_capture_attempt01');u=read(folder/'utility_capture_receipt.json');f=read(folder/'channel_feature_receipt.json');meta=pre['sequences'][seq];source=Path(meta['source']);rows=read(source/seq/'frames.json')
  if not u['passed'] or not f['passed'] or u['sequence']!=seq or f['sequence']!=seq or u['feature_receipt_sha256']!=digest(folder/'channel_feature_receipt.json') or f['checkpoint_sha256']!=digest(a.checkpoint) or f['preflight_sha256']!=digest(a.preflight/'manifest.json') or digest(source/seq/'frames.json')!=meta['frames_metadata_sha256'] or u['targets']!=meta['targets'] or f['targets']!=meta['targets']:raise ValueError('Capture or input provenance differs')
  pipelines.add(f['feature_pipeline_sha256']);byframe={z['frame']:z for z in f['features']};shots={z['frame']:z['sha256'] for z in u['snapshots']}
  for i in u['targets']:
   feature=folder/'features'/f'{i:06d}.pt';snapshot=folder/'snapshots'/f'{i:06d}.pt';frame=videos[seq][i]
   if digest(feature)!=byframe[i]['file_sha256'] or digest(snapshot)!=shots[i] or digest(frame.image_path)!=rows[i]['image_sha256'] or digest(frame.mask_path)!=rows[i]['label_sha256']:raise ValueError('Captured feature, read or input changed')
   x=torch.load(feature,map_location='cpu',weights_only=True);y=torch.load(snapshot,map_location='cpu',weights_only=True)
   if x['frame']!=i or y['frame']!=i or tuple(x['feature'].shape)!=(32,256,256) or tuple(y['decoder_inputs']['backbone_features'].shape)!=(1,256,64,64) or tuple(y['decoder_outputs'][3].shape)!=(1,1,256,256):raise ValueError('Feature alignment or geometry differs')
   samples.append({'split':split,'sequence':seq,'frame':i,'image':str(frame.image_path),'image_sha256':rows[i]['image_sha256'],'label':str(frame.mask_path),'label_sha256':rows[i]['label_sha256'],'feature':str(feature),'feature_sha256':digest(feature),'snapshot':str(snapshot),'snapshot_sha256':digest(snapshot)})
if len(pipelines)!=1 or len(samples)!=25:raise ValueError('Feature pipeline or sample coverage differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
result={'passed':True,'samples':samples,'checkpoint_sha256':digest(a.checkpoint),'feature_pipeline_sha256':pipelines.pop(),'split_sha256':digest(a.preflight/'manifest.json'),'source_sha256':digest(Path(__file__)),'scope':'25 previously inspected fixed reads, sequence-separated only; patient independence unknown. No priority sequence in training or validation.'};(out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'manifest_sha256':digest(out/'manifest.json')},indent=2)+'\n');print(json.dumps({'passed':True,'samples':len(samples)}))
