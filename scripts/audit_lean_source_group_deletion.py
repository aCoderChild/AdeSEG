#!/usr/bin/env python3
"""Score previously frozen, independently repeated group deletions."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for name in ('capture','replay','verification','deletion-verification','merged','output'):p.add_argument('--'+name,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());r=read(a.replay/'completion.json');v=read(a.verification/'verification.json');cap=read(a.capture/'source_capture_receipt.json')
if not r['passed'] or not v['passed'] or not cap['passed'] or v['first_sha256']!=digest(a.replay/'completion.json') or v['capture_sha256']!=digest(a.capture/'source_capture_receipt.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_source_group_deletion.py') or r['protocol_sha256']!=digest(ROOT/'protocols/lean_source_group_deletion_v1.json') or r['sequence']!=cap.get('sequence',r['sequence']):raise ValueError('Frozen replay or verification invalid')
gate=read(a.deletion_verification/'verification.json')
if not gate['passed'] or gate['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_source_group_deletion.py'):raise ValueError('Deletion gate invalid')
if r['sequence'] != 'seq3':
 candidate=read(a.replay/'candidate_receipt.json')
 if not candidate['passed'] or candidate['completion_sha256']!=digest(a.replay/'completion.json') or candidate['gate_verification_sha256']!=digest(a.deletion_verification/'verification.json') or candidate['runner_sha256']!=digest(ROOT/'scripts/replay_lean_source_group_candidate.py'):
  raise ValueError('Candidate gate receipt invalid')
seq=r['sequence'];videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};frames=videos[seq];source=Path(read(a.merged/'locations.json')[seq]);done=read(source/'completion.json');record=read(source/'records.json')[0]
if digest(source/'records.json')!=done['records_sha256'] or digest(source/seq/'frames.json')!=record['frames_sha256']:raise ValueError('Source metadata changed')
rows=read(source/seq/'frames.json');results=[]
for e in r['events']:
 if 'groups' not in e:continue
 i=e['frame'];f=frames[i]
 if digest(f.mask_path)!=rows[i]['label_sha256']:raise ValueError('Label changed')
 truth=load_label_mask(f.mask_path,[1])==1;present=bool(truth.any());scores={}
 for name,sha in e['masks'].items():
  path=a.replay/f'{i:06d}_{name}.png'
  if digest(path)!=sha:raise ValueError('Mask changed')
  pred=np.asarray(Image.open(path))>0
  if pred.shape!=truth.shape:raise ValueError('Mask geometry differs')
  den=int(pred.sum()+truth.sum());scores[name]={'dice':2*int((pred&truth).sum())/den if den else 1.,'false_alarm':bool(pred.any()) if not present else None,'predicted_pixels':int(pred.sum())}
 results.append({'frame':i,'present':present,'label_sha256':rows[i]['label_sha256'],'scores':scores})
if len(results)!=v['eligible_reads']:raise ValueError('Eligible frame count differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'sequence':seq,'frames':results,'excluded_reads':v['excluded_reads'],'replay_sha256':digest(a.replay/'completion.json'),'verification_sha256':digest(a.verification/'verification.json'),'protocol_sha256':r['protocol_sha256'],'runner_sha256':digest(Path(__file__)),'scope':'Local deletion effects conditional on full read; not full-video efficacy.'}
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json')},indent=2)+'\n');print(json.dumps({'passed':True,'sequence':seq,'reads':len(results)}))
