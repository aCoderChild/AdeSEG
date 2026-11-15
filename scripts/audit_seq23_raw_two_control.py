#!/usr/bin/env python3
"""Audit the seq23 raw-two control against three saved automatic arms."""
import argparse,collections,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output
from scripts.audit_lean_token_window import verify_events

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('baseline','raw-window','fused-window','raw-two','fused-audit','historical','verification','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();audit_done=read(a.fused_audit/'completion.json');aud=read(a.fused_audit/'results.json');v=read(a.verification/'verification.json');vr=read(a.verification/'raw_two_control_receipt.json')
 if not audit_done['passed'] or digest(a.fused_audit/'results.json')!=audit_done['results_sha256'] or audit_done['source_sha256']!=digest(ROOT/'scripts/audit_lean_fused_window_control.py') or 'seq23' not in aud['sequences']:
  raise ValueError('Fused audit invalid')
 if not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or not vr['passed'] or vr['delegate_receipt_sha256']!=digest(a.verification/'verification.json') or vr['dispatcher_sha256']!=digest(ROOT/'scripts/run_lean_raw_two_control.py') or vr['protocol_sha256']!=digest(ROOT/'protocols/lean_raw_two_window_control_v1.json'):
  raise ValueError('Raw-two gate invalid')
 ident=v['identity'];fork=ROOT/'external/Lean-SAM2-raw-two-control'
 if ident['runner_sha256']!=digest(ROOT/'scripts/run_lean_token_window.py') or ident['protocol_sha256']!=digest(ROOT/'protocols/lean_raw_two_window_control_v1.json') or ident['runtime_sha256']!=digest(ROOT/'modeling/lean_token_window_storage.py') or ident['policy_sha256']!=digest(ROOT/'modeling/token_memory_window.py') or ident['observer_sha256']!=digest(ROOT/'modeling/lean_token_storage_observer.py') or any(ident['patch_sha256'][k]!=digest(fork/k) for k in ('sam2/modeling/sam2_base.py','sam2/sam2_video_predictor_topk_insurance.py')):raise ValueError('Raw-two source changed')
 runs={'baseline':a.baseline,'raw_window':a.raw_window,'fused_window':a.fused_window,'raw_two':a.raw_two};records={};meta={}
 for name,path in runs.items():
  d=read(path/'completion.json');r=read(path/'frames.json')
  if not d['passed'] or digest(path/'frames.json')!=d['frames_sha256'] or digest(path/'positions.json')!=d['positions_sha256'] or not read(path/'parameter_integrity_after.json')['unchanged']:raise ValueError('Run records or model invalid')
  records[name]=r;meta[name]=d
 ob=read(a.raw_two/'token_window_observation.json');outer=read(a.raw_two/'raw_two_control_receipt.json')
 if not ob['passed'] or ob['mode']!='candidate' or ob['verification_sha256']!=digest(a.verification/'verification.json') or ob['baseline_completion_sha256']!=digest(a.baseline/'completion.json') or ob['source_sha256']!=digest(ROOT/'scripts/run_seq23_token_window.py') or ob['records_sha256']!=meta['raw_two']['frames_sha256'] or digest(a.raw_two/'token_window_events.json')!=ob['events_sha256'] or not outer['passed'] or outer['delegate_receipt_sha256']!=digest(a.raw_two/'token_window_observation.json') or outer['dispatcher_sha256']!=digest(ROOT/'scripts/run_lean_raw_two_control.py') or outer['protocol_sha256']!=digest(ROOT/'protocols/lean_raw_two_window_control_v1.json'):
  raise ValueError('Raw-two provenance invalid')
 if any(digest(path/'prompt.json')!=meta['baseline']['prompt_sha256'] for path in runs.values()):raise ValueError('Prompt differs')
 frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq23'];first=read(a.baseline/'prompt.json')['first_detection_index'];positions={k:read(p/'positions.json') for k,p in runs.items()}
 telem=verify_events(a.raw_two,first,len(frames),positions['baseline']);events=read(a.raw_two/'token_window_events.json');reads=[x for x in events if x['stage']=='after_read']
 if any(x.get('history_read_mode')!='raw_two_latest' for x in reads):raise ValueError('Raw-two path not active')
 if any(len(rows)!=len(frames) for rows in records.values()):raise ValueError('Frame count differs')
 for name,rows in records.items():
  if [x['mask_sha256'] for x in rows[:first+3]]!=[x['mask_sha256'] for x in records['baseline'][:first+3]]:raise ValueError('Causal prompt prefix differs')
 target=[json.loads(line) for line in (a.historical/'full/seq23/frames.jsonl').read_text().splitlines()];target_present={x['frame'] for x in target if x['present']};values=[]
 for i,f in enumerate(frames):
  br=records['baseline'][i]
  if digest(f.image_path)!=br['image_sha256'] or digest(f.mask_path)!=br['label_sha256']:raise ValueError('Input changed')
  gt=load_label_mask(f.mask_path,[1])==1;present=bool(gt.any());scores={}
  for name,path in runs.items():
   r=records[name][i];file=path/f'{i:06d}.png'
   if digest(file)!=r['mask_sha256'] or r['image_sha256']!=br['image_sha256'] or r['label_sha256']!=br['label_sha256']:raise ValueError('Mask or inputs changed')
   mask=np.asarray(Image.open(file))>0
   if mask.shape!=gt.shape:raise ValueError('Mask shape differs')
   den=int(mask.sum()+gt.sum());score=2*int((mask&gt).sum())/den if den else 1.;alarm=bool(mask.any()) if not present else None
   if r['present']!=present or abs(r['dice']-score)>1e-12 or r['false_alarm']!=alarm:raise ValueError('Stored score differs')
   scores[name]={'dice':score,'false_alarm':alarm,'mask_sha256':r['mask_sha256']}
  prior=aud['sequences']['seq23']['per_frame'][i]
  if prior['present']!=present or any(abs(scores[k]['dice']-prior['scores'][k]['dice'])>1e-12 or scores[k]['false_alarm']!=prior['scores'][k]['false_alarm'] for k in ('baseline','raw_window','fused_window')):raise ValueError('Prior audited score differs')
  values.append({'frame':i,'present':present,'matched':i in target_present,'scores':scores})
 if any(values[t['frame']]['present']!=t['present'] for t in target):raise ValueError('Historical target mismatch')
 result={}
 for name in runs:
  vals=[r['scores'][name]['dice'] for r in values if r['matched']];empty=[r['scores'][name]['false_alarm'] for r in values if not r['present']]
  keys=positions[name];result[name]={'matched_present_dice':float(np.mean(vals)),'matched_present_count':len(vals),'false_alarms':sum(empty),'empty_count':len(empty),
    'mean_spatial_keys':float(np.mean([z['spatial_keys'] for z in keys])),'max_spatial_keys':max(z['spatial_keys'] for z in keys)}
 fusedpos=positions['fused_window'];twopos=positions['raw_two']
 if len(fusedpos)!=len(twopos):raise ValueError('Read count differs')
 key_matches=sum(x['spatial_keys']==y['spatial_keys'] for x,y in zip(fusedpos,twopos));pointer_matches=sum(x['pointer_keys']==y['pointer_keys'] for x,y in zip(fusedpos,twopos))
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 payload={'scope':'Retrospective selected seq23; all arms use the same no-inset input and automatic prompt.','conditions':result,'frames':values,'raw_two_telemetry':telem,
  'matched_spatial_key_counts':key_matches,'matched_pointer_key_counts':pointer_matches,'read_count':len(fusedpos),'fused_audit_sha256':audit_done['results_sha256'],'source_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
 lines=['# Sequence 23: raw-two matched-read control','',payload['scope'],'','| Arm | Matched present Dice | Empty false alarms | Mean spatial keys/read |', '| --- | ---: | ---: | ---: |']
 for name,x in result.items():lines.append(f"| {name} | {x['matched_present_dice']:.6f} | {x['false_alarms']} / {x['empty_count']} | {x['mean_spatial_keys']:.1f} |")
 lines+=['',f"Raw-two and fused spatial-key counts match on {key_matches}/{len(fusedpos)} reads; pointer-key counts match on {pointer_matches}/{len(fusedpos)}. Equal counts do not establish identical tokens or histories. The raw-two control uses only the two newest raw nonconditioning entries, whereas fused uses an author-fused older entry and newest raw entry. Both retain the same spatial storage budget, but predictions can alter subsequent token selection. This is a selected, previously inspected video, not a same-state causal ablation or evidence of generalization."]
 (out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'source_sha256':digest(Path(__file__))},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
