#!/usr/bin/env python3
"""Audit the priority raw-two control against fixed baseline and window arms."""
import argparse,json,sys
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
 for k in ('baseline','raw-window','fused-window','raw-two','raw-audit','merged','historical','verification','fused-verification','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--sequence',choices=('seq8','seq20'),required=True)
 a=p.parse_args();audit_done=read(a.raw_audit/'completion.json');aud=read(a.raw_audit/'results.json');v=read(a.verification/'verification.json');vr=read(a.verification/'raw_two_control_receipt.json')
 if not audit_done['passed'] or digest(a.raw_audit/'results.json')!=audit_done['results_sha256'] or audit_done['source_sha256']!=digest(ROOT/'scripts/audit_lean_token_window.py') or a.sequence not in aud['sequences']:
  raise ValueError('Prior raw-window audit invalid')
 if not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or not vr['passed'] or vr['delegate_receipt_sha256']!=digest(a.verification/'verification.json') or vr['dispatcher_sha256']!=digest(ROOT/'scripts/run_lean_raw_two_control.py') or vr['protocol_sha256']!=digest(ROOT/'protocols/lean_raw_two_window_control_v1.json'):raise ValueError('Raw-two gate invalid')
 ident=v['identity'];fork=ROOT/'external/Lean-SAM2-raw-two-control'
 if ident['runner_sha256']!=digest(ROOT/'scripts/run_lean_token_window.py') or ident['protocol_sha256']!=digest(ROOT/'protocols/lean_raw_two_window_control_v1.json') or ident['runtime_sha256']!=digest(ROOT/'modeling/lean_token_window_storage.py') or ident['policy_sha256']!=digest(ROOT/'modeling/token_memory_window.py') or ident['observer_sha256']!=digest(ROOT/'modeling/lean_token_storage_observer.py') or any(ident['patch_sha256'][k]!=digest(fork/k) for k in ('sam2/modeling/sam2_base.py','sam2/sam2_video_predictor_topk_insurance.py')):raise ValueError('Raw-two source changed')
 merge=read(a.merged/'completion.json')
 if not merge['passed'] or digest(a.merged/'results.json')!=merge['results_sha256'] or digest(a.historical/'comparison.json')!=merge['historical_comparison_sha256']:raise ValueError('Historical or merged reference changed')
 source=Path(read(a.merged/'locations.json')[a.sequence]);orig=read(source/'records.json')[0];sd=read(source/'completion.json');folder=source/a.sequence
 if not sd['passed'] or digest(source/'records.json')!=sd['records_sha256'] or digest(folder/'frames.json')!=orig['frames_sha256'] or digest(folder/'prompt.json')!=orig['prompt_sha256']:raise ValueError('Original source changed')
 originals=read(folder/'frames.json');first=read(folder/'prompt.json')['first_detection_index'];runs={'baseline':a.baseline,'raw_window':a.raw_window,'fused_window':a.fused_window,'raw_two':a.raw_two};records={};positions={};saved_rows={}
 for name,path in runs.items():
  done=read(path/'completion.json');r=read(path/'records.json')
  if not done['passed'] or digest(path/'records.json')!=done['records_sha256'] or digest(path/'frames.json')!=r['frames_sha256'] or digest(path/'positions.json')!=r['positions_sha256'] or not read(path/'parameter_integrity_after.json')['unchanged']:raise ValueError('Run record invalid')
  records[name]=r;positions[name]=read(path/'positions.json');saved_rows[name]=read(path/'frames.json')
 ob=read(a.raw_two/'token_window_observation.json');outer=read(a.raw_two/'raw_two_control_receipt.json')
 if not ob['passed'] or ob['identity']!=ident or ob['records_sha256']!=digest(a.raw_two/'records.json') or ob['reference_records_sha256']!=digest(a.baseline/'records.json') or digest(a.raw_two/'token_window_events.json')!=ob['events_sha256'] or not outer['passed'] or outer['dispatcher_sha256']!=digest(ROOT/'scripts/run_lean_raw_two_control.py') or outer['protocol_sha256']!=digest(ROOT/'protocols/lean_raw_two_window_control_v1.json') or outer['delegate_receipt_sha256']!=digest(a.raw_two/'token_window_observation.json'):raise ValueError('Raw-two provenance invalid')
 fv=read(a.fused_verification/'verification.json');fr=read(a.fused_verification/'fused_control_receipt.json');fo=read(a.fused_window/'token_window_observation.json');fc=read(a.fused_window/'fused_control_receipt.json')
 if not fv['passed'] or fv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or not fr['passed'] or fr['delegate_receipt_sha256']!=digest(a.fused_verification/'verification.json') or fr['dispatcher_sha256']!=digest(ROOT/'scripts/run_lean_fused_window_control.py') or fr['protocol_sha256']!=digest(ROOT/'protocols/lean_fused_token_window_control_v1.json'):raise ValueError('Fused gate invalid')
 if not fo['passed'] or fo['identity']!=fv['identity'] or fo['records_sha256']!=digest(a.fused_window/'records.json') or fo['reference_records_sha256']!=digest(a.baseline/'records.json') or digest(a.fused_window/'token_window_events.json')!=fo['events_sha256'] or not fc['passed'] or fc['delegate_receipt_sha256']!=digest(a.fused_window/'token_window_observation.json') or fc['dispatcher_sha256']!=digest(ROOT/'scripts/run_lean_fused_window_control.py') or fc['protocol_sha256']!=digest(ROOT/'protocols/lean_fused_token_window_control_v1.json'):raise ValueError('Fused provenance invalid')
 for name,path in runs.items():
  for key in ('positions','selections','memory_events'):
   if digest(path/f'{key}.json')!=records[name][key+'_sha256']:raise ValueError('Trace changed')
 verify_events(a.fused_window,first,len(originals),positions['baseline'])
 if any(x.get('history_read_mode')!='author_fused_old_tail' for x in read(a.fused_window/'token_window_events.json') if x['stage']=='after_read'):raise ValueError('Fused path inactive')
 frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}[a.sequence]
 if any(len(r['mask_sha256'])!=len(frames) for r in records.values()) or len(originals)!=len(frames):raise ValueError('Frame count differs')
 for r in records.values():
  if r['mask_sha256'][:first+3]!=records['baseline']['mask_sha256'][:first+3]:raise ValueError('Causal prompt prefix differs')
 telemetry=verify_events(a.raw_two,first,len(frames),positions['baseline']);reads=[x for x in read(a.raw_two/'token_window_events.json') if x['stage']=='after_read']
 if any(x.get('history_read_mode')!='raw_two_latest' for x in reads):raise ValueError('Raw-two path inactive')
 target=[json.loads(line) for line in (a.historical/'full'/a.sequence/'frames.jsonl').read_text().splitlines()];target_present={x['frame'] for x in target if x['present']};values=[]
 for i,f in enumerate(frames):
  if digest(f.image_path)!=originals[i]['image_sha256'] or digest(f.mask_path)!=originals[i]['label_sha256']:raise ValueError('Input changed')
  gt=load_label_mask(f.mask_path,[1])==1;present=bool(gt.any());scores={}
  for name,path in runs.items():
   h=records[name]['mask_sha256'][i];file=path/f'{i:06d}.png'
   if digest(file)!=h:raise ValueError('Mask changed')
   mask=np.asarray(Image.open(file))>0
   if mask.shape!=gt.shape:raise ValueError('Mask shape differs')
   den=int(mask.sum()+gt.sum());score=2*int((mask&gt).sum())/den if den else 1.;alarm=bool(mask.any()) if not present else None
   stored=saved_rows[name][i]
   if stored['frame']!=i or stored['present']!=present or stored['mask_sha256']!=h or abs(stored['dice']-score)>1e-12 or stored['false_alarm']!=alarm:raise ValueError('Stored metric differs')
   scores[name]={'dice':score,'false_alarm':alarm,'mask_sha256':h}
  prior=aud['sequences'][a.sequence]['per_frame'][i]
  if prior['present']!=present or any(abs(scores[k]['dice']-prior['scores'][q]['dice'])>1e-12 or scores[k]['false_alarm']!=prior['scores'][q]['false_alarm'] for k,q in (('baseline','baseline'),('raw_window','candidate'))):raise ValueError('Prior audited score differs')
  values.append({'frame':i,'present':present,'matched':i in target_present,'scores':scores})
 if any(values[t['frame']]['present']!=t['present'] for t in target):raise ValueError('Historical target mismatch')
 result={}
 for name in runs:
  vals=[r['scores'][name]['dice'] for r in values if r['matched']];empty=[r['scores'][name]['false_alarm'] for r in values if not r['present']];keys=positions[name]
  result[name]={'matched_present_dice':float(np.mean(vals)),'matched_present_count':len(vals),'false_alarms':sum(empty),'empty_count':len(empty),
                'mean_spatial_keys':float(np.mean([z['spatial_keys'] for z in keys])),'max_spatial_keys':max(z['spatial_keys'] for z in keys)}
 fusedpos=positions['fused_window'];twopos=positions['raw_two']
 if len(fusedpos)!=len(twopos):raise ValueError('Read count differs')
 matches=sum(x['spatial_keys']==y['spatial_keys'] for x,y in zip(fusedpos,twopos));ptr=sum(x['pointer_keys']==y['pointer_keys'] for x,y in zip(fusedpos,twopos))
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 payload={'scope':f'Retrospective selected {a.sequence}; all arms use the same automatic prompt and video frames.','conditions':result,'frames':values,'raw_two_telemetry':telemetry,
          'matched_spatial_key_counts':matches,'matched_pointer_key_counts':ptr,'read_count':len(fusedpos),'raw_audit_sha256':audit_done['results_sha256'],'source_sha256':digest(Path(__file__)), 'protocol_sha256':digest(ROOT/'protocols/lean_raw_fused_priority_followup_v1.json')}
 (out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
 lines=[f'# {a.sequence}: raw-two matched-read control','',payload['scope'],'','| Arm | Matched present Dice | Empty false alarms | Mean spatial keys/read |','| --- | ---: | ---: | ---: |']
 for name,x in result.items():lines.append(f"| {name} | {x['matched_present_dice']:.6f} | {x['false_alarms']} / {x['empty_count']} | {x['mean_spatial_keys']:.1f} |")
 lines+=['',f"Raw-two and fused spatial-key counts match on {matches}/{len(fusedpos)} reads; pointer-key counts match on {ptr}/{len(fusedpos)}. Equal counts do not establish identical tokens or histories. Both use the same storage budget, but predictions can affect subsequent token selection. These are selected, previously inspected video frames, not a same-state causal ablation or proof of generalization."]
 (out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'source_sha256':digest(Path(__file__)), 'protocol_sha256':digest(ROOT/'protocols/lean_raw_fused_priority_followup_v1.json')},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
