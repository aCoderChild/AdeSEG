#!/usr/bin/env python3
"""Independently compare fused and raw token-window reads on fixed videos."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output
from scripts.audit_lean_token_window import verify_events
CONTROL=ROOT/'external/Lean-SAM2-fused-window-control'
PROTOCOL=ROOT/'protocols/lean_fused_token_window_control_v1.json'
DISPATCH=ROOT/'scripts/run_lean_fused_window_control.py'
PATCHED=('sam2/modeling/sam2_base.py','sam2/sam2_video_predictor_topk_insurance.py')
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('root','raw-root','raw-audit','indexed-root','crop-baseline','merged','historical','verification','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--sequence',action='append',choices=('seq11','seq23'))
 a=p.parse_args();raw_done=read(a.raw_audit/'completion.json');raw=read(a.raw_audit/'results.json');v=read(a.verification/'verification.json');vr=read(a.verification/'fused_control_receipt.json');merge=read(a.merged/'completion.json')
 if not raw_done['passed'] or digest(a.raw_audit/'results.json')!=raw_done['results_sha256'] or raw_done['source_sha256']!=digest(ROOT/'scripts/audit_lean_token_window.py') or not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or not vr['passed'] or vr['dispatcher_sha256']!=digest(DISPATCH) or vr['delegate_receipt_sha256']!=digest(a.verification/'verification.json') or vr['protocol_sha256']!=digest(PROTOCOL):raise ValueError('Gate or raw audit changed')
 ident=v['identity']
 if ident['runner_sha256']!=digest(ROOT/'scripts/run_lean_token_window.py') or ident['protocol_sha256']!=digest(PROTOCOL) or ident['runtime_sha256']!=digest(ROOT/'modeling/lean_token_window_storage.py') or ident['policy_sha256']!=digest(ROOT/'modeling/token_memory_window.py') or ident['observer_sha256']!=digest(ROOT/'modeling/lean_token_storage_observer.py') or any(ident['patch_sha256'][x]!=digest(CONTROL/x) for x in PATCHED):raise ValueError('Control source changed')
 if not merge['passed'] or digest(a.merged/'results.json')!=merge['results_sha256'] or digest(a.historical/'comparison.json')!=merge['historical_comparison_sha256']:raise ValueError('Merged or historical reference changed')
 locations=read(a.merged/'locations.json');videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};results={}
 for name in (a.sequence or ('seq11','seq23')):
  frames=videos[name];source=Path(locations[name]);orig=read(source/'records.json')[0];orig_done=read(source/'completion.json');folder=source/name
  if not orig_done['passed'] or digest(source/'records.json')!=orig_done['records_sha256'] or digest(folder/'frames.json')!=orig['frames_sha256']:raise ValueError('Original input receipt changed')
  originals=read(folder/'frames.json');window=a.raw_root/f'{name}_attempt01';control=a.root/f'{name}_attempt01';rawob=read(window/'token_window_observation.json');ob=read(control/'token_window_observation.json');outer=read(control/'fused_control_receipt.json')
  if not rawob['passed'] or not ob['passed'] or not outer['passed'] or outer['dispatcher_sha256']!=digest(DISPATCH) or outer['protocol_sha256']!=digest(PROTOCOL) or outer['delegate_receipt_sha256']!=digest(control/'token_window_observation.json') or digest(control/'token_window_events.json')!=ob['events_sha256'] or not read(control/'parameter_integrity_after.json')['unchanged'] or not read(window/'parameter_integrity_after.json')['unchanged']:
   raise ValueError('Candidate receipt invalid')
  if name=='seq23':
   baseline=a.crop_baseline;bd=read(baseline/'completion.json');cd=read(control/'completion.json');wd=read(window/'completion.json')
   if not bd['passed'] or not cd['passed'] or not wd['passed'] or ob['source_sha256']!=digest(ROOT/'scripts/run_seq23_token_window.py') or ob['verification_sha256']!=digest(a.verification/'verification.json') or ob['baseline_completion_sha256']!=digest(baseline/'completion.json') or digest(control/'frames.json')!=cd['frames_sha256'] or digest(window/'frames.json')!=wd['frames_sha256'] or digest(baseline/'frames.json')!=bd['frames_sha256'] or digest(control/'prompt.json')!=bd['prompt_sha256'] or digest(window/'prompt.json')!=bd['prompt_sha256']:
    raise ValueError('Cropped setup differs')
   if digest(control/'positions.json')!=cd['positions_sha256'] or digest(window/'positions.json')!=wd['positions_sha256'] or digest(baseline/'positions.json')!=bd['positions_sha256']:raise ValueError('Cropped position trace changed')
   br=read(baseline/'frames.json');wr=read(window/'frames.json');cr=read(control/'frames.json');bh=[r['mask_sha256'] for r in br];wh=[r['mask_sha256'] for r in wr];ch=[r['mask_sha256'] for r in cr];first=read(baseline/'prompt.json')['first_detection_index']
  else:
   baseline=a.indexed_root/'part_a_attempt01'/name;bd=read(baseline/'completion.json');brc=read(baseline/'records.json');cd=read(control/'completion.json');crc=read(control/'records.json');wd=read(window/'completion.json');wrc=read(window/'records.json')
   if not bd['passed'] or not cd['passed'] or not wd['passed'] or digest(baseline/'records.json')!=bd['records_sha256'] or digest(control/'records.json')!=cd['records_sha256'] or digest(window/'records.json')!=wd['records_sha256'] or ob['identity']!=ident or ob['records_sha256']!=cd['records_sha256'] or ob['reference_records_sha256']!=bd['records_sha256']:
    raise ValueError('Indexed setup differs')
   for key in ('frames','positions','selections','memory_events'):
    if digest(control/f'{key}.json')!=crc[key+'_sha256'] or digest(window/f'{key}.json')!=wrc[key+'_sha256'] or digest(baseline/f'{key}.json')!=brc[key+'_sha256']:raise ValueError('Trace changed')
   br=read(baseline/'frames.json');wr=read(window/'frames.json');cr=read(control/'frames.json');bh=brc['mask_sha256'];wh=wrc['mask_sha256'];ch=crc['mask_sha256'];first=read(folder/'prompt.json')['first_detection_index']
  if not(len(frames)==len(originals)==len(br)==len(wr)==len(cr)==len(bh)==len(wh)==len(ch)):raise ValueError('Frame coverage differs')
  if bh[:first+3]!=wh[:first+3] or bh[:first+3]!=ch[:first+3]:raise ValueError('Prompt causal prefix differs')
  positions=read(baseline/'positions.json');telem=verify_events(control,first,len(frames),positions);rawtelem=verify_events(window,first,len(frames),positions)
  events=read(control/'token_window_events.json');reads=[e for e in events if e['stage']=='after_read']
  if any(e.get('history_read_mode')!='author_fused_old_tail' for e in reads) or max(e.get('fusion_update_count',0) for e in reads)<1:raise ValueError('Fused history was not used')
  targets=[json.loads(line) for line in (a.historical/'full'/name/'frames.jsonl').read_text().splitlines()];target_set={x['frame'] for x in targets if x['present']};rows=[]
  for i,(f,o) in enumerate(zip(frames,originals)):
   if digest(f.image_path)!=o['image_sha256'] or digest(f.mask_path)!=o['label_sha256']:raise ValueError('Frame or label changed')
   truth=load_label_mask(f.mask_path,[1])==1;present=bool(truth.any());scores={}
   for arm,path,h in [('baseline',baseline/f'{i:06d}.png',bh[i]),('raw_window',window/f'{i:06d}.png',wh[i]),('fused_window',control/f'{i:06d}.png',ch[i])]:
    if digest(path)!=h:raise ValueError('Saved mask changed')
    pred=np.asarray(Image.open(path))>0
    if pred.shape!=truth.shape:raise ValueError('Mask geometry differs')
    den=int(pred.sum()+truth.sum());scores[arm]={'dice':2*int((pred&truth).sum())/den if den else 1.,'false_alarm':bool(pred.any()) if not present else None}
   prior=raw['sequences'][name]['per_frame'][i]
   if prior['present']!=present or any(abs(scores[k]['dice']-prior['scores'][q]['dice'])>1e-12 or scores[k]['false_alarm']!=prior['scores'][q]['false_alarm'] for k,q in [('baseline','baseline'),('raw_window','candidate')]):raise ValueError('Raw audit score changed')
   rows.append({'frame':i,'present':present,'matched':i in target_set,'scores':scores})
  if any(rows[t['frame']]['present']!=t['present'] for t in targets):raise ValueError('Historical target labels differ')
  result={'frames':len(frames),'telemetry':telem,'raw_telemetry':rawtelem,'same_masks_fused_vs_baseline':sum(x==y for x,y in zip(ch,bh)),
          'same_masks_fused_vs_raw':sum(x==y for x,y in zip(ch,wh)),'fused_updates':max(e['fusion_update_count'] for e in reads),'per_frame':rows}
  for arm in ('baseline','raw_window','fused_window'):
   vals=[r['scores'][arm]['dice'] for r in rows if r['matched']];empty=[r['scores'][arm]['false_alarm'] for r in rows if not r['present']]
   result[arm]={'matched_present_dice':float(np.mean(vals)),'matched_present_count':len(vals),'all_empty_false_alarms':sum(empty),'all_empty_count':len(empty)}
  results[name]=result
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 payload={'scope':'Selected seq11/23 automatic videos; shared prompt, input and represented spatial-token budget for raw and fused window arms. Seq23 uses no-inset input.','sequences':results,
  'raw_audit_sha256':raw_done['results_sha256'],'verification_sha256':digest(a.verification/'verification.json'),'source_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
 lines=['# Fused versus raw token-window reads','',payload['scope'],'','| Sequence | Original indexed baseline | Raw token window | Fused token window | Empty false alarms baseline / raw / fused |', '| --- | ---: | ---: | ---: | ---: |']
 for name,x in results.items():
  b=x['baseline'];r=x['raw_window'];f=x['fused_window'];lines.append(f"| {name} | {b['matched_present_dice']:.6f} | {r['matched_present_dice']:.6f} | {f['matched_present_dice']:.6f} | {b['all_empty_false_alarms']} / {r['all_empty_false_alarms']} / {f['all_empty_false_alarms']} ({b['all_empty_count']}) |")
 lines+=['','These are selected, previously inspected videos. Raw and fused arms preserve the same budget but read different numbers and representations of memory tokens; predictions and subsequent selections can diverge. The comparison does not isolate a same-state causal effect or prove generalization. The existing author fusion reliability weighting remains only in the fused control; no new confidence gate or threshold was introduced. The represented token budget excludes pointer, position, mask and auxiliary storage and transient allocation.','']
 (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'source_sha256':digest(Path(__file__))},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
