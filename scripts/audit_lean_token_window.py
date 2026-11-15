#!/usr/bin/env python3
"""Independently audit matched four-sequence token-window predictions."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output
from scripts.run_lean_token_window import FORK,PATCHED,PROTOCOL

def read(p):return json.loads(p.read_text())
def verify_events(path,first,nframes,baseline_positions):
 events=read(path/'token_window_events.json');groups={}
 for e in events:groups.setdefault(e['frame'],[]).append(e)
 if sorted(groups)!=list(range(first+1,nframes)):raise ValueError('Memory event frame coverage differs')
 expected=['after_read','before_admission','after_compression','admission_plan','after_write','after_bank_write'];maximum=evictions=0;max_history=0
 for frame,items in groups.items():
  if [e['stage'] for e in items]!=expected:raise ValueError('Memory event order differs')
  plan=items[3];write=items[4]
  if plan['condition_tokens']!=4096 or plan['new_tokens']!=4096 or plan['insurance_reserve']!=3072 or plan['budget']!=16384 or plan['planned_reserved_tokens']!=write['spatial_tokens']+3072 or plan['selected_frames'][-1]!=frame or plan['selected_frames']!=sorted(set(plan['selected_frames'])) or set(plan['selected_frames'])&set(plan['evict_frames']):raise ValueError('Invalid memory admission')
  for e in items:
   if e['stage']=='admission_plan':continue
   if e['total_tokens']!=e['spatial_tokens']+e['insurance_tokens'] or e['spatial_tokens']+3072>16384 or e['total_tokens']>16384 or e['insurance_tokens']>3072:raise ValueError('Spatial-token budget violated')
   maximum=max(maximum,e['total_tokens'])
  evictions+=len(plan['evict_frames']);max_history=max(max_history,len(plan['selected_frames']))
 positions=read(path/'positions.json')
 if len(positions)!=len(baseline_positions) or [p['pointer_keys'] for p in positions]!=[p['pointer_keys'] for p in baseline_positions]:raise ValueError('Pointer token counts differ')
 return {'reads':len(groups),'max_represented_spatial_plus_insurance':maximum,'evictions':evictions,'max_admitted_history_frames':max_history}

def main():
 p=argparse.ArgumentParser()
 for x in ('root','indexed-root','crop-baseline','merged','historical','verification','output'):p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args();v=read(a.verification/'verification.json');merge=read(a.merged/'completion.json')
 if not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or not merge['passed'] or digest(a.merged/'results.json')!=merge['results_sha256'] or digest(a.historical/'comparison.json')!=merge['historical_comparison_sha256']:
  raise ValueError('Gate or reference provenance invalid')
 ident=v['identity']
 if ident['runner_sha256']!=digest(ROOT/'scripts/run_lean_token_window.py') or ident['protocol_sha256']!=digest(PROTOCOL) or ident['runtime_sha256']!=digest(ROOT/'modeling/lean_token_window_storage.py') or ident['policy_sha256']!=digest(ROOT/'modeling/token_memory_window.py') or ident['observer_sha256']!=digest(ROOT/'modeling/lean_token_storage_observer.py') or any(ident['patch_sha256'][x]!=digest(FORK/x) for x in PATCHED):raise ValueError('Candidate source changed')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};loc=read(a.merged/'locations.json');hist={x['sequence']:x for x in read(a.historical/'comparison.json')['sequences']};results={}
 for name in ('seq8','seq11','seq20','seq23'):
  frames=videos[name];source=Path(loc[name]);old=read(source/'records.json')[0];source_done=read(source/'completion.json');folder=source/name
  if not source_done['passed'] or digest(source/'records.json')!=source_done['records_sha256'] or digest(folder/'frames.json')!=old['frames_sha256'] or digest(folder/'prompt.json')!=old['prompt_sha256']:raise ValueError('Original data receipts changed')
  oldrows=read(folder/'frames.json');run=a.root/f'{name}_attempt01';ob=read(run/'token_window_observation.json');events_path=run/'token_window_events.json'
  if not ob['passed'] or digest(events_path)!=ob['events_sha256'] or not read(run/'parameter_integrity_after.json')['unchanged']:raise ValueError('Candidate observation invalid')
  if name=='seq23':
   baseline=a.crop_baseline;bd=read(baseline/'completion.json');cd=read(run/'completion.json');baseline_rows=read(baseline/'frames.json');candidate_rows=read(run/'frames.json')
   if ob['mode']!='candidate' or ob['source_sha256']!=digest(ROOT/'scripts/run_seq23_token_window.py') or ob['verification_sha256']!=digest(a.verification/'verification.json') or ob['baseline_completion_sha256']!=digest(baseline/'completion.json') or not bd['passed'] or not cd['passed'] or digest(baseline/'frames.json')!=bd['frames_sha256'] or digest(run/'frames.json')!=cd['frames_sha256'] or digest(run/'frames.json')!=ob['records_sha256'] or digest(run/'prompt.json')!=bd['prompt_sha256']:
    raise ValueError('Cropped candidate or baseline invalid')
   first=read(run/'prompt.json')['first_detection_index'];baseline_hashes=[x['mask_sha256'] for x in baseline_rows];candidate_hashes=[x['mask_sha256'] for x in candidate_rows]
   if digest(baseline/'positions.json')!=bd['positions_sha256'] or digest(run/'positions.json')!=cd['positions_sha256']:raise ValueError('Cropped positions changed')
  else:
   refs=list(a.indexed_root.glob(f'part_*_attempt01/{name}'))
   if len(refs)!=1:raise ValueError('Indexed baseline missing')
   baseline=refs[0];br=read(baseline/'records.json');bd=read(baseline/'completion.json');cr=read(run/'records.json');cd=read(run/'completion.json')
   if not bd['passed'] or not cd['passed'] or digest(baseline/'records.json')!=bd['records_sha256'] or digest(run/'records.json')!=cd['records_sha256'] or digest(run/'records.json')!=ob['records_sha256'] or ob['identity']!=ident or ob['variant']!='window' or ob['mode']!='candidate' or ob['reference_records_sha256']!=bd['records_sha256']:
    raise ValueError('Indexed candidate or baseline invalid')
   for key in ('frames','positions','selections','memory_events'):
    if digest(baseline/f'{key}.json')!=br[key+'_sha256'] or digest(run/f'{key}.json')!=cr[key+'_sha256']:raise ValueError('Trace changed')
   baseline_rows=read(baseline/'frames.json');candidate_rows=read(run/'frames.json');baseline_hashes=br['mask_sha256'];candidate_hashes=cr['mask_sha256'];first=read(folder/'prompt.json')['first_detection_index']
  if not(len(frames)==len(oldrows)==len(baseline_rows)==len(candidate_rows)==len(baseline_hashes)==len(candidate_hashes)):raise ValueError('Frame coverage differs')
  if candidate_hashes[:first+3]!=baseline_hashes[:first+3]:raise ValueError('Prompt and pre-intervention prefix differs')
  telemetry=verify_events(run,first,len(frames),read(baseline/'positions.json'))
  scored=[]
  for i,(frame,oldrow,brow,crow) in enumerate(zip(frames,oldrows,baseline_rows,candidate_rows)):
   if digest(frame.image_path)!=oldrow['image_sha256'] or digest(frame.mask_path)!=oldrow['label_sha256']:raise ValueError('Image or label changed')
   gt=load_label_mask(frame.mask_path,[1])==1;present=bool(gt.any());vals={}
   for key,path,row,checksum in [('baseline',baseline/f'{i:06d}.png',brow,baseline_hashes[i]),('candidate',run/f'{i:06d}.png',crow,candidate_hashes[i])]:
    if digest(path)!=checksum or row['mask_sha256']!=checksum:raise ValueError('Prediction changed')
    pred=np.asarray(Image.open(path))>0
    if pred.shape!=gt.shape:raise ValueError('Prediction geometry differs')
    den=int(pred.sum()+gt.sum());dice=2*int((pred&gt).sum())/den if den else 1.;alarm=bool(pred.any()) if not present else None
    if row['present']!=present or not np.isclose(row['dice'],dice,rtol=0,atol=1e-12) or row['false_alarm']!=alarm:raise ValueError('Stored metric differs')
    vals[key]={'dice':dice,'false_alarm':alarm,'predicted_pixels':int(pred.sum())}
   scored.append({'frame':i,'present':present,'scores':vals})
  targets=[json.loads(x) for x in (a.historical/'full'/name/'frames.jsonl').read_text().splitlines()]
  if len(targets)!=hist[name]['targets'] or any(scored[x['frame']]['present']!=x['present'] for x in targets):raise ValueError('Historical frame protocol changed')
  summary={'frames':len(frames),'first_prompt':first,'native_gt_dice':hist[name]['native_dice'],'lean_gt_dice':hist[name]['lean_dice'],'telemetry':telemetry}
  for arm in ('baseline','candidate'):
   all_present=[x['scores'][arm]['dice'] for x in scored if x['present']];all_empty=[x['scores'][arm]['false_alarm'] for x in scored if not x['present']]
   matched_present=[scored[x['frame']]['scores'][arm]['dice'] for x in targets if x['present']];matched_empty=[scored[x['frame']]['scores'][arm]['false_alarm'] for x in targets if not x['present']]
   summary[arm]={'all_present_dice':float(np.mean(all_present)),'all_present_count':len(all_present),'all_empty_false_alarms':sum(all_empty),'all_empty_count':len(all_empty),
    'matched_present_dice':float(np.mean(matched_present)),'matched_present_count':len(matched_present),'matched_empty_false_alarms':sum(matched_empty),'matched_empty_count':len(matched_empty)}
  summary['per_frame']=scored;results[name]=summary
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'scope':'Four previously inspected sequences; both arms use the same automatic prompt and input for each video. Seq23 uses the no-inset crop in both arms. Historical native/Lean use ground-truth initial masks and are unequal-prompt descriptive targets.',
         'sequences':results,'verification_sha256':digest(a.verification/'verification.json'),'source_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Token-budgeted Lean: four-sequence comparison','',result['scope'],'', '| Sequence | Native GT prompt | Lean GT prompt | Automatic baseline | Token window | All-empty false alarms baseline / window |', '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for name,x in results.items():
  b=x['baseline'];c=x['candidate'];lines.append(f"| {name} | {x['native_gt_dice']:.3f} | {x['lean_gt_dice']:.3f} | {b['matched_present_dice']:.3f} | {c['matched_present_dice']:.3f} | {b['all_empty_false_alarms']} / {c['all_empty_false_alarms']} ({b['all_empty_count']}) |")
 lines+=['','Matched Dice averages labeled-present historical target frames per sequence; empty-frame counts cover all labeled-empty frames. The candidate retains the full initial prompt and newest memory, budgets represented spatial plus insurance features at 16,384, eagerly compresses previously selected entries, reads retained raw history and disables the prior fused old-tail replacement. Ages beyond the learned temporal slots reuse the oldest noncondition embedding. This budget excludes pointers, position tensors, masks, auxiliary state and transient allocations; it is not a total RAM cap.','', 'These selected, previously inspected videos are exploratory. GT-prompt references use different initialization; they do not establish a matched automatic improvement. No new detection, confidence gate, learned weights or inference threshold was added. The model was not evaluated as a runtime benchmark.']
 (out/'report.md').write_text('\n'.join(lines)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'source_sha256':digest(Path(__file__))},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
