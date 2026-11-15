#!/usr/bin/env python3
"""Independently score fixed early-FPN contrast in Lean token ranking."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output

def read(p): return json.loads(p.read_text())

def main():
 p=argparse.ArgumentParser(); p.add_argument('--part',type=Path,action='append',required=True)
 for name in ('merged','historical','cpu-root','verification','gate','output'): p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args(); v=read(a.verification); protocol=read(ROOT/'protocols/lean_early_contrast_anchor_v1.json')
 merge=read(a.merged/'completion.json'); oldcpu=read(a.cpu_root/'audited_cohort_attempt01/completion.json')
 if (not v['passed'] or not merge['passed'] or digest(a.merged/'results.json')!=merge['results_sha256']
     or digest(a.historical/'comparison.json')!=merge['historical_comparison_sha256']
     or not oldcpu['passed'] or digest(a.cpu_root/'audited_cohort_attempt01/results.json')!=oldcpu['results_sha256']
     or digest(a.gate/'contrast_intervention.json')!=v['active_intervention_sha256']): raise ValueError('Reference receipts invalid')
 reference=read(a.cpu_root/'audited_cohort_attempt01/results.json')['sequences']; history={x['sequence']:x for x in read(a.historical/'comparison.json')['sequences']}
 locations=read(a.merged/'locations.json')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}; result={}
 for part in a.part:
  setup,done=read(part/'setup.json'),read(part/'completion.json')
  if not done['passed'] or done['sequences']!=setup['sequences'] or setup['verification_sha256']!=digest(a.verification) or setup['runner_sha256']!=v['identity']['runner_sha256']:
   raise ValueError('Partition invalid')
  for name in setup['sequences']:
   if name in result: raise ValueError('Overlapping partitions')
   run=part/name; meta=read(run/'setup.json'); record=read(run/'records.json'); receipt=read(run/'completion.json'); intervention=read(run/'contrast_intervention.json')
   if (not receipt['passed'] or digest(run/'records.json')!=receipt['records_sha256'] or meta['mode']!='candidate'
       or not intervention['passed'] or intervention['identity']!=v['identity'] or intervention['mode']!='study' or intervention['sequence']!=name or intervention['strength']!=.5 or intervention['model_records_sha256']!=digest(run/'records.json') or intervention['contrast_events_sha256']!=digest(run/'contrast_events.json') or meta['sequence']!=name or not read(run/'parameter_integrity_after.json')['unchanged']): raise ValueError(f'{name}: run invalid')
   for key,file in [('positions_sha256','positions.json'),('selections_sha256','selections.json'),('memory_events_sha256','memory_events.json'),('frames_sha256','frames.json')]:
    if digest(run/file)!=record[key]: raise ValueError(f'{name}: {file} changed')
   positions=read(run/'positions.json'); selections=read(run/'selections.json')
   if not positions or any(x['query_rotations']!=4 or x['key_rotations']!=4 for x in positions) or any(x['candidates']!=4096 or x['kept']!=1024 for x in selections): raise ValueError('Position or memory budget changed')
   contrast=read(run/'contrast_events.json')
   if len(contrast)!=intervention['score_events'] or any(x['tokens']!=4096 for x in contrast) or sum(x['mean_abs_score_change']>0 for x in contrast)!=intervention['changed_score_events']:
    raise ValueError('Contrast score events changed')
   refs=[d/name for d in (a.cpu_root/'part_a_attempt01',a.cpu_root/'part_b_attempt01') if (d/name/'completion.json').exists()]
   if len(refs)!=1: raise ValueError('Reference missing or duplicated')
   ref=refs[0]; ref_rec=read(ref/'records.json'); ref_done=read(ref/'completion.json'); ref_rows=read(ref/'frames.json')
   if not ref_done['passed'] or digest(ref/'records.json')!=ref_done['records_sha256'] or digest(ref/'frames.json')!=ref_rec['frames_sha256'] or intervention['reference_records_sha256']!=ref_done['records_sha256']: raise ValueError('Reference changed')
   if digest(ref/'selections.json')!=ref_rec['selections_sha256']: raise ValueError('Reference selections changed')
   old_selections=read(ref/'selections.json')
   selection_key=lambda x:(x['frame_index'],x['source_frame_index'])
   old_selection_map={selection_key(x):x for x in old_selections}
   new_selection_map={selection_key(x):x for x in selections}
   if len(old_selection_map)!=len(old_selections) or len(new_selection_map)!=len(selections): raise ValueError('Ambiguous selection events')
   matched_keys=old_selection_map.keys() & new_selection_map.keys()
   changed_sets=sum(old_selection_map[k]['indices_sha256']!=new_selection_map[k]['indices_sha256'] for k in matched_keys)
   if intervention['changed_masks']!=sum(x!=y for x,y in zip(record['mask_sha256'],ref_rec['mask_sha256'])):
    raise ValueError('Changed-mask count uses the wrong reference')
   if intervention['changed_selections']!=(record['selections_sha256']!=ref_rec['selections_sha256']):
    raise ValueError('Changed-selection flag uses the wrong reference')
   original=Path(locations[name]); old_done=read(original/'completion.json'); old_audit=read(original/'independent_audit.json'); old_rec=read(original/'records.json')[0]; old_folder=original/name
   if not old_done['passed'] or not old_audit['passed'] or digest(original/'records.json')!=old_done['records_sha256'] or old_audit['records_sha256']!=old_done['records_sha256'] or digest(old_folder/'frames.json')!=old_rec['frames_sha256']:
    raise ValueError('Input reference invalid')
   old_rows=read(old_folder/'frames.json'); rows=read(run/'frames.json'); frames=videos[name]
   if not(len(rows)==len(ref_rows)==len(old_rows)==len(frames)==record['frames']==len(record['mask_sha256'])): raise ValueError('Frame count mismatch')
   values=[]; pixel_differences=[]
   for i,(row,rr,old,frame) in enumerate(zip(rows,ref_rows,old_rows,frames)):
    path=run/f'{i:06d}.png'; ref_path=ref/f'{i:06d}.png'
    if (row['frame']!=i or rr['frame']!=i or digest(path)!=row['mask_sha256'] or row['mask_sha256']!=record['mask_sha256'][i]
        or digest(ref_path)!=rr['mask_sha256'] or rr['mask_sha256']!=ref_rec['mask_sha256'][i]
        or digest(frame.image_path)!=old['image_sha256'] or digest(frame.mask_path)!=old['label_sha256']): raise ValueError(f'{name}/{i}: mask/input changed')
    truth=load_label_mask(frame.mask_path,[1])==1; scores=[]; masks=[]
    for x,prev in ((path,row),(ref_path,rr)):
     pred=np.asarray(Image.open(x))>0
     if pred.shape!=truth.shape: raise ValueError('Mask shape changed')
     den=int(pred.sum()+truth.sum()); present=bool(truth.any()); dice=2*int((pred & truth).sum())/den if den else 1.; alarm=bool(pred.any()) if not present else None
     if prev['present']!=present or not np.isclose(prev['dice'],dice,rtol=0,atol=1e-12) or prev['false_alarm']!=alarm: raise ValueError('Scoring mismatch')
     scores.append((dice,alarm)); masks.append(pred)
    pixel_differences.append(float(np.mean(masks[0]!=masks[1])))
    values.append((present,*scores))
   targets=[json.loads(line) for line in (a.historical/'full'/name/'frames.jsonl').read_text().splitlines()]
   if len(targets)!=history[name]['targets'] or any(values[x['frame']][0]!=x['present'] for x in targets): raise ValueError('Historical targets changed')
   cp=[x[1][0] for x in values if x[0]]; rp=[x[2][0] for x in values if x[0]]
   ce=[x[1][1] for x in values if not x[0]]; re=[x[2][1] for x in values if not x[0]]
   ch=[values[x['frame']][1][0] for x in targets if x['present']]; rh=[values[x['frame']][2][0] for x in targets if x['present']]
   che=[values[x['frame']][1][1] for x in targets if not x['present']]; rhe=[values[x['frame']][2][1] for x in targets if not x['present']]
   if not np.isclose(np.mean(rp),reference[name]['candidate_all_dice'],atol=1e-12,rtol=0) or not np.isclose(np.mean(rh),reference[name]['candidate_historical_dice'],atol=1e-12,rtol=0): raise ValueError('Reference audit differs')
   result[name]={'all_present':len(cp),'all_empty':len(ce),'candidate_all_dice':float(np.mean(cp)),'reference_all_dice':float(np.mean(rp)),
    'candidate_all_false_alarms':sum(ce),'reference_all_false_alarms':sum(re),'historical_present':len(ch),'historical_empty':len(che),
    'candidate_historical_dice':float(np.mean(ch)),'reference_historical_dice':float(np.mean(rh)),
    'candidate_historical_false_alarms':sum(che),'reference_historical_false_alarms':sum(rhe),
    'native_gt_dice':history[name]['native_dice'],'lean_gt_dice':history[name]['lean_dice'],'changed_masks':intervention['changed_masks'],
    'matched_selection_events':len(matched_keys),'changed_selection_sets':changed_sets,
    'mean_image_pixel_difference':float(np.mean(pixel_differences)),'max_image_pixel_difference':float(np.max(pixel_differences))}
 if set(result)!=set(protocol['cohort']): raise ValueError('Frozen cohort incomplete')
 rows=list(result.values()); delta=np.array([x['candidate_historical_dice']-x['reference_historical_dice'] for x in rows]); rng=np.random.default_rng(20261114)
 boot=rng.choice(delta,(10000,len(delta)),replace=True).mean(axis=1)
 summary={'scope':'Four user-requested difficult videos selected after prior inspection. Same automatic prompts, weights, indexed positions, frame schedule and token budget; candidate uses a fixed blend of early-FPN contrast into the existing multiscale anchor score. Historical native/Lean used GT prompts.', 'sequences':result,
  'reference_all_dice':float(np.mean([x['reference_all_dice'] for x in rows])),'candidate_all_dice':float(np.mean([x['candidate_all_dice'] for x in rows])),
  'reference_all_false_alarms':sum(x['reference_all_false_alarms'] for x in rows),'candidate_all_false_alarms':sum(x['candidate_all_false_alarms'] for x in rows),'all_empty':sum(x['all_empty'] for x in rows),
  'reference_historical_dice':float(np.mean([x['reference_historical_dice'] for x in rows])),'candidate_historical_dice':float(np.mean([x['candidate_historical_dice'] for x in rows])),
  'delta_ci95':np.quantile(boot,[.025,.975]).tolist(),'reference_historical_false_alarms':sum(x['reference_historical_false_alarms'] for x in rows),
  'candidate_historical_false_alarms':sum(x['candidate_historical_false_alarms'] for x in rows),'historical_empty':sum(x['historical_empty'] for x in rows),
  'below_best_gt_reference':sum(x['candidate_historical_dice']<max(x['native_gt_dice'],x['lean_gt_dice'])-1e-12 for x in rows)}
 out=validate_output(a.output); out.mkdir(parents=True,exist_ok=False); (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
 lines=['# Early-FPN contrast in Lean ranking: independent selected-video audit','',summary['scope'],'', '| Sequence | Native GT | Lean GT | Indexed baseline | Early-contrast ranking | Change | False alarms reference / contrast (all labeled-empty) |','| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
 for name,x in result.items():
  lines.append(f"| {name} | {x['native_gt_dice']:.3f} | {x['lean_gt_dice']:.3f} | {x['reference_historical_dice']:.3f} | {x['candidate_historical_dice']:.3f} | {x['candidate_historical_dice']-x['reference_historical_dice']:+.3f} | {x['reference_all_false_alarms']} / {x['candidate_all_false_alarms']} ({x['all_empty']}) |")
 lines += ['', 'Selection-set and prediction changes (fraction of all image pixels):']
 for name,x in result.items():
  lines.append(f"- {name}: {x['changed_selection_sets']}/{x['matched_selection_events']} matched token selections differ; mean pixel difference {100*x['mean_image_pixel_difference']:.4f}%, maximum {100*x['max_image_pixel_difference']:.4f}%.")
 lines += ['',f"All-frame mean Dice {summary['reference_all_dice']:.4f} / {summary['candidate_all_dice']:.4f}; labeled-empty false alarms {summary['reference_all_false_alarms']} / {summary['candidate_all_false_alarms']} of {summary['all_empty']}.",
  f"Matched mean Dice {summary['reference_historical_dice']:.4f} / {summary['candidate_historical_dice']:.4f}; paired sequence-bootstrap 95% interval for change {summary['delta_ci95']}; matched empty-frame false alarms {summary['reference_historical_false_alarms']} / {summary['candidate_historical_false_alarms']} of {summary['historical_empty']}. Below the better GT-prompt reference: {summary['below_best_gt_reference']}/{len(rows)}.",
  'Selected exploratory results do not establish generalization. The early feature descriptors come from the automatic prompt, which is labeled empty on seq20 and seq23. No labels entered inference; no new re-detection, threshold, or adaptive model gate was used.','']
 (out/'report.md').write_text('\n'.join(lines)); (out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'source_sha256':digest(Path(__file__))},indent=2)+'\n')
 print(json.dumps({k:v for k,v in summary.items() if k!='sequences'},indent=2))
if __name__=='__main__':main()
