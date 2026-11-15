#!/usr/bin/env python3
"""Score frozen spread and recent windows on the same annotated video."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser();p.add_argument('--sequence',choices=('seq8','seq11','seq20','seq23'),required=True)
for k in ('run','recent','verification','recent-audit','merged','historical','crop-baseline','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();v=read(a.verification/'verification.json');prior=read(a.recent_audit/'results.json');prior_done=read(a.recent_audit/'completion.json');merged=read(a.merged/'completion.json')
if not v['passed'] or v['sequence']!=a.sequence or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_temporal_spread_run.py') or v['run_observation_sha256']!=digest(a.run/'token_window_observation.json') or v['recent_observation_sha256']!=digest(a.recent/'token_window_observation.json') or v['run_frames_sha256']!=digest(a.run/'frames.json') or v['recent_frames_sha256']!=digest(a.recent/'frames.json') or not prior_done['passed'] or prior_done['results_sha256']!=digest(a.recent_audit/'results.json') or prior_done['source_sha256']!=digest(ROOT/'scripts/audit_lean_token_window.py') or not merged['passed'] or merged['results_sha256']!=digest(a.merged/'results.json') or merged['historical_comparison_sha256']!=digest(a.historical/'comparison.json'):raise ValueError('Frozen run or reference changed')
frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}[a.sequence];source=Path(read(a.merged/'locations.json')[a.sequence]);original=read(source/a.sequence/'frames.json');source_done=read(source/'completion.json');source_records=read(source/'records.json')[0]
if not source_done['passed'] or source_done['records_sha256']!=digest(source/'records.json') or source_records['frames_sha256']!=digest(source/a.sequence/'frames.json'):raise ValueError('Original source changed')
if a.sequence=='seq23':
 crop_done=read(a.crop_baseline/'completion.json');old=read(a.crop_baseline/'frames.json')
 if not crop_done['passed'] or crop_done['frames_sha256']!=digest(a.crop_baseline/'frames.json'):raise ValueError('Crop reference changed')
else:old=original
rows=[];runrows=read(a.run/'frames.json');recentrows=read(a.recent/'frames.json');hist=[json.loads(line) for line in (a.historical/'full'/a.sequence/'frames.jsonl').read_text().splitlines()];matched={x['frame'] for x in hist if x['present']};audit=prior['sequences'][a.sequence]['per_frame']
if not(len(frames)==len(old)==len(runrows)==len(recentrows)==len(audit)):raise ValueError('Frame coverage differs')
for i,f in enumerate(frames):
 if digest(f.image_path)!=old[i]['image_sha256'] or digest(f.mask_path)!=old[i]['label_sha256']:raise ValueError('Image or label changed')
 truth=load_label_mask(f.mask_path,[1])==1;present=bool(truth.any())
 if a.sequence=='seq23' and truth[:,:VIEWPORT[0]].any():raise ValueError('Labeled foreground outside viewport')
 scores={}
 for arm,folder,stored in (('recent',a.recent,recentrows[i]),('spread',a.run,runrows[i])):
  path=folder/f'{i:06d}.png';h=digest(path)
  if h!=stored['mask_sha256']:raise ValueError('Mask changed')
  pred=np.asarray(Image.open(path))>0
  if pred.shape!=truth.shape:raise ValueError('Prediction shape differs')
  den=int(pred.sum()+truth.sum());dice=2*int((pred&truth).sum())/den if den else 1.;fa=bool(pred.any()) if not present else None
  if stored['present']!=present or abs(stored['dice']-dice)>1e-12 or stored['false_alarm']!=fa:raise ValueError('Stored metric differs')
  scores[arm]={'dice':dice,'false_alarm':fa,'mask_sha256':h}
 ref=audit[i]
 if ref['present']!=present or abs(ref['scores']['candidate']['dice']-scores['recent']['dice'])>1e-12 or ref['scores']['candidate']['false_alarm']!=scores['recent']['false_alarm']:raise ValueError('Prior recent-window audit differs')
 rows.append({'frame':i,'present':present,'matched':i in matched,'label_sha256':old[i]['label_sha256'],'image_sha256':old[i]['image_sha256'],'scores':scores})
if any(rows[t['frame']]['present']!=t['present'] for t in hist):raise ValueError('Historical target labels differ')
summary={}
for name in ('recent','spread'):
 vals=[r['scores'][name]['dice'] for r in rows if r['matched']];empty=[r['scores'][name]['false_alarm'] for r in rows if not r['present']]
 summary[name]={'matched_present_dice':float(np.mean(vals)),'matched_present_count':len(vals),'false_alarms':sum(empty),'empty_count':len(empty)}
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
result={'sequence':a.sequence,'scope':'Previously inspected automatic-prompt video; local token-budgeted full-propagation comparison, not independent generalization.','summary':summary,'frames':rows,'verification_sha256':digest(a.verification/'verification.json'),'recent_audit_sha256':digest(a.recent_audit/'results.json'),'verifier':v,'runner_sha256':digest(Path(__file__))}
(out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
r,s=summary['recent'],summary['spread'];lines=[f'# {a.sequence}: temporal-spread versus recent token window','',result['scope'],'','| Policy | Matched-present Dice | Labeled-empty false alarms |','| --- | ---: | ---: |',f"| Recent | {r['matched_present_dice']:.6f} | {r['false_alarms']} / {r['empty_count']} |",f"| Temporal spread | {s['matched_present_dice']:.6f} | {s['false_alarms']} / {s['empty_count']} |",'',f"Spatial read-key counts match on {v['same_spatial_key_count_reads']}/{v['read_count']} reads; pointer counts match on {v['same_pointer_key_count_reads']}/{v['read_count']}. Nonrecent history was retained on {v['nonrecent_admissions']}/{v['admissions']} admissions; maximum retained age was {v['max_retained_age']} frames.",'','Both methods use the same automatic initialization and nominal represented-feature budget; past memory features, exact ages and propagation feedback may differ. Stored feature tokens plus reserved insurance are counted, not full RAM or latency. All frames were previously inspected; initial empty labels can be incomplete.','']
(out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md'),'runner_sha256':digest(Path(__file__))},indent=2)+'\n');print('\n'.join(lines))
