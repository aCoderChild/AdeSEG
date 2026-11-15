#!/usr/bin/env python3
"""Independently verify post-hoc mask-scale summaries."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def measure(path):
 with Image.open(path) as im:gray=np.asarray(im.convert('L'))
 m=gray>(0 if gray.max()<=1 else 127);yy,xx=np.where(m)
 if not len(xx):raise ValueError('Expected positive mask')
 return float((xx.max()-xx.min()+1)*(yy.max()-yy.min()+1)/m.size),float(np.count_nonzero(m)/m.size)
p=argparse.ArgumentParser()
for k in ('audit','dataset','c6-audit','c6-inputs','priority-box-audit','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();done=read(a.audit/'completion.json');d=read(a.audit/'results.json');dataset=read(a.dataset/'dataset.json');c=read(a.c6_audit/'results.json');inputs=read(a.c6_inputs/'inputs.json');priority=read(a.priority_box_audit/'results.json')
if not done['passed'] or done['results_sha256']!=digest(a.audit/'results.json') or done['report_sha256']!=digest(a.audit/'report.md') or d['dataset_sha256']!=digest(a.dataset/'dataset.json') or d['c6_audit_sha256']!=digest(a.c6_audit/'results.json') or d['priority_box_audit_sha256']!=digest(a.priority_box_audit/'results.json') or d['runner_sha256']!=digest(ROOT/'scripts/audit_polypgen_detector_scale.py') or d['protocol_sha256']!=digest(ROOT/'protocols/polypgen_detector_scale_diagnostic_v1.json'):raise ValueError('Audit changed')
by_hash={s:{r['image_sha256']:r for r in d['groups'][s]} for s in ('train','val')};samples={'train':[],'val':[]};c6_inputs={r['id']:r for r in inputs['inputs']};c6_by_path={c6_inputs[r['id']]['image']:r for r in c['rows']}
for item in dataset['included']:
 if not item['boxes_xyxy']:continue
 split=item['split'];row=by_hash[split][item['image_sha256']];path=Path(item['source_mask'])
 if digest(path)!=item['mask_sha256'] or row['mask_sha256']!=item['mask_sha256']:raise ValueError('Mask changed')
 box,frac=measure(path)
 if abs(box-row['union_box_fraction'])>1e-12 or abs(frac-row['mask_fraction'])>1e-12:raise ValueError('Scale differs')
 if split=='val':
  matched=c6_by_path[item['staged_image']]
  if row['detected']!={name:matched['scores'][name]['detected'] for name in ('original','candidate')}:raise ValueError('Detection grouping changed')
 samples[split].append(row)
for split,rows in samples.items():
 if len(rows)!=len(by_hash[split]) or len(rows)!=d['summaries'][split]['images']:raise ValueError('Sample coverage changed')
 arr=np.array([r['union_box_fraction'] for r in rows]);f=np.array([r['mask_fraction'] for r in rows]);s=d['summaries'][split]
 for q in (0,.25,.5,.75,1):
  if abs(s['union_box_fraction_quantiles'][str(q)]-float(np.quantile(arr,q)))>1e-12:raise ValueError('Quantile differs')
 if abs(s['mask_fraction_median']-float(np.median(f)))>1e-12:raise ValueError('Median differs')
for name in ('original','candidate'):
 for status in (True,False):
  rows=[x for x in samples['val'] if x['detected'][name]==status];summary=d['strata'][name][str(status).lower()]
  if (summary is None)!= (len(rows)==0):raise ValueError('Stratum differs')
  if summary and (summary['images']!=len(rows) or abs(summary['union_box_fraction_quantiles']['0.5']-float(np.median([x['union_box_fraction'] for x in rows])))>1e-12):raise ValueError('Stratum summary differs')
videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};pri={(x['sequence'],x['frame']):x for x in priority['results'] if x['present']}
for row in d['priority']:
 k=(row['sequence'],row['frame']);ref=pri[k];path=videos[k[0]][k[1]].mask_path
 if digest(path)!=row['label_sha256'] or row['label_sha256']!=ref['label_sha256']:raise ValueError('Priority mask changed')
 box,frac=measure(path)
 if abs(box-row['union_box_fraction'])>1e-12 or abs(frac-row['mask_fraction'])>1e-12:raise ValueError('Priority area differs')
 for split,items in samples.items():
  rank=float(np.mean([x['union_box_fraction']<=box for x in items]))
  if abs(rank-row['still_fraction_at_or_below'][split])>1e-12:raise ValueError('Percentile differs')
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'train_positive':len(samples['train']),'val_positive':len(samples['val']),'priority_present':len(d['priority']),'results_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__))},indent=2)+'\n');print('Mask scales, quantiles, detection strata and selected video values independently verified.')
