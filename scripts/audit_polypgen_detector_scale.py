#!/usr/bin/env python3
"""Describe foreground-mask scale in curated stills and fixed video prompts."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_binary_mask,load_label_mask
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def scale(mask):
 mask=np.asarray(mask,dtype=bool)
 if mask.ndim!=2 or not mask.any():raise ValueError('Expected positive mask')
 y,x=np.nonzero(mask);h,w=mask.shape
 return {'union_box_fraction':float((x.max()-x.min()+1)*(y.max()-y.min()+1)/(h*w)),'mask_fraction':float(mask.sum()/(h*w))}
def describe(rows):
 if not rows:return None
 x=np.array([r['union_box_fraction'] for r in rows]);y=np.array([r['mask_fraction'] for r in rows])
 return {'images':len(rows),'union_box_fraction_quantiles':{str(q):float(np.quantile(x,q)) for q in (0,.25,.5,.75,1)},'mask_fraction_median':float(np.median(y))}
def main():
 p=argparse.ArgumentParser()
 for k in ('dataset','dataset-verification','c6-audit','c6-verification','c6-inputs','priority-box-audit','priority-box-verification','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();d=read(a.dataset/'dataset.json');dv=read(a.dataset_verification/'verification.json');c=read(a.c6_audit/'results.json');cv=read(a.c6_verification/'verification.json');pb=read(a.priority_box_audit/'results.json');pv=read(a.priority_box_verification/'verification.json')
 if not dv['passed'] or dv['dataset_sha256']!=digest(a.dataset/'dataset.json') or dv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_pilot_dataset.py') or not cv['passed'] or cv['audit_sha256']!=digest(a.c6_audit/'results.json') or cv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_c6_detector_audit.py') or not pv['passed'] or pv['audit_sha256']!=digest(a.priority_box_audit/'results.json') or pv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_one_shot_boxes.py'):raise ValueError('Verified data or prior scores changed')
 inputs={r['id']:r for r in read(a.c6_inputs/'inputs.json')['inputs']}
 # Bind the label-free image manifest by its receipt hash.
 if digest(a.c6_inputs/'inputs.json')!=c['inputs_sha256']:raise ValueError('C6 input manifest changed')
 c6_by_path={inputs[r['id']]['image']:r for r in c['rows']};groups={'train':[],'val':[]}
 for row in d['included']:
  if not row['boxes_xyxy']:continue
  path=Path(row['source_mask'])
  if digest(path)!=row['mask_sha256']:raise ValueError('Still-image mask changed')
  x=scale(load_binary_mask(path));item={'split':row['split'],'image_sha256':row['image_sha256'],'mask_sha256':row['mask_sha256'],**x}
  if row['split']=='val':
   old=c6_by_path[row['staged_image']]
   if not old['present'] or old['mask_sha256']!=row['mask_sha256']:raise ValueError('C6 score alignment differs')
   item['detected']={k:old['scores'][k]['detected'] for k in ('original','candidate')}
  groups[row['split']].append(item)
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};priority=[]
 for row in pb['results']:
  if not row['present']:continue
  seq=row['sequence'];i=row['frame'];frame=videos[seq][i]
  if digest(frame.mask_path)!=row['label_sha256']:raise ValueError('Priority mask changed')
  x=scale(load_label_mask(frame.mask_path,[1])==1)
  ranks={k:float(np.mean([v['union_box_fraction']<=x['union_box_fraction'] for v in values])) for k,values in groups.items()}
  priority.append({'sequence':seq,'frame':i,'label_sha256':row['label_sha256'],**x,'still_fraction_at_or_below':ranks})
 if len(groups['train'])!=1257 or len(groups['val'])!=83 or {r['sequence'] for r in priority}!={'seq8','seq11'}:raise ValueError('Positive scope changed')
 summaries={k:describe(v) for k,v in groups.items()};strata={name:{str(value).lower():describe([x for x in groups['val'] if x['detected'][name]==value]) for value in (True,False)} for name in ('original','candidate')}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'scope':'Post-hoc geometry description, not detector causality or independent evaluation.','dataset_sha256':digest(a.dataset/'dataset.json'),'c6_audit_sha256':digest(a.c6_audit/'results.json'),'priority_box_audit_sha256':digest(a.priority_box_audit/'results.json'),'protocol_sha256':digest(ROOT/'protocols/polypgen_detector_scale_diagnostic_v1.json'),'runner_sha256':digest(Path(__file__)),'groups':groups,'summaries':summaries,'strata':strata,'priority':priority}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');lines=['# Post-hoc detector foreground scale diagnosis','','All areas below are tight bounding rectangles around the union binary mask, divided by the full image area. This can enclose multiple polyps and is not necessarily the size of an individual object. The two sequence frames and detector outcomes were previously inspected; C6 selected the checkpoint. This is a descriptive comparison, not evidence of why detection failed.','','| Set | Labeled-present images | Median union-mask box fraction | 25th / 75th percentile |','| --- | ---: | ---: | ---: |']
 for k in ('train','val'):
  s=summaries[k];q=s['union_box_fraction_quantiles'];lines.append(f"| {k} | {s['images']} | {q['0.5']:.4f} | {q['0.25']:.4f} / {q['0.75']:.4f} |")
 lines+=['','| Fixed video prompt | Union-mask box fraction | Fraction of training / C6 positive stills at or below |','| --- | ---: | ---: |']
 for x in priority:lines.append(f"| {x['sequence']} frame {x['frame']} | {x['union_box_fraction']:.4f} | {x['still_fraction_at_or_below']['train']:.1%} / {x['still_fraction_at_or_below']['val']:.1%} |")
 lines+=['','| C6 detector | Candidate status | Images | Median union-mask box fraction |','| --- | --- | ---: | ---: |']
 for name in ('original','candidate'):
  for status in ('true','false'):
   z=strata[name][status];lines.append(f"| {name} | {'detected' if status=='true' else 'no box'} | {z['images'] if z else 0} | {z['union_box_fraction_quantiles']['0.5']:.4f} |" if z else f"| {name} | {'detected' if status=='true' else 'no box'} | 0 | — |")
 lines+=['','Mask sizes, candidate detection and source-center differences can be associated for many reasons; this comparison cannot attribute the video misses to scale. No patient-level independence or full-video improvement is established.','']
 (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
