#!/usr/bin/env python3
"""Score frozen paired top-one detector boxes on the C6 validation set."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_binary_mask
from evaluation.prompt_geometry_audit import evaluate_prompt_geometry
from scripts.run_prompt_feature_refinement import box_iou
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def measure(box,truth,gt_boxes):
 present=bool(truth.any())
 if box is None:return {'detected':False,'best_gt_box_iou':0. if present else None,'mask_coverage':0. if present else None,'empty_false_alarm':False if not present else None}
 stats=evaluate_prompt_geometry(box,np.zeros(truth.shape,dtype=bool),truth)
 return {'detected':True,'best_gt_box_iou':max(box_iou(box,g) for g in gt_boxes) if present else None,'mask_coverage':stats['box_truth_coverage'],'empty_false_alarm':True if not present else None}
def main():
 p=argparse.ArgumentParser()
 for k in ('predictions','prediction-verification','inputs','dataset','dataset-verification','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();pred=read(a.predictions/'completion.json');pv=read(a.prediction_verification/'verification.json');src=read(a.inputs/'inputs.json');d=read(a.dataset/'dataset.json');dv=read(a.dataset_verification/'verification.json')
 if not all(x['passed'] for x in (pred,pv,src,dv)) or pv['first_sha256']!=digest(a.predictions/'completion.json') or pv['inputs_sha256']!=digest(a.inputs/'inputs.json') or pv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_c6_predictions.py') or pred['inputs_sha256']!=pv['inputs_sha256'] or src['dataset_sha256']!=digest(a.dataset/'dataset.json') or dv['dataset_sha256']!=src['dataset_sha256'] or dv['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_pilot_dataset.py'):raise ValueError('Frozen predictions or data changed')
 by_path={x['staged_image']:x for x in d['included'] if x['split']=='val'};rows=[]
 for event,inp in zip(pred['events'],src['inputs'],strict=True):
  if event['id']!=inp['id'] or event['image_sha256']!=inp['image_sha256'] or digest(Path(inp['image']))!=inp['image_sha256'] or inp['image'] not in by_path:raise ValueError('Input identity changed')
  entry=by_path[inp['image']];mask=Path(entry['source_mask'])
  if digest(mask)!=entry['mask_sha256'] or [digest(Path(x)) for x in entry['source_boxes']]!=entry['source_boxes_sha256']:raise ValueError('Validation label changed')
  truth=load_binary_mask(mask).astype(bool);gt=entry['boxes_xyxy']
  if truth.shape!=(inp['height'],inp['width']) or bool(truth.any())!=bool(gt):raise ValueError('Label geometry or presence differs')
  values={name:measure(event['boxes'][name]['box'],truth,gt) for name in ('original','candidate')}
  rows.append({'id':inp['id'],'image_sha256':inp['image_sha256'],'mask_sha256':entry['mask_sha256'],'present':bool(truth.any()),'gt_boxes':gt,'scores':values})
 positive=[r for r in rows if r['present']];negative=[r for r in rows if not r['present']]
 if len(positive)!=83 or len(negative)!=5:raise ValueError('C6 label counts changed')
 summary={}
 for name in ('original','candidate'):
  summary[name]={'positive_detections':sum(r['scores'][name]['detected'] for r in positive),'positive_total':len(positive),'mean_best_box_iou':float(np.mean([r['scores'][name]['best_gt_box_iou'] for r in positive])),'mean_mask_coverage':float(np.mean([r['scores'][name]['mask_coverage'] for r in positive])),'empty_false_alarms':sum(r['scores'][name]['empty_false_alarm'] for r in negative),'empty_total':len(negative)}
 diff=np.asarray([r['scores']['candidate']['best_gt_box_iou']-r['scores']['original']['best_gt_box_iou'] for r in positive]);paired={'candidate_wins':int((diff>0).sum()),'ties':int((diff==0).sum()),'candidate_losses':int((diff<0).sum()),'mean_iou_difference':float(diff.mean())}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'scope':'Previously selected C6 validation images, paired top-one boxes; not independent final testing.','predictions_sha256':digest(a.predictions/'completion.json'),'inputs_sha256':digest(a.inputs/'inputs.json'),'dataset_sha256':digest(a.dataset/'dataset.json'),'runner_sha256':digest(Path(__file__)),'summary':summary,'paired':paired,'rows':rows}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');lines=['# C6 matched top-one detector diagnosis','','Both detectors use the same 88 C6 still images and the existing 0.25 detection setting. The new checkpoint was selected on C6; the old detector’s training split is unknown, so this is not an independent or equal-exposure comparison. On positive images the single highest-confidence box is compared with whichever supplied ground-truth box it overlaps best, and mask coverage is measured against all labeled foreground. This is not COCO mAP or a full-video result.','','| Detector | Positive images with a box | Mean best GT-box IoU | Mean foreground coverage | Labeled-empty false alarms |','| --- | ---: | ---: | ---: | ---: |']
 for name,s in summary.items():lines.append(f"| {name} | {s['positive_detections']} / {s['positive_total']} | {s['mean_best_box_iou']:.4f} | {s['mean_mask_coverage']:.4f} | {s['empty_false_alarms']} / {s['empty_total']} |")
 lines+=['',f"Paired positive-image IoU comparison: new detector wins {paired['candidate_wins']}, ties {paired['ties']}, loses {paired['candidate_losses']}; mean difference {paired['mean_iou_difference']:+.4f}. Results do not identify why it missed the two previously inspected video initialization frames. There are only five labeled-empty C6 examples; patient/video independence and original-model training exposure remain unknown.",'']
 (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
