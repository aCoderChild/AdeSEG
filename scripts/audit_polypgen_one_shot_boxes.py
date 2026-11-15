#!/usr/bin/env python3
"""Score frozen one-shot detector boxes on their original initialization frames."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from evaluation.prompt_geometry_audit import evaluate_prompt_geometry
from modeling.seq23_viewport import VIEWPORT
from scripts.memory_token_io import digest,validate_output
from scripts.run_prompt_feature_refinement import gt_box,box_iou

def read(p):return json.loads(p.read_text())
def geometry(box,truth):
 present=bool(truth.any())
 if box is None:return {'detected':False,'box_iou':0. if present else None,'box_truth_coverage':0. if present else None,'box_fraction':0.,'labeled_empty_false_alarm':False if not present else None}
 z=evaluate_prompt_geometry(box,np.zeros(truth.shape,dtype=bool),truth)
 return {'detected':True,'box_iou':box_iou(box,gt_box(truth)) if present else None,'box_truth_coverage':z['box_truth_coverage'],'box_fraction':z['box_fraction'],'labeled_empty_false_alarm':True if not present else None}
def main():
 p=argparse.ArgumentParser()
 for k in ('predictions','verification','preflight','crop-reference','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();pred=read(a.predictions/'completion.json');v=read(a.verification/'verification.json');m=read(a.preflight/'manifest.json');crop=read(a.crop_reference/'completion.json')
 if not pred['passed'] or pred['mode']!='candidate' or not v['passed'] or v['predictions_sha256']!=digest(a.predictions/'completion.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_one_shot_predictions.py') or pred['preflight_sha256']!=digest(a.preflight/'manifest.json') or not crop['passed'] or crop['frames_sha256']!=digest(a.crop_reference/'frames.json'):raise ValueError('Frozen predictions or reference changed')
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};results=[]
 for e in pred['events']:
  seq=e['sequence'];i=e['frame'];meta=m['sequences'][seq];frame=videos[seq][i];source=Path(meta['source'])/seq
  if digest(source/'frames.json')!=meta['frames_metadata_sha256'] or digest(frame.image_path)!=e['image_sha256']:raise ValueError('Source metadata changed')
  rows=read(a.crop_reference/'frames.json') if e['viewport'] else read(source/'frames.json')
  if digest(frame.mask_path)!=rows[i]['label_sha256']:raise ValueError('Label changed')
  truth=load_label_mask(frame.mask_path,[1])==1
  if e['viewport'] and np.any(truth[:,:VIEWPORT[0]]):raise ValueError('Annotated foreground outside viewport')
  old=e['original_box_viewport']
  if e['viewport']:old=[old[0]+VIEWPORT[0],old[1]+VIEWPORT[1],old[2]+VIEWPORT[0],old[3]+VIEWPORT[1]]
  results.append({'sequence':seq,'frame':i,'present':bool(truth.any()),'image_sha256':e['image_sha256'],'label_sha256':rows[i]['label_sha256'],'original_box':old,'candidate_box':e['candidate_box_full'],'oracle_box':gt_box(truth),'original':geometry(old,truth),'candidate':geometry(e['candidate_box_full'],truth)})
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'scope':'Retrospective scoring of one frozen detector call at the original automatic initialization time; no later search.','prediction_sha256':digest(a.predictions/'completion.json'),'verification_sha256':digest(a.verification/'verification.json'),'runner_sha256':digest(Path(__file__)),'results':results}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');lines=['# One-shot prompt detector: fixed initialization frames','','The original automatic initialization time is held fixed. A missing candidate detection remains missing; no later search is performed. Box IoU uses the tight label box and coverage measures the proportion of labeled foreground inside the detector box. Empty labels may be incomplete. The original detector training split and patient independence are unknown; the C6 validation center was used to select the new checkpoint. This is not a full-video result.','','| Sequence | Frame | Label | Original detection | New detection | Original / new box IoU | Original / new label coverage | Labeled-empty false alarm: original / new |','| --- | ---: | --- | :---: | :---: | ---: | ---: | :---: |']
 for r in results:
  old=r['original'];new=r['candidate'];fmt=lambda z:'—' if z is None else f'{z:.4f}'
  lines.append(f"| {r['sequence']} | {r['frame']} | {'present' if r['present'] else 'empty'} | {'yes' if old['detected'] else 'no'} | {'yes' if new['detected'] else 'no'} | {fmt(old['box_iou'])} / {fmt(new['box_iou'])} | {fmt(old['box_truth_coverage'])} / {fmt(new['box_truth_coverage'])} | {'—' if r['present'] else str(old['labeled_empty_false_alarm'])+' / '+str(new['labeled_empty_false_alarm'])} |")
 lines+=['','An absent candidate box cannot initialize the model under this fixed one-shot schedule. The two labeled-empty cases do not establish whether the polyp is truly absent, and four selected frames cannot establish generalization.','']
 (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
