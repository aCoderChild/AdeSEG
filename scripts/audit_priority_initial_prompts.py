#!/usr/bin/env python3
"""Post-hoc prompt geometry audit of four archived automatic initializations."""
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


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for k in ('merged','crop-reference','crop-gate-a','crop-gate-b','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();merged=read(a.merged/'completion.json');locations=read(a.merged/'locations.json')
    if not merged['passed'] or merged['results_sha256']!=digest(a.merged/'results.json'):raise ValueError('Merged reference changed')
    crop=read(a.crop_reference/'completion.json');ga=read(a.crop_gate_a/'gate.json');gb=read(a.crop_gate_b/'gate.json')
    if not crop['passed'] or crop['prompt_sha256']!=digest(a.crop_reference/'prompt.json') or crop['frames_sha256']!=digest(a.crop_reference/'frames.json') or crop['setup_sha256']!=digest(a.crop_reference/'setup.json') or ga!=gb or ga['setup']!=read(a.crop_reference/'setup.json') or digest(a.crop_reference/'prompt.png')!=ga['files']['prompt.png'] or digest(a.crop_gate_a/'prompt.png')!=ga['files']['prompt.png'] or digest(a.crop_gate_b/'prompt.png')!=gb['files']['prompt.png']:
        raise ValueError('Inset-excluded prompt reference changed')
    videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};results=[]
    for seq in ('seq8','seq11','seq20','seq23'):
        if seq=='seq23':
            folder=a.crop_reference;prompt=read(folder/'prompt.json');i=prompt['first_detection_index'];box=prompt['box_viewport_xyxy'];x0,y0,_,_=VIEWPORT
            box=[box[0]+x0,box[1]+y0,box[2]+x0,box[3]+y0];mask_path=folder/'prompt.png';rows=read(folder/'frames.json');policy='fixed inset-excluded viewport'
        else:
            source=Path(locations[seq]);done=read(source/'completion.json');record=read(source/'records.json')[0];folder=source/seq
            if not done['passed'] or digest(source/'records.json')!=done['records_sha256'] or digest(folder/'frames.json')!=record['frames_sha256'] or digest(folder/'prompt.json')!=record['prompt_sha256']:
                raise ValueError('Automatic run metadata changed: '+seq)
            prompt=read(folder/'prompt.json');i=prompt['first_detection_index'];box=prompt['box'];mask_path=folder/'prompt_refined.png';rows=read(folder/'frames.json');policy='original full frame'
            if digest(mask_path)!=prompt['refined_mask_sha256']:raise ValueError('Automatic prompt mask changed')
        if i is None or i<0 or i>=len(videos[seq]):raise ValueError('Missing prompt index')
        frame=videos[seq][i]
        if digest(frame.image_path)!=rows[i]['image_sha256'] or digest(frame.mask_path)!=rows[i]['label_sha256']:raise ValueError('Input or label identity changed')
        pred=np.asarray(Image.open(mask_path))>0;gt=load_label_mask(frame.mask_path,[1])==1
        values=evaluate_prompt_geometry(box,pred,gt)
        results.append({'sequence':seq,'frame':i,'policy':policy,'box_xyxy':box,'prompt_mask_sha256':digest(mask_path),
          'image_sha256':rows[i]['image_sha256'],'label_sha256':rows[i]['label_sha256'],'geometry':values})
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    result={'scope':'Retrospective post-hoc audit of saved initial automatic predictions; labels were not used to generate prompts.',
      'results':results,'merged_sha256':digest(a.merged/'results.json'),'crop_reference_sha256':digest(a.crop_reference/'completion.json'),
      'geometry_sha256':digest(ROOT/'evaluation/prompt_geometry_audit.py'),'runner_sha256':digest(Path(__file__))}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# Automatic initial prompt quality on four selected sequences','','Labels are used only for this post-hoc audit. Box coverage uses pixel centers; empty-frame prompt Dice is undefined. Sequence 23 uses the fixed, retrospectively chosen inset-excluded viewport.','','| Sequence | Initial frame | Label present | Box fraction of full image | Mask fraction | Mask Dice | Labeled polyp covered by box | Empty-frame prompt false alarm |','| --- | ---: | :---: | ---: | ---: | ---: | ---: | :---: |']
    for r in results:
        v=r['geometry'];dice='—' if v['prompt_mask_dice'] is None else f"{v['prompt_mask_dice']:.6f}";coverage='—' if v['box_truth_coverage'] is None else f"{v['box_truth_coverage']:.6f}";fa='—' if v['empty_prompt_false_alarm'] is None else str(v['empty_prompt_false_alarm'])
        lines.append(f"| {r['sequence']} | {r['frame']} | {'yes' if v['present'] else 'no'} | {v['box_fraction']:.4f} | {v['prompt_mask_fraction']:.4f} | {dice} | {coverage} | {fa} |")
    lines+=['','A labeled-empty frame is not proof that the video has no unlabeled polyp. These measurements do not evaluate future-frame detector recall or establish causality for later propagation.','']
    (out/'report.md').write_text('\n'.join(lines));(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'report_sha256':digest(out/'report.md')},indent=2)+'\n');print(str(out/'report.md'))
if __name__=='__main__':main()
