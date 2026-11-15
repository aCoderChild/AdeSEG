#!/usr/bin/env python3
"""Independently recompute archived prompt geometry from source images/masks."""
import argparse,json,sys
from pathlib import Path
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser()
    for k in ('audit','merged','crop-reference','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();done=read(a.audit/'completion.json');d=read(a.audit/'results.json')
    if not done['passed'] or done['results_sha256']!=digest(a.audit/'results.json') or done['report_sha256']!=digest(a.audit/'report.md') or d['runner_sha256']!=digest(ROOT/'scripts/audit_priority_initial_prompts.py') or d['geometry_sha256']!=digest(ROOT/'evaluation/prompt_geometry_audit.py'):
        raise ValueError('Prompt audit changed')
    videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};locations=read(a.merged/'locations.json')
    if {x['sequence'] for x in d['results']}!={'seq8','seq11','seq20','seq23'}:raise ValueError('Sequence set changed')
    for r in d['results']:
        seq=r['sequence'];i=r['frame'];f=videos[seq][i]
        mask_path=a.crop_reference/'prompt.png' if seq=='seq23' else Path(locations[seq])/seq/'prompt_refined.png'
        if digest(mask_path)!=r['prompt_mask_sha256'] or digest(f.image_path)!=r['image_sha256'] or digest(f.mask_path)!=r['label_sha256']:
            raise ValueError('Source identity changed')
        pred=torch.from_numpy(np.asarray(Image.open(mask_path))>0);gt=torch.from_numpy(load_label_mask(f.mask_path,[1])==1);h,w=gt.shape
        y,x=torch.meshgrid(torch.arange(h,dtype=torch.float64)+.5,torch.arange(w,dtype=torch.float64)+.5,indexing='ij');x0,y0,x1,y1=r['box_xyxy']
        b=(x>=x0)&(x<x1)&(y>=y0)&(y<y1);n=int(gt.sum());q=int(pred.sum());area=int(b.sum());both=int((pred & gt).sum());covered=int((b & gt).sum());v=r['geometry']
        if v['truth_pixels']!=n or v['prompt_mask_pixels']!=q or v['box_pixels']!=area or v['present']!=(n>0) or abs(v['box_fraction']-area/(h*w))>1e-12 or abs(v['prompt_mask_fraction']-q/(h*w))>1e-12:
            raise ValueError('Prompt arithmetic mismatch')
        if n:
            if abs(v['prompt_mask_dice']-2*both/(n+q))>1e-12 or abs(v['box_truth_coverage']-covered/n)>1e-12 or abs(v['box_precision_pixels']-covered/area)>1e-12:raise ValueError('Prompt overlap mismatch')
        elif v['empty_prompt_false_alarm']!=bool(q) or v['prompt_mask_dice'] is not None:raise ValueError('Empty handling mismatch')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'passed':True,'sequences':4,'audit_results_sha256':digest(a.audit/'results.json'),'verifier_sha256':digest(Path(__file__)),'scope':'Post-hoc geometry only; GT not used in prediction.'}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
