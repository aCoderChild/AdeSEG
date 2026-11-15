#!/usr/bin/env python3
"""Audit CPU and duplicate MPS native-SAM2 automatic-prompt short replays.

Only source images and frozen predictions are read; no annotations are opened.
The repeatability limit is an engineering tolerance, not an inference threshold.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.run_native_sam2_automatic_priority import identity


def read(path):return json.loads(Path(path).read_text())


def audit(cpu, mps_a, mps_b, checkpoint):
    paths=[Path(cpu),Path(mps_a),Path(mps_b)]
    runs=[read(path/'receipt.json') for path in paths]
    expected=[identity(checkpoint,'cpu'),identity(checkpoint,'mps'),identity(checkpoint,'mps')]
    for path,run,want in zip(paths,runs,expected,strict=True):
        if (not run.get('passed') or run.get('full_video') is not False or run.get('identity')!=want
                or not run.get('deterministic') or run.get('post_prompt_frames',0)<10
                or run.get('blocking_pointer_transfer') != (run.get('device')=='mps')):
            raise ValueError('Replay identity, coverage or runtime differs')
        records=run['records']
        if len(records)!=run['frames'] or [r['frame'] for r in records]!=list(range(len(records))):
            raise ValueError('Replay frame order differs')
        if any(digest(path/f"{r['frame']:06d}.png")!=r['mask_sha256'] for r in records):
            raise ValueError('Frozen replay mask changed')
    shared=('sequence','prompt_frame','prompt','rebased_start','state_sha256','post_prompt_frames','frames')
    if any(any(run.get(k)!=runs[0].get(k) for k in shared) for run in runs[1:]):
        raise ValueError('Replay prompt, frames or loaded model state differs')
    for run in runs[1:]:
        for a,b in zip(runs[0]['records'],run['records'],strict=True):
            if any(a[k]!=b[k] for k in ('frame','source_frame_index','image_sha256','post_prompt')):
                raise ValueError('Replay input sequence differs')
    rows=[]
    for row in runs[0]['records']:
        frame=row['frame']
        masks=[np.asarray(Image.open(path/f'{frame:06d}.png').convert('L'))>0 for path in paths]
        if len({x.shape for x in masks})!=1:raise ValueError('Replay mask shape differs')
        repeat=int((masks[1]!=masks[2]).sum());cross=int((masks[0]!=masks[1]).sum())
        if repeat > max(1, int(masks[1].size*0.0001)):
            raise ValueError('MPS replay variability exceeds predeclared 0.01% image-area tolerance')
        union=int((masks[0]|masks[1]).sum());intersection=int((masks[0]&masks[1]).sum())
        rows.append({'frame':frame,'mps_repeat_pixels':repeat,'cpu_mps_pixels':cross,
                     'cpu_mps_iou':intersection/union if union else 1.0})
    return {'passed':True,'scope':'Short label-free native automatic-prompt engineering gate; no efficacy or long-video guarantee.',
            'sequence':runs[0]['sequence'],'prompt':runs[0]['prompt'],'prompt_frame':runs[0]['prompt_frame'],
            'state_sha256':runs[0]['state_sha256'],'mps_identity':runs[1]['identity'],
            'repeat_pixel_fraction_limit':0.0001,'rows':rows,
            'receipt_sha256':[digest(path/'receipt.json') for path in paths],
            'verifier_sha256':digest(Path(__file__))}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('cpu','mps-a','mps-b','checkpoint','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    a=parser.parse_args();result=audit(a.cpu,a.mps_a,a.mps_b,a.checkpoint)
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'sequence':result['sequence'],'frames':len(result['rows']),
                      'cpu_mps_pixels':sum(r['cpu_mps_pixels'] for r in result['rows']),
                      'mps_repeat_pixels':sum(r['mps_repeat_pixels'] for r in result['rows'])}))


if __name__=='__main__':main()
