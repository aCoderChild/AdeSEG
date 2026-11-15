#!/usr/bin/env python3
"""Describe pixel changes between independently audited paired predictions."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest  # noqa: E402


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--merged',type=Path,required=True)
    p.add_argument('--part',type=Path,action='append',required=True)
    a=p.parse_args()
    complete=json.loads((a.audit/'completion.json').read_text())
    if not complete['passed'] or digest(a.audit/'results.json') != complete['results_sha256']: raise ValueError('Audit changed')
    names=json.loads((a.audit/'results.json').read_text())['sequences']
    locations=json.loads((a.merged/'locations.json').read_text())
    summary={}
    for part in a.part:
        for name in json.loads((part/'completion.json').read_text())['sequences']:
            old=Path(locations[name])/name; new=part/name
            old_rows=json.loads((old/'frames.json').read_text()); new_rows=json.loads((new/'frames.json').read_text())
            if len(old_rows) != len(new_rows): raise ValueError('Coverage differs')
            changed=0; fractions=[]; ious=[]
            for i,(x,y) in enumerate(zip(old_rows,new_rows)):
                aa=old/f'{i:06d}.png'; bb=new/f'{i:06d}.png'
                if digest(aa)!=x['mask_sha256'] or digest(bb)!=y['mask_sha256']: raise ValueError('Mask changed')
                maska=np.asarray(Image.open(aa))>0; maskb=np.asarray(Image.open(bb))>0
                if maska.shape != maskb.shape: raise ValueError('Shape mismatch')
                xor=np.count_nonzero(maska ^ maskb); union=np.count_nonzero(maska | maskb)
                changed+=bool(xor); fractions.append(xor/maska.size)
                ious.append(np.count_nonzero(maska & maskb)/union if union else 1.)
            summary[name]={'frames':len(old_rows),'changed_mask_frames':changed,
                           'mean_changed_image_fraction':float(np.mean(fractions)),
                           'mean_prediction_pair_iou':float(np.mean(ious))}
    if set(summary) != set(names): raise ValueError('Coverage differs')
    out=a.audit/'mask_pair_analysis.json'
    if out.exists(): raise FileExistsError(out)
    out.write_text(json.dumps({'scope':'Mask-to-mask similarity, not accuracy or causal importance. Mean image fraction weights frames equally.',
                              'results_sha256':complete['results_sha256'],'sequences':summary},indent=2)+'\n')
    print(json.dumps(summary,indent=2))


if __name__=='__main__': main()
