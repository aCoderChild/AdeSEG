#!/usr/bin/env python3
"""Independently check saved point coordinates against the automatic mask."""
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
    p.add_argument('--run',type=Path,required=True)
    p.add_argument('--merged',type=Path,required=True)
    a=p.parse_args()
    done=json.loads((a.run/'completion.json').read_text()); audit=json.loads((a.run/'independent_audit.json').read_text())
    if not done['passed'] or not audit['passed'] or digest(a.run/'predictions.json') != done['predictions_sha256'] or digest(a.run/'results.json') != done['results_sha256']:
        raise ValueError('Pilot incomplete')
    loc=json.loads((a.merged/'locations.json').read_text()); rows=json.loads((a.run/'predictions.json').read_text())
    checked=0
    for row in rows:
        name=row['sequence']; folder=Path(loc[name])/name
        if digest(folder/'prompt.json') != row['old_prompt_sha256']: raise ValueError('Prompt changed')
        prompt=json.loads((folder/'prompt.json').read_text()); maskfile=folder/'prompt_provisional.png'
        if digest(maskfile) != prompt['provisional_mask_sha256']: raise ValueError('Mask changed')
        if row['points'] is None: continue
        mask=np.asarray(Image.open(maskfile))>0; h,w=mask.shape
        pos,neg=row['points']; x0,y0,x1,y1=prompt['box']
        for x,y in (pos,neg):
            if not (0 <= x < w and 0 <= y < h and x0 <= x < x1 and y0 <= y < y1): raise ValueError('Point outside image or box')
        if not mask[int(pos[1]),int(pos[0])] or mask[int(neg[1]),int(neg[0])]:
            raise ValueError('Point outside intended provisional region')
        checked+=1
    receipt={'passed':True,'checked':checked,'total':len(rows),'predictions_sha256':done['predictions_sha256'],'audit_source_sha256':digest(Path(__file__))}
    path=a.run/'geometry_audit.json'
    if path.exists(): raise FileExistsError(path)
    path.write_text(json.dumps(receipt,indent=2)+'\n'); print(json.dumps(receipt,indent=2))


if __name__=='__main__': main()
