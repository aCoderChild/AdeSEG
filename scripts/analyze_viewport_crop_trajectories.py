#!/usr/bin/env python3
"""Post-hoc framewise diagnostic of audited fixed-crop video results."""
import argparse
import json
from pathlib import Path
import sys
import numpy as np
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest  # noqa: E402


def read(path): return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--audit',type=Path,required=True)
    p.add_argument('--merged',type=Path,required=True)
    p.add_argument('--part',type=Path,action='append',required=True)
    a=p.parse_args()
    done=read(a.audit/'completion.json')
    if not done['passed'] or digest(a.audit/'results.json')!=done['results_sha256']: raise ValueError('Candidate audit changed')
    selected=read(a.audit/'results.json')['sequences']; locations=read(a.merged/'locations.json')
    output={}
    for part in a.part:
        for name in read(part/'completion.json')['sequences']:
            old=Path(locations[name])/name; new=part/name
            orig=read(old/'frames.json'); candidate=read(new/'frames.json'); record=read(new/'records.json')
            if digest(new/'frames.json') != record['frames_sha256'] or len(orig)!=len(candidate): raise ValueError('Frame records changed')
            changes=[]
            for i,(x,y) in enumerate(zip(orig,candidate)):
                if i!=x['index'] or i!=y['frame'] or x['present']!=y['present']: raise ValueError('Frame alignment differs')
                if x['present']:
                    changes.append({'frame':i,'original_dice':x['dice'],'candidate_dice':y['dice'],'difference':y['dice']-x['dice']})
            worst=min(changes,key=lambda z:z['difference']);best=max(changes,key=lambda z:z['difference'])
            d=np.array([x['difference'] for x in changes]); first=record['first_detection_index']
            output[name]={'first_detection_index':first,'present_frames':len(changes),
                'candidate_minus_original_mean':float(d.mean()),'candidate_worse_frames':int((d < 0).sum()),
                'candidate_better_frames':int((d > 0).sum()),'candidate_equal_frames':int((d == 0).sum()),
                'worst_frame':worst,'best_frame':best,
                'original_zero_dice_frames':sum(x['original_dice']==0 for x in changes),
                'candidate_zero_dice_frames':sum(x['candidate_dice']==0 for x in changes)}
    if set(output)!=set(selected): raise ValueError('Coverage differs')
    file=a.audit/'trajectory_diagnostics.json'
    if file.exists(): raise FileExistsError(file)
    file.write_text(json.dumps({'scope':'Post-hoc all-frame descriptive differences; correlated frames, not causal proof of a particular submodule.',
                               'results_sha256':done['results_sha256'],'sequences':output},indent=2)+'\n')
    print(json.dumps(output,indent=2))


if __name__=='__main__': main()
