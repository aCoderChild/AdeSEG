#!/usr/bin/env python3
"""Verify one fresh MPS smoke per author method, without annotation access."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_v2 import DATA, REPRESENTATIVE, identity, input_data, new_output
from scripts.source_group_label_guard_v1 import install_mask_open_guard


def audit(smoke, checkpoint, detector, extended_preflight, remaining_preflight):
    path=Path(smoke).resolve(strict=True);receipt=path/'receipt.json';before=digest(receipt)
    run=json.loads(receipt.read_text())
    if digest(receipt)!=before:raise ValueError('Smoke receipt changed during parse')
    method=run['method'];seq=run['sequence']
    if method not in REPRESENTATIVE or seq!=REPRESENTATIVE[method]:raise ValueError('Smoke representative differs')
    want=identity(method,seq,checkpoint,detector,extended_preflight,remaining_preflight,'mps')
    frames,start,_,prompt=input_data(seq,extended_preflight,remaining_preflight,checkpoint,detector)
    records=run.get('records',[])
    if (run.get('passed') is not True or run.get('full_video') is not False or not run.get('deterministic')
            or run.get('identity')!=want or run.get('label_guard',{}).get('open_attempts')!=0
            or run.get('prompt_frame')!=start or run.get('prompt')!=prompt or run.get('device')!='mps'
            or run.get('noncausal')!=(method=='sam2long') or not run.get('state_sha256')
            or len(records)!=min(len(frames),start+5) or [r['frame'] for r in records]!=list(range(len(records)))):
        raise ValueError('Smoke identity, guard or coverage differs')
    for i,row in enumerate(records):
        file=path/f'{i:06d}.png';frame=frames[i]
        if (row.get('source_frame_index')!=frame.frame_index or row.get('image_sha256')!=digest(frame.image_path)
                or row.get('post_prompt')!=(i>=start) or row.get('mask_sha256')!=digest(file)):
            raise ValueError('Smoke input or mask identity differs')
        with Image.open(frame.image_path) as source, Image.open(file) as image:
            mask=np.asarray(image.convert('L'))
            if image.format!='PNG' or image.size!=source.size or not np.isin(mask,(0,255)).all() or (i<start and mask.any()):
                raise ValueError('Smoke mask geometry or encoding differs')
        if row.get('mask_sha256')!=digest(file):raise ValueError('Smoke mask changed during read')
    return {'passed':True,'scope':'One MPS smoke; no CPU comparison or repeated-run equivalence claim.',
        'identity':run['identity'],'prompt':prompt,'state_sha256':run['state_sha256'],
        'receipt_paths':[{'path':str(receipt),'sha256':before}],
        'frames_checked':len(records),'verifier_sha256':digest(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('smoke','checkpoint','detector','extended-preflight','remaining-preflight','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=new_output(a.output);guard=install_mask_open_guard(DATA)
    result=audit(a.smoke,a.checkpoint,a.detector,a.extended_preflight,a.remaining_preflight)
    if guard['open_attempts']:raise ValueError('Annotation-mask open attempted')
    result['label_guard']=dict(guard);out.mkdir(parents=True,exist_ok=False)
    (out/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'method':result['identity']['method'],
                      'sequence':result['identity']['sequence'],'frames':result['frames_checked']}))


if __name__=='__main__':main()
