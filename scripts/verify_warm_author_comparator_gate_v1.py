#!/usr/bin/env python3
"""Audit two MPS warm-prompt smokes per method without opening labels."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_v2 import EXPERIMENT_ROOT,new_output
from scripts.run_warm_author_comparator_v1 import DATA,METHODS,frozen_preflight,identity,input_data
from scripts.source_group_label_guard_v1 import install_mask_open_guard

REPRESENTATIVES=('seq6','seq23')


def require(value,message):
    if not value:raise ValueError(message)


def audit(method,smokes,plan):
    require(method in METHODS and set(smokes)==set(REPRESENTATIVES),'Smoke scope differs')
    state=None;common=None;rows=[];processes=set()
    for seq in REPRESENTATIVES:
        root=Path(smokes[seq]).resolve(strict=True);require(root.is_relative_to(EXPERIMENT_ROOT.resolve(strict=True)),'Smoke outside experiment root')
        path=root/'receipt.json';sha=digest(path);rec=json.loads(path.read_text())
        require(digest(path)==sha,'Smoke receipt changed during read')
        frames,start,prompt,_=input_data(seq,plan);expected=identity(method,seq,plan);records=rec.get('records',[])
        require(rec.get('passed') is True and rec.get('full_video') is False and rec.get('method')==method and
                rec.get('sequence')==seq and rec.get('identity')==expected and rec.get('prompt')==prompt and
                rec.get('prompt_frame')==start and rec.get('noncausal')==(method=='sam2long') and
                rec.get('feature_callback_enabled') is True and
                rec.get('label_guard',{}).get('open_attempts')==0 and len(records)==min(len(frames),start+5) and
                isinstance(rec.get('process_id'),int) and rec['process_id'] not in processes and rec.get('state_sha256') and
                isinstance(rec.get('prediction_wall_seconds'),(int,float)) and math.isfinite(rec['prediction_wall_seconds']) and
                rec['prediction_wall_seconds']>0,
                'Warm smoke identity or coverage differs')
        if method=='lean':require(bool(rec.get('positions')),'Lean smoke did not exercise indexed memory attention')
        if method=='sam2long':
            pathway=rec.get('pathway_audit') or {}
            require(pathway.get('audited_post_prompt_frames')==len(records)-start-1 and
                    pathway.get('pathways_after_initial_step')==3,'SAM2Long pathway smoke differs')
        processes.add(rec['process_id'])
        feature=rec.get('warm_feature_callback') or {}
        require(all(isinstance(feature.get(k),str) and len(feature[k])==64 for k in
                    ('early_sha256','positive_sha256','negative_sha256')) and
                isinstance(feature.get('cache_before'),list) and isinstance(feature.get('cache_after'),list) and
                isinstance(feature.get('early_shape'),list) and len(feature['early_shape'])==3,
                'Warm feature-access callback evidence missing')
        for i,(frame,row) in enumerate(zip(frames,records,strict=False)):
            path=root/f'{i:06d}.png';sha_mask=row['mask_sha256']
            require(row.get('frame')==i and row.get('source_frame_index')==frame.frame_index and
                    row.get('image_sha256')==digest(frame.image_path) and row.get('post_prompt')==(i>=start) and
                    digest(path)==sha_mask,'Warm smoke input or output differs')
            if i>=start:require(isinstance(row.get('logits_sha256'),str) and len(row['logits_sha256'])==64,
                                'Warm smoke logits fingerprint missing')
            with Image.open(frame.image_path) as src,Image.open(path) as image:
                values=np.asarray(image.convert('L'))
                require(image.format=='PNG' and image.size==src.size and np.isin(values,(0,255)).all() and
                        (i>=start or not values.any()),'Warm smoke geometry or binary encoding differs')
            require(digest(path)==sha_mask,'Warm smoke changed during read')
        shared={k:v for k,v in expected.items() if k!='sequence'}
        if state is not None:require(state==rec['state_sha256'] and common==shared,'Warm smoke state/source differs')
        state=rec['state_sha256'];common=shared;rows.append({'sequence':seq,'receipt_path':str(root/'receipt.json'),'receipt_sha256':sha})
    return {'passed':True,'method':method,'state_sha256':state,'common_identity':common,'smokes':rows,
            'verifier_sha256':digest(Path(__file__)),
            'scope':'Two MPS warm-prompt smokes; no cross-device or internal-state equivalence claim.'}


def verify_gate(path,expected,plan):
    path=Path(path).resolve(strict=True);require(path.is_relative_to(EXPERIMENT_ROOT.resolve(strict=True)),'Gate outside experiments')
    sha=digest(path);gate=json.loads(path.read_text());require(digest(path)==sha,'Gate changed while reading')
    require(gate.get('label_guard',{}).get('open_attempts')==0,'Gate label guard differs')
    smokes={row['sequence']:Path(row['receipt_path']).parent for row in gate.get('smokes',[])}
    checked=audit(expected['method'],smokes,plan)
    require({k:v for k,v in gate.items() if k!='label_guard'}==checked and
            checked['common_identity']=={k:v for k,v in expected.items() if k!='sequence'},
            'Warm gate cannot be reconstructed')
    return checked


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=METHODS,required=True)
    for name in ('seq6','seq23','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=new_output(a.output);guard=install_mask_open_guard(DATA);plan=frozen_preflight()
    result=audit(a.method,{'seq6':a.seq6,'seq23':a.seq23},plan)
    require(guard['open_attempts']==0,'Annotation open attempted');result['label_guard']=dict(guard)
    out.mkdir(parents=True);(out/'gate.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'method':a.method,'smokes':2}))


if __name__=='__main__':main()
