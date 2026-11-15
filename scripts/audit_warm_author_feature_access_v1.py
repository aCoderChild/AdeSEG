#!/usr/bin/env python3
"""Report same-MPS warm-smoke effects of one extra feature-access callback."""
import argparse
import hashlib
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_v2 import EXPERIMENT_ROOT,new_output
from scripts.run_warm_author_comparator_v1 import DATA,METHODS,frozen_preflight,identity,input_data
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_warm_author_comparator_gate_v1 import verify_gate


def require(value,message):
    if not value:raise ValueError(message)


def audit(method,gate_path,control,plan):
    control=Path(control).resolve(strict=True);gate_path=Path(gate_path).resolve(strict=True)
    require(control.is_relative_to(EXPERIMENT_ROOT.resolve(strict=True)),'Control outside experiment root')
    checked=verify_gate(gate_path,identity(method,'seq6',plan),plan)
    entry=next(x for x in checked['smokes'] if x['sequence']=='seq6')
    enabled_path=Path(entry['receipt_path']);enabled_sha=entry['receipt_sha256']
    require(digest(enabled_path)==enabled_sha,'Enabled smoke receipt differs from reconstructed gate')
    raw=enabled_path.read_bytes();enabled=json.loads(raw)
    require(hashlib.sha256(raw).hexdigest()==enabled_sha and digest(enabled_path)==enabled_sha,
            'Enabled smoke receipt changed during read')
    path=control/'receipt.json';sha=digest(path);disabled=json.loads(path.read_text())
    require(digest(path)==sha and digest(enabled_path)==enabled_sha,'Smoke receipt changed during read')
    frames,start,prompt,_=input_data('seq6',plan);records=disabled.get('records',[])
    require(disabled.get('passed') is True and disabled.get('full_video') is False and
            disabled.get('feature_callback_enabled') is False and not disabled.get('warm_feature_callback') and
            disabled.get('identity')==identity(method,'seq6',plan) and disabled.get('method')==method and
            disabled.get('sequence')=='seq6' and disabled.get('prompt')==prompt and disabled.get('prompt_frame')==start and
            disabled.get('state_sha256')==checked['state_sha256'] and
            disabled.get('noncausal')==(method=='sam2long') and disabled.get('label_guard',{}).get('open_attempts')==0 and
            isinstance(disabled.get('process_id'),int) and disabled['process_id']!=enabled.get('process_id') and
            len(records)==len(enabled['records'])==min(len(frames),start+5),'Feature-access control identity differs')
    differences=[]
    for i,(frame,left,right) in enumerate(zip(frames,enabled['records'],records,strict=False)):
        path_left=enabled_path.parent/f'{i:06d}.png';path_right=control/f'{i:06d}.png'
        require(right.get('frame')==left.get('frame')==i and right.get('source_frame_index')==frame.frame_index and
                right.get('image_sha256')==left.get('image_sha256')==digest(frame.image_path) and
                right.get('post_prompt')==left.get('post_prompt')==(i>=start) and
                digest(path_left)==left['mask_sha256'] and digest(path_right)==right['mask_sha256'],
                'Feature-access control output differs from its receipt')
        with Image.open(path_left) as x,Image.open(path_right) as y:
            a=np.asarray(x.convert('L'));b=np.asarray(y.convert('L'))
            require(x.format==y.format=='PNG' and a.shape==b.shape and np.isin(a,(0,255)).all() and
                    np.isin(b,(0,255)).all(),'Feature-access control mask geometry differs')
        require(digest(path_left)==left['mask_sha256'] and digest(path_right)==right['mask_sha256'],
                'Feature-access mask changed during reading')
        if i>=start:
            require(all(isinstance(r.get('logits_sha256'),str) and len(r['logits_sha256'])==64 for r in (left,right)),
                    'Feature-access logits fingerprint missing')
            differences.append({'frame':i,'mask_disagree_pixels':int(np.count_nonzero(a!=b)),
                                'logits_hash_equal':left['logits_sha256']==right['logits_sha256']})
    require(digest(path)==sha and digest(enabled_path)==enabled_sha,'Smoke receipt changed during audit')
    return {'passed':True,'method':method,'sequence':'seq6','gate_sha256':digest(gate_path),
            'enabled_receipt_sha256':enabled_sha,'control_receipt_sha256':sha,'differences':differences,
            'auditor_sha256':digest(Path(__file__)),
            'scope':'Same-device short output diagnostic; equality is not required and internal state is not exhaustively compared.'}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--method',choices=METHODS,required=True)
    for name in ('gate','control','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();out=new_output(a.output);guard=install_mask_open_guard(DATA);plan=frozen_preflight()
    result=audit(a.method,a.gate,a.control,plan);require(guard['open_attempts']==0,'Annotation open attempted')
    result['label_guard']=dict(guard);out.mkdir(parents=True)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'method':a.method,'compared_frames':len(result['differences'])}))


if __name__=='__main__':main()
