#!/usr/bin/env python3
"""Label-free fixed-read audit of a smooth bounded logit anchor.

Compare native logits with s*tanh(logits/s), after native-video resizing.
No label is read, no model is trained, no object score or pointer is changed.
The modified probabilities may alter memory-encoder input and trajectories.
"""
import argparse,json,sys
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture,to_video
from scripts.raw_multimask_shared_v1 import load_manifests

SCALE=16.0

def compare(logits,shape):
    native=to_video(logits.float(),shape);bounded=to_video(SCALE*torch.tanh(logits.float()/SCALE),shape)
    if not torch.isfinite(native).all() or not torch.isfinite(bounded).all():raise ValueError('Nonfinite logits')
    a=native>0;b=bounded>0;diff=(native.sigmoid()-bounded.sigmoid()).abs()
    return {'shape':list(shape),'native_positive_pixels':int(a.sum()),'bounded_positive_pixels':int(b.sum()),
            'mask_changed_pixels':int((a!=b).sum()),'native_only_pixels':int((a&~b).sum()),'bounded_only_pixels':int((b&~a).sum()),
            'mean_abs_probability_change':float(diff.mean()),'max_abs_probability_change':float(diff.max()),
            'mean_native_probability':float(native.sigmoid().mean()),'mean_bounded_probability':float(bounded.sigmoid().mean()),
            'native_logit_min':float(native.min()),'native_logit_max':float(native.max()),
            'bounded_logit_min':float(bounded.min()),'bounded_logit_max':float(bounded.max())}

def audit(data_preflight,raw_audit,output):
    m,mhash,_,rhash=load_manifests(data_preflight,raw_audit);rows=[]
    for item in m['samples']:
        _,_,logits,shape=read_capture(item)
        row={'sequence':item['sequence'],'frame':item['frame'],'split':item['split'],'capture_device':item['capture_device'],'snapshot_sha256':item['snapshot_sha256']}
        row.update(compare(logits,shape));rows.append(row)
    if len(rows)!=49 or sum(x['split']=='train' for x in rows)!=39:raise ValueError('Frozen read coverage differs')
    out=validate_output(output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'scope':'Label-free fixed-read CPU audit; no inference, no validation labels, no efficacy evidence.',
            'transform':'16 * tanh(native low-resolution logits / 16), then resize bilinearly to original video size',
            'scale':SCALE,'source_sha256':digest(Path(__file__)),'manifest_sha256':mhash,'raw_manifest_sha256':rhash,
            'warning':'Even with the same low-resolution sign, nonlinear compression before interpolation can alter video-size masks; probability changes may alter memory encoding and later trajectories.',
            'summary':{'reads':len(rows),'reads_with_mask_changes':sum(x['mask_changed_pixels']>0 for x in rows),'total_mask_changed_pixels':sum(x['mask_changed_pixels'] for x in rows),
                       'maximum_mask_changed_pixels':max(x['mask_changed_pixels'] for x in rows),'max_abs_probability_change':max(x['max_abs_probability_change'] for x in rows),
                       'mean_per_read_abs_probability_change':sum(x['mean_abs_probability_change'] for x in rows)/len(rows)},'rows':rows}
    path=out/'audit.json';path.write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'audit_sha256':digest(path),'source_sha256':digest(Path(__file__))},indent=2)+'\n')
    print(json.dumps(result['summary'],indent=2))

if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('data-preflight','raw-audit','output'):p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args();torch.set_num_threads(2);audit(a.data_preflight,a.raw_audit,a.output)
