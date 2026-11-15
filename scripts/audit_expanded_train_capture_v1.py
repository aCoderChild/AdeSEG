#!/usr/bin/env python3
"""Verify label-free MPS capture did not alter the full reference trajectory."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
from scripts.train_multilevel_refiner_seq4_empty_v2 import to_video


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('preflight','reference','capture','output'):p.add_argument('--'+k,required=True,type=Path)
    a=p.parse_args();pre=read(a.preflight/'manifest.json');done=read(a.preflight/'completion.json')
    if not done['passed'] or done['manifest_sha256']!=digest(a.preflight/'manifest.json'):raise ValueError('Preflight changed')
    ref=read(a.reference/'receipt.json');cap=read(a.capture/'receipt.json')
    if (not ref['passed'] or not cap['passed'] or ref['mode']!='reference' or cap['mode']!='capture'
            or ref['sequence']!=cap['sequence'] or ref['identity']!=cap['identity']
            or ref['identity']['preflight_sha256']!=digest(a.preflight/'manifest.json')
            or ref['identity']['runner_sha256']!=digest(ROOT/'scripts/run_expanded_train_capture_v1.py')
            or ref['identity']['observer_sha256']!=digest(ROOT/'modeling/lean_mps_decoder_observer.py')):
        raise ValueError('Reference/capture runtime identities differ')
    spec=pre['sequences'][ref['sequence']]
    if ref['prompt_frame']!=spec['prompt_frame'] or ref['targets']!=spec['targets'] or cap['targets']!=spec['targets']:
        raise ValueError('Capture target plan differs')
    if len(ref['rows'])!=spec['frames']-spec['prompt_frame'] or len(cap['rows'])!=len(ref['rows']):
        raise ValueError('Trajectory coverage differs')
    if ref['rows']!=cap['rows'] or ref['positions']!=cap['positions'] or ref['selection']!=cap['selection'] or ref['anchor_events']!=cap['anchor_events'] or ref['token_window_events']!=cap['token_window_events']:
        raise ValueError('Observer changed masks or memory decisions')
    for row in ref['rows']:
        for folder in (a.reference,a.capture):
            if digest(folder/f"{row['frame']:06d}.png")!=row['mask_sha256']:raise ValueError('Saved trajectory mask changed')
    writes=[x for x in ref['token_window_events'] if x['stage']=='after_write']
    if len(writes)!=len(ref['rows'])-1 or any(x['total_tokens']>16384 or x['insurance_tokens']>3072 for x in writes):
        raise ValueError('Window budget or coverage differs')
    obs=cap['observations']
    if sorted(x['frame'] for x in obs)!=spec['targets'] or not torch.backends.mps.is_available():raise ValueError('Observed target coverage or MPS differs')
    for item in obs:
        feature=Path(item['feature_file']);snapshot=Path(item['snapshot_file'])
        if (not feature.resolve().is_relative_to(a.capture.resolve()) or not snapshot.resolve().is_relative_to(a.capture.resolve())
                or digest(feature)!=item['feature_sha256'] or digest(snapshot)!=item['snapshot_sha256']):
            raise ValueError('Observed sample changed')
        f=torch.load(feature,map_location='cpu',weights_only=True);s=torch.load(snapshot,map_location='cpu',weights_only=True)
        if (f['frame']!=item['frame'] or s['frame']!=item['frame'] or f['feature'].shape!=(32,256,256)
                or f['foreground'].shape!=(32,) or f['background'].shape!=(32,)
                or not torch.equal(f['feature'].unsqueeze(0),s['decoder_inputs']['high_res_features'][0])
                or s['decoder_outputs'][3].shape!=(1,1,256,256)):
            raise ValueError('Observed decoder sample geometry differs')
        with torch.inference_mode():
            video=to_video(s['decoder_outputs'][3].float().to('mps'),(s['video_height'],s['video_width']))
            recomputed=(video>0).cpu().numpy()[0,0]
        original=np.asarray(Image.open(a.capture/f"{item['frame']:06d}.png").convert('L'))>0
        if not np.array_equal(recomputed,original):raise ValueError('Observed decoder output does not reproduce video mask')
    result={'passed':True,'scope':'Full-trajectory label-free observer equivalence, not patient independence or complete attention capture.',
            'sequence':ref['sequence'],'prompt_frame':ref['prompt_frame'],'targets':spec['targets'],'frames':len(ref['rows']),
            'max_total_tokens':max(x['total_tokens'] for x in writes),'evictions':sum(bool(x['evict_frames']) for x in ref['token_window_events'] if x['stage']=='admission_plan'),
            'identity':ref['identity'],'reference_receipt_sha256':digest(a.reference/'receipt.json'),'capture_receipt_sha256':digest(a.capture/'receipt.json'),
            'auditor_sha256':digest(Path(__file__))}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({k:result[k] for k in ('passed','sequence','frames','targets','max_total_tokens','evictions')}),flush=True)


if __name__=='__main__':main()
