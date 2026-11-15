#!/usr/bin/env python3
"""Label-free paired first-prompt engineering gate for the fixed crop study."""
import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest, stage_frames  # noqa: E402
from modeling.early_feature_prompt import early_feature_points  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402
from scripts.run_author_baseline import source_identity, build_author_predictor, parameter_fingerprints, assert_parameter_integrity  # noqa: E402


def read(p): return json.loads(p.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','merged','crop-diagnostic','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    merged=read(a.merged/'completion.json'); crop_done=read(a.crop_diagnostic/'completion.json')
    if not merged['passed'] or digest(a.merged/'results.json') != merged['results_sha256'] or not crop_done['passed'] or digest(a.crop_diagnostic/'results.json') != crop_done['results_sha256']:
        raise ValueError('Reference or crop diagnostic invalid')
    crop_data=read(a.crop_diagnostic/'results.json'); location=Path(read(a.merged/'locations.json')['seq23'])
    setup=read(location/'setup.json'); done=read(location/'completion.json'); audit=read(location/'independent_audit.json')
    if (not done['passed'] or not audit['passed'] or digest(location/'records.json') != done['records_sha256']
        or audit['records_sha256'] != done['records_sha256'] or digest(a.checkpoint) != setup['signature']['checkpoint_sha256']
        or crop_data['checkpoint_sha256'] != setup['signature']['detector_sha256']):
        raise ValueError('Model or source run differs')
    folder=location/'seq23'; record=read(location/'records.json')[0]
    if digest(folder/'prompt.json') != record['prompt_sha256'] or digest(folder/'frames.json') != record['frames_sha256']:
        raise ValueError('Source metadata changed')
    old=read(folder/'prompt.json'); frames=read(folder/'frames.json'); index=old['first_detection_index']
    choice=crop_data['cases']['seq23']; crop=choice['predictions']['right_two_thirds']['box_full_image']
    if crop is None or choice['index'] != index or choice['image_sha256'] != frames[index]['image_sha256']:
        raise ValueError('Crop candidate does not match source image')
    videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
    source=videos['seq23']
    if digest(source[index].image_path) != frames[index]['image_sha256']:
        raise ValueError('Image changed')
    out=validate_output(a.output); out.mkdir(parents=True,exist_ok=False); stage=out/'input'; stage.mkdir(); stage_frames(source,stage)
    torch.set_num_threads(2); repo,revision=source_identity('lean')
    model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),repo)
    before=parameter_fingerprints(model)
    state=model.init_state(video_path=str(stage),offload_video_to_cpu=True,offload_state_to_cpu=False)
    arms={}
    with torch.inference_mode():
        for arm,box in [('original',old['box']),('fixed_crop',crop)]:
            _, ids, logits=model.add_new_points_or_box(state,frame_idx=index,obj_id=1,box=np.asarray(box,np.float32))
            if ids != [1]: raise ValueError('Object ID differs')
            provisional=logits[0,0].detach().cpu().numpy()>0
            if arm=='original' and not np.array_equal(provisional,np.asarray(Image.open(folder/'prompt_provisional.png'))>0):
                raise ValueError('Original box mask differs')
            _,backbone,_,_,_=model._get_image_feature(state,index,batch_size=1)
            points,status=early_feature_points(backbone,provisional,np.asarray(box,np.float32),seed=20261114)
            if arm=='original' and (status != old['point_status'] or (None if points is None else points['feature_points']) != old['points']):
                raise ValueError('Original points differ')
            if points is not None:
                _,ids,logits=model.add_new_points_or_box(state,frame_idx=index,obj_id=1,
                    points=np.asarray(points['feature_points'],np.float32),labels=np.array([1,0],np.int32),clear_old_points=False)
            mask=logits[0,0].detach().cpu().numpy()>0
            if arm=='original' and not np.array_equal(mask,np.asarray(Image.open(folder/'prompt_refined.png'))>0):
                raise ValueError('Original refined mask differs')
            Image.fromarray(mask.astype(np.uint8)*255).save(out/f'{arm}.png')
            arms[arm]={'box':box,'points':None if points is None else points['feature_points'],
                       'point_status':status,'mask_sha256':digest(out/f'{arm}.png')}
            model.reset_state(state)
    if arms['fixed_crop']['mask_sha256'] == arms['original']['mask_sha256']:
        raise ValueError('Candidate did not exercise changed prompt')
    assert_parameter_integrity(model,before,out,'after')
    result={'scope':'Label-free fixed crop seq23 prompt engineering gate; no video outcome.',
            'sequence':'seq23','frame':index,'lean_revision':revision,'checkpoint_sha256':digest(a.checkpoint),
            'crop_results_sha256':crop_done['results_sha256'],'reference_records_sha256':done['records_sha256'],
            'source_sha256':digest(Path(__file__)),'arms':arms}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json')},indent=2)+'\n')
    print(json.dumps(result,indent=2))


if __name__=='__main__': main()
