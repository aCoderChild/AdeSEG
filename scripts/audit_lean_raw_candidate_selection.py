#!/usr/bin/env python3
"""Retrospective training-only scoring of frozen author decoder candidates.

No live inference or memory state is modified. Decoder tensors were captured by
one replay with the snapshot's original `multimask_output` and no-object score;
the replay's post-suppression outputs and archived masks were exactly checked.
Predicted-IoU and early-feature scores are frozen before training labels are
opened. A label-selected candidate is an offline oracle only, never inference.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask,save_binary_mask
from scripts.memory_token_io import digest,validate_output
from scripts.train_multilevel_refiner_seq4_empty_v2 import to_video
from scripts.train_region_conditioned_hardcase_v1 import load_manifest


def feature_scores(masks, feature, foreground, background):
    """Soft-mask means of prompt-derived foreground and FG/BG cosine scores."""
    if masks.ndim!=4 or masks.shape[0]!=1 or feature.ndim!=3 or tuple(feature.shape[1:])!=tuple(masks.shape[-2:]) or foreground.shape!=(feature.shape[0],) or background.shape!=foreground.shape:
        raise ValueError('Feature, prototype or mask geometry differs')
    if not all(torch.isfinite(x).all() for x in (masks,feature,foreground,background)):
        raise ValueError('Nonfinite candidate scoring input')
    f=F.normalize(feature.float(),dim=0,eps=1e-6)
    fg=F.normalize(foreground.float(),dim=0,eps=1e-6)
    bg=F.normalize(background.float(),dim=0,eps=1e-6)
    fg_map=torch.einsum('chw,c->hw',f,fg)
    contrast_map=fg_map-torch.einsum('chw,c->hw',f,bg)
    probabilities=masks[0].float().sigmoid()
    denominator=probabilities.sum((-2,-1)).clamp_min(1e-6)
    foreground_scores=(probabilities*fg_map).sum((-2,-1))/denominator
    contrast_scores=(probabilities*contrast_map).sum((-2,-1))/denominator
    return foreground_scores,contrast_scores


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('data-preflight','raw-diagnostic','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();manifest,mhash=load_manifest(a.data_preflight)
    receipt=json.loads((a.raw_diagnostic/'completion.json').read_text())
    if not receipt['passed'] or receipt['manifest_sha256']!=mhash or receipt['runner_sha256']!=digest(ROOT/'scripts/diagnose_lean_raw_decoder_train_reads.py') or receipt['frozen_predictions_sha256']!=digest(a.raw_diagnostic/'frozen_predictions.json'):
        raise ValueError('Frozen raw diagnostic provenance changed')
    frozen=json.loads((a.raw_diagnostic/'frozen_predictions.json').read_text())
    train={(x['sequence'],x['frame']):x for x in manifest['samples'] if x['split']=='train'}
    if len(train)!=19 or len(frozen['rows'])!=19:raise ValueError('Training-only coverage differs')
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    pred_dir=out/'frozen_candidates';pred_dir.mkdir();automatic=[];seen=set()
    for row in frozen['rows']:
        key=(row['sequence'],row['frame'])
        if key not in train or key in seen:raise ValueError('Unexpected or repeated training read')
        seen.add(key);item=train[key]
        if digest(item['feature'])!=item['feature_sha256'] or digest(row['tensor'])!=row['tensor_sha256'] or digest(row['raw'])!=row['raw_sha256'] or digest(row['native'])!=row['native_sha256']:
            raise ValueError('Frozen input changed')
        feature=torch.load(item['feature'],map_location='cpu',weights_only=True)
        raw=torch.load(row['tensor'],map_location='cpu',weights_only=True)
        masks=raw['raw_multimasks'].float();ious=raw['raw_ious'].float()
        if feature['frame']!=row['frame'] or masks.shape[:2]!=ious.shape or masks.shape[0]!=1 or int(ious.argmax(-1)[0])!=row['selected_index']:
            raise ValueError('Capture frame or author selection differs')
        fg,contrast=feature_scores(masks,feature['feature'],feature['foreground'],feature['background'])
        indices={'predicted_iou':int(ious.argmax(-1)[0]),'foreground':int(fg.argmax()),'foreground_background':int(contrast.argmax())}
        paths=[]
        for index in range(masks.shape[1]):
            mask=(to_video(masks[:,index:index+1],tuple(row['shape']))>0).numpy()[0,0]
            path=pred_dir/f"{key[0]}_{key[1]:06d}_candidate{index}.png";save_binary_mask(mask,path)
            paths.append({'path':str(path),'sha256':digest(path)})
        chosen=np.asarray(Image.open(paths[indices['predicted_iou']]['path']).convert('L'))>0
        if not np.array_equal(chosen,np.asarray(Image.open(row['raw']).convert('L'))>0):raise ValueError('Predicted-IoU selection differs from frozen raw mask')
        automatic.append({'sequence':key[0],'frame':key[1],'shape':row['shape'],'score':row['score'],
                          'predicted_ious':ious[0].tolist(),'foreground_scores':fg.tolist(),'foreground_background_scores':contrast.tolist(),
                          'choices':indices,'candidates':paths,'native':row['native'],'native_sha256':row['native_sha256']})
    (out/'frozen_automatic_choices.json').write_text(json.dumps(automatic,indent=2)+'\n')
    # Training GT is used only below, after candidate masks and choices are saved.
    scored=[]
    for row in automatic:
        item=train[(row['sequence'],row['frame'])]
        if digest(item['label'])!=item['label_sha256']:raise ValueError('Training label changed')
        truth=load_label_mask(Path(item['label']),[1])==1
        if list(truth.shape)!=row['shape']:raise ValueError('Target shape differs')
        candidates=[]
        for c in row['candidates']:
            pred=np.asarray(Image.open(c['path']).convert('L'))>0
            intersection=int((pred&truth).sum());den=int(pred.sum()+truth.sum())
            candidates.append({'dice':2*intersection/den if den else 1.,'predicted_pixels':int(pred.sum()),'overlap_pixels':intersection})
        native=np.asarray(Image.open(row['native']).convert('L'))>0;den=int(native.sum()+truth.sum())
        scores={'native':{'dice':2*int((native&truth).sum())/den if den else 1.,'predicted_pixels':int(native.sum())}}
        for name,index in row['choices'].items():scores[name]=candidates[index]
        best=int(np.argmax([x['dice'] for x in candidates]));scores['label_oracle']=candidates[best]
        scored.append({'sequence':row['sequence'],'frame':row['frame'],'present':bool(truth.any()),'target_pixels':int(truth.sum()),
                       'object_score':row['score'],'predicted_ious':row['predicted_ious'],'foreground_scores':row['foreground_scores'],
                       'foreground_background_scores':row['foreground_background_scores'],'choices':row['choices'],
                       'label_oracle_index':best,'candidates':candidates,'scores':scores})
    present=[r for r in scored if r['present']];empty=[r for r in scored if not r['present']]
    names=('native','predicted_iou','foreground','foreground_background','label_oracle')
    summary={name:{'present_frames':len(present),'mean_present_dice':float(np.mean([r['scores'][name]['dice'] for r in present])),
                   'empty_frames':len(empty),'empty_false_alarms':sum(r['scores'][name]['predicted_pixels']>0 for r in empty),
                   'mean_empty_predicted_pixels':float(np.mean([r['scores'][name]['predicted_pixels'] for r in empty]))} for name in names}
    result={'passed':True,'scope':'Retrospective training-only frozen-read diagnostic. Raw-candidate outputs ignore native suppression only as an offline hypothetical; label_oracle is GT-selected and is not deployable. No live inference or memory writes altered.',
            'manifest_sha256':mhash,'raw_diagnostic_sha256':digest(a.raw_diagnostic/'completion.json'),
            'frozen_automatic_choices_sha256':digest(out/'frozen_automatic_choices.json'),'runner_sha256':digest(Path(__file__)),
            'summary':summary,'rows':scored}
    (out/'completion.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'summary':summary,'case24':next(x for x in scored if x['sequence']=='seq4' and x['frame']==24)},indent=2),flush=True)


if __name__=='__main__':main()
