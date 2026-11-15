#!/usr/bin/env python3
"""Score two audited no-detection videos after all predictions were frozen.

No object was initialized, so there is no post-prompt frame or video-refiner
inference. Report all-frame detection misses and labeled-empty false alarms;
never average these cases as if their post-prompt Dice were zero or one.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.generate_unarchived_auto_prompt_v1 import (BASE,DATA,load_gate,signature,
    validate_detection_trace,verify_independent_audit)
from scripts.memory_token_io import digest,validate_output

SEQUENCES=('seq1','seq7');ARMS=('A','B','C','D')
CHECKPOINT=BASE/'author_model_assets/SAM2.1/sam2.1_hiera_base_plus.pt'
DETECTOR=ROOT/'checkpoints/polypgen_yolov8n.pt'
TARGET_AUDITOR=ROOT/'scripts/audit_unarchived_no_detection_targets_v1.py'


def read(path):return json.loads(Path(path).read_text())


def require(ok,message):
    if not ok:raise ValueError(message)


def install_evidence_label_guard(manifest):
    """Forbid CPython mask opens until evidence validation has completed."""
    manifest=Path(manifest);blocked=set()
    for line in manifest.read_text().splitlines():
        if line.strip():
            path=Path(json.loads(line)['mask'])
            blocked.add(os.path.realpath(path if path.is_absolute() else manifest.parent/path))
    state={'blocked_paths':len(blocked),'verification_open_attempts':0,'allow_score':False}
    def hook(event,args):
        if event!='open' or not args or not isinstance(args[0],(str,bytes,os.PathLike)):return
        if not state['allow_score'] and os.path.realpath(os.fsdecode(args[0])) in blocked:
            state['verification_open_attempts']+=1
            raise PermissionError('Label open before no-detection evidence validation')
    sys.addaudithook(hook)
    return state


def verify_one_target(folder,sequence,frames,sig,gate_sha,reference_audit_sha):
    """Rehash and decode frozen predictions; no annotation mask is opened."""
    folder=Path(folder);rec=read(folder/'completion.json');prompt=read(folder/'prompt.json')
    expected={'passed':True,'mode':'target','sequence':sequence,'signature':sig,
              'reference_gate_sha256':gate_sha,'independent_audit_sha256':reference_audit_sha}
    require(all(rec.get(k)==v for k,v in expected.items()),'Target completion identity differs')
    require(rec.get('label_guard',{}).get('open_attempts')==0,'Target attempted annotation access')
    for key,name in (('prompt_sha256','prompt.json'),('frames_sha256','frames.json'),
                     ('detections_sha256','detections.json'),('duplicate_screen_sha256','duplicate_screen.json')):
        require(rec.get(key)==digest(folder/name),f'Target {name} hash differs')
    inventory=read(folder/'frames.json');attempts=read(folder/'detections.json')
    require(len(inventory)==len(frames) and len(frames)>0,'Target frame count differs')
    for index,(row,frame) in enumerate(zip(inventory,frames,strict=True)):
        require(row=={'frame':index,'source_frame_index':frame.frame_index,
                      'image_path':str(frame.image_path.resolve()),'image_sha256':digest(frame.image_path)},
                'Target source image identity differs')
    validate_detection_trace(attempts,inventory,prompt)
    require(prompt.get('status')=='no_detection' and prompt.get('first_detection_index') is None and
            prompt.get('box') is None and prompt.get('confidence') is None and prompt.get('points') is None and
            prompt.get('point_status')=='no_detection' and prompt.get('lean_mask_status')=='no_detection' and
            prompt.get('post_prompt_frames')==0,'Target is not a no-detection case')
    require(set(prompt.get('empty_arm_masks',{}))==set(ARMS),'Four empty outputs missing')
    for arm in ARMS:
        rows=prompt['empty_arm_masks'][arm]
        require([x['frame'] for x in rows]==list(range(len(frames))),'Prediction coverage differs')
        for index,(row,frame) in enumerate(zip(rows,frames,strict=True)):
            file=folder/arm/f'{index:06d}.png'
            require(digest(file)==row['sha256'],'Prediction hash differs')
            with Image.open(frame.image_path) as source:size=source.size
            with Image.open(file) as image:
                require(image.format=='PNG' and image.size==size and not np.asarray(image.convert('L')).any(),
                        'No-detection prediction is not an image-sized empty mask')
    return rec,prompt,inventory,attempts


def verify_evidence(target_a,target_b,gate_folder,reference_audit,target_audit,videos,sig):
    """Fail closed on reference gate, independent audit and both target repeats."""
    a,b=Path(target_a).resolve(),Path(target_b).resolve()
    require(a!=b and all(p.is_relative_to(BASE.resolve()) for p in (a,b)), 'Target roots differ or alias')
    require(Path(target_audit).resolve().is_relative_to(BASE.resolve()),'Independent target audit outside experiment folder')
    gate=load_gate(gate_folder,sig);verify_independent_audit(reference_audit,gate_folder,gate)
    audit=read(target_audit)
    expected={'passed':True,'audit_source_sha256':digest(TARGET_AUDITOR),
              'gate_sha256':digest(Path(gate_folder)/'gate.json'),
              'reference_audit_sha256':digest(reference_audit),
              'target_roots':[str(a),str(b)],'input_frames_checked':sum(len(videos[s]) for s in SEQUENCES),
              'detector_attempts_checked':2*sum(len(videos[s]) for s in SEQUENCES),
              'prediction_masks_checked':8*sum(len(videos[s]) for s in SEQUENCES),
              'screen_frames_checked':sum(len(frames) for frames in videos.values())}
    require(all(audit.get(k)==v for k,v in expected.items()) and
            audit.get('label_guard',{}).get('open_attempts')==0 and
            type(audit.get('label_guard',{}).get('blocked_paths')) is int and
            audit['label_guard']['blocked_paths']>0,'Independent target audit differs')
    evidence={}
    for seq in SEQUENCES:
        pair=[];audited=audit['completion_sha256'][seq]
        require(len(audited)==2,'Independent target audit run coverage differs')
        for root,audited_row in zip((a,b),audited,strict=True):
            folder=root/seq
            rec,prompt,inventory,attempts=verify_one_target(folder,seq,videos[seq],sig,
                expected['gate_sha256'],expected['reference_audit_sha256'])
            require(audited_row['folder']==str(folder.resolve()) and
                    audited_row['completion_sha256']==digest(folder/'completion.json') and
                    audited_row['pid']==rec['pid'] and audited_row['run_uuid']==rec['run_uuid'] and
                    audited_row['duplicate_screen_sha256']==rec['duplicate_screen_sha256'],
                    'Independent audit target identity differs')
            pair.append((rec,prompt,inventory,attempts))
        x,y=pair
        require(x[0]['pid']!=y[0]['pid'] and x[0]['run_uuid']!=y[0]['run_uuid'] and
                x[1:]==y[1:],'Independent target replays differ')
        evidence[seq]={'folder':str((a/seq).resolve()),'completion_sha256':digest(a/seq/'completion.json'),
                       'replay_completion_sha256':digest(b/seq/'completion.json'),
                       'prompt':x[1],'frames':x[2]}
    return evidence,audit


def score(videos,evidence):
    """Read labels only after all frozen predictions and independent audit pass."""
    rows={};summaries={}
    for seq in SEQUENCES:
        frame_rows=[];folder=Path(evidence[seq]['folder']);present=0;empty=0
        for i,frame in enumerate(videos[seq]):
            image_sha=evidence[seq]['frames'][i]['image_sha256']
            require(digest(frame.image_path)==image_sha,'Source image changed before scoring')
            expected_masks={arm:evidence[seq]['prompt']['empty_arm_masks'][arm][i]['sha256'] for arm in ARMS}
            for arm,expected in expected_masks.items():
                require(digest(folder/arm/f'{i:06d}.png')==expected,'Prediction changed before scoring')
            label_sha=digest(frame.mask_path)
            truth=load_label_mask(frame.mask_path,[1])==1
            require(digest(frame.mask_path)==label_sha,'Label changed during decoding')
            with Image.open(frame.image_path) as image:
                require(truth.shape==(image.height,image.width),'Label geometry differs')
            require(digest(frame.image_path)==image_sha,'Source image changed during scoring')
            for arm,expected in expected_masks.items():
                require(digest(folder/arm/f'{i:06d}.png')==expected,'Prediction changed during scoring')
            visible=bool(truth.any());present+=int(visible);empty+=int(not visible)
            frame_rows.append({'frame':i,'source_frame_index':frame.frame_index,
                'image_sha256':image_sha,'label_sha256':label_sha,
                'present':visible,'prediction_sha256':expected_masks,
                'predicted_pixels':0,'dice_on_present':0.0 if visible else None,
                'recall_on_present':0.0 if visible else None,'empty_false_alarm':False if not visible else None})
        rows[seq]=frame_rows
        summaries[seq]={'frame_count':len(frame_rows),'detector_calls':len(frame_rows),
            'initialization':'no_detection','prompt_frame':None,'post_prompt_frame_count':0,
            'post_prompt_present_dice':None,'all_frame_present':present,'all_frame_empty':empty,
            'all_frame_present_dice':0.0 if present else None,'all_frame_present_recall':0.0 if present else None,
            'pre_initialization_present_missed':present,'zero_overlap_present':present,
            'labeled_empty_false_alarms':0,'labeled_empty_denominator':empty,'empty_predicted_pixels_mean':0.0 if empty else None,
            'arms':'A/B/C/D identical empty outputs; no object initialization or refiner inference'}
    return rows,summaries


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for name in ('target-a','target-b','gate','reference-audit','target-audit','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    require(os.environ.get('PYTHONHASHSEED')=='0','Launch with PYTHONHASHSEED=0')
    out=validate_output(a.output)
    require(out.is_relative_to(BASE.resolve()),'Output outside requested Drive experiment folder')
    require(not out.exists(),'Output already exists')
    guard=install_evidence_label_guard(DATA);torch.set_num_threads(2)
    sig,_=signature(CHECKPOINT,DETECTOR)
    videos={**load_manifest(DATA,'dev'),**load_manifest(DATA,'test')}
    evidence,audit=verify_evidence(a.target_a,a.target_b,a.gate,a.reference_audit,a.target_audit,videos,sig)
    require(guard['verification_open_attempts']==0,'Label access attempted before evidence validation')
    out.mkdir(parents=True,exist_ok=False)
    guard['allow_score']=True
    rows,summaries=score(videos,evidence)
    result={'passed':True,'scope':'Previously inspected no-initialization sequences; all-frame evaluation only.',
        'patient_mapping':'User reports each sequence is a different patient; no patient IDs independently verified.',
        'gate_sha256':digest(a.gate/'gate.json'),'reference_audit_sha256':digest(a.reference_audit),
        'target_audit_sha256':digest(a.target_audit),'target_audit_source_sha256':digest(TARGET_AUDITOR),
        'generator_sha256':digest(ROOT/'scripts/generate_unarchived_auto_prompt_v1.py'),
        'scorer_sha256':digest(Path(__file__)),'data_manifest_sha256':digest(DATA),
        'evidence_completion_sha256':{s:[evidence[s]['completion_sha256'],evidence[s]['replay_completion_sha256']] for s in SEQUENCES},
        'verification_guard':{'blocked_paths':guard['blocked_paths'],'verification_open_attempts':guard['verification_open_attempts']},
        'summaries':summaries,'rows':rows,
        'limitations':['No detection is not clinical absence.','No object was initialized and no refiner video inference ran.',
          'CPython open guard does not cover arbitrary native I/O.','Exact/decoded duplicate checks do not exclude near duplicates.',
          'Label hashes were first recorded at scoring, not frozen before inference.',
          'The detector training split and clinical annotation completeness are unknown.']}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    lines=['# No-initialization all-frame automatic results','',
      'Both fresh CPU detector replays found no box on either sequence. A/B/C/D are identical empty outputs; no model/refiner propagation occurred.',
      'Post-prompt Dice is not defined because there was no prompt. All counts below include every frame.', '',
      '| Sequence | Frames | Labeled-present | Missed present | Labeled-empty false alarms |', '|---|---:|---:|---:|---:|']
    for s in SEQUENCES:
        x=summaries[s];lines.append(f"| {s} | {x['frame_count']} | {x['all_frame_present']} | {x['pre_initialization_present_missed']} | {x['labeled_empty_false_alarms']}/{x['labeled_empty_denominator']} |")
    lines += ['', 'Labeled-empty is not necessarily clinically confirmed absence. Label hashes were first recorded at scoring. User reports each sequence is a distinct patient; these are previously inspected development cases.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),
        'report_sha256':digest(out/'report.md')},indent=2)+'\n')
    print(json.dumps({'passed':True,'summary':summaries,'output':str(out)}))


if __name__=='__main__':main()
