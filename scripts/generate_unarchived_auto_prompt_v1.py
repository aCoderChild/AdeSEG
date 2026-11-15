#!/usr/bin/env python3
"""Reproduce the historical CPU prompt path without opening label masks.

Two fresh seq6 replays must match the archived prompt before generating seq1/7.
The CPython audit guard is not a protection against arbitrary native I/O.
"""
import argparse
import hashlib
from importlib.metadata import version
import json
import math
import os
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import uuid

import numpy as np
import PIL
from PIL import Image
import cv2
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest, stage_frames
from modeling.early_feature_prompt import early_feature_points
from modeling.lean_multiscale_anchor import install_multiscale_anchor
from scripts.memory_token_io import digest, validate_output
from scripts.preflight_remaining_boundary_video_v1 import (
    EXPECTED_TRAIN, EXPECTED_VALIDATION, TRAIN_PREFLIGHT,
    checked_training_preflight, image_metadata_only,
)
from scripts.run_author_baseline import build_author_predictor, source_identity
from scripts.run_prompt_feature_refinement import detect_box
from scripts.source_group_label_guard_v1 import install_mask_open_guard

BASE = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment')
DATA = ROOT/'data/polypgen_sequence.jsonl'
PROTOCOL = ROOT/'protocols/NO_ARCHIVED_PROMPT_SEQ1_SEQ7_V1.md'
ARCHIVE = BASE/'lean_auto_multiscale_v1/cohort_part_a_attempt01/seq6'
CHECKPOINT_SHA = 'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
DETECTOR_SHA = '858dfcffa13408382f612c7b57498908e11c0b28c8b24a948685806ad3c3c58e'
ALL_SEQUENCES = {f'seq{i}' for i in range(1,24)}
FILES = ('scripts/generate_unarchived_auto_prompt_v1.py', 'scripts/run_lean_auto_multiscale.py',
         'scripts/audit_unarchived_prompt_gate_v1.py',
         'scripts/run_author_baseline.py', 'scripts/run_prompt_feature_refinement.py',
         'scripts/preflight_remaining_boundary_video_v1.py', 'scripts/source_group_label_guard_v1.py',
         'scripts/memory_token_io.py', 'datasets/video.py', 'datasets/io.py', 'evaluation/segmentation.py',
         'evaluation/davis2017/metrics.py', 'training/tracking.py', 'modeling/samurai_auto_prompt.py',
         'evaluation/statistics.py', 'modeling/region_bias.py', 'modeling/region_tokens.py',
         'modeling/seq23_viewport.py', 'modeling/early_feature_prompt.py',
         'modeling/prompt_feature_refinement.py', 'modeling/lean_multiscale_anchor.py',
         'protocols/lean_auto_multiscale_early_prompt_v1.json', 'protocols/lean_multiscale_anchor_v1.json',
         'protocols/NO_ARCHIVED_PROMPT_SEQ1_SEQ7_V1.md')


def read(path):
    return json.loads(Path(path).read_text())


def output_folder(path):
    path = validate_output(path)
    if not path.is_relative_to(BASE.resolve()):
        raise ValueError('Output must be under the requested Drive experiment folder')
    path.mkdir(parents=True, exist_ok=False)
    return path


def signature(checkpoint, detector):
    checkpoint, detector = Path(checkpoint), Path(detector)
    if digest(checkpoint) != CHECKPOINT_SHA or digest(detector) != DETECTOR_SHA:
        raise ValueError('Checkpoint or detector does not match the archived automatic model')
    repo, revision = source_identity('lean')
    old = read(ARCHIVE/'setup.json')
    records=read(ARCHIVE/'records.json'); done=read(ARCHIVE/'completion.json')
    if (not done.get('passed') or done.get('records_sha256') != digest(ARCHIVE/'records.json') or
            len(records)!=1 or records[0].get('sequence')!='seq6' or
            records[0].get('prompt_sha256')!=digest(ARCHIVE/'seq6/prompt.json') or
            records[0].get('frames_sha256')!=digest(ARCHIVE/'seq6/frames.json')):
        raise ValueError('Archived seq6 reference receipt differs')
    s = old['signature']
    expected = {'revision': revision, 'checkpoint_sha256': CHECKPOINT_SHA,
                'detector_sha256': DETECTOR_SHA, 'manifest_sha256': digest(DATA),
                'protocol_sha256': digest(ROOT/'protocols/lean_auto_multiscale_early_prompt_v1.json'),
                'memory_protocol_sha256': digest(ROOT/'protocols/lean_multiscale_anchor_v1.json'),
                'early_module_sha256': digest(ROOT/'modeling/early_feature_prompt.py'),
                'prompt_module_sha256': digest(ROOT/'modeling/prompt_feature_refinement.py'),
                'memory_module_sha256': digest(ROOT/'modeling/lean_multiscale_anchor.py'),
                'lean_runner_sha256': digest(ROOT/'scripts/run_author_baseline.py'),
                'runner_sha256': digest(ROOT/'scripts/run_lean_auto_multiscale.py'),
                'bypass_sha256': digest(repo/'checkpoints/bypass_bottleneck_base.pth'),
                'lean_config_sha256': digest(repo/'configs/lean_sam2/lean_sam2.yaml'),
                'torch': torch.__version__, 'ultralytics': version('ultralytics'),
                'numpy': np.__version__, 'python': sys.version,
                'device': 'cpu float32; 2 threads'}
    if old.get('mode') != 'cohort' or old.get('sequences') != ['seq6'] or any(s.get(k) != v for k,v in expected.items()):
        raise ValueError('Current automatic prompt environment differs from archived seq6')
    # The historical receipt did not record the hash of the detector helper.
    # Its current hash is bound below, but cannot prove the historical bytes.
    return {'checkpoint_sha256': CHECKPOINT_SHA, 'detector_sha256': DETECTOR_SHA,
            'manifest_sha256': digest(DATA), 'author': expected,
            'source_sha256': {name: digest(ROOT/name) for name in FILES},
            'archived_setup_sha256': digest(ARCHIVE/'setup.json'),
            'archived_records_sha256': digest(ARCHIVE/'records.json'),
            'archived_completion_sha256': digest(ARCHIVE/'completion.json'),
            'archived_prompt_sha256': digest(ARCHIVE/'seq6/prompt.json'),
            'archived_frames_sha256': digest(ARCHIVE/'seq6/frames.json'),
            'archived_mask_sha256': digest(ARCHIVE/'seq6/prompt_refined.png'),
            'archived_detector_helper_hash_was_not_recorded': True,
            'pillow':PIL.__version__, 'opencv':cv2.__version__,
            'archived_pillow_and_opencv_versions_were_not_recorded':True,
            'pythonhashseed': os.environ.get('PYTHONHASHSEED'), 'device': 'cpu',
            'torch_num_threads': 2}, repo


def image_inventory(frames):
    result, size = [], None
    for index, frame in enumerate(frames):
        if frame.frame_index != index:
            raise ValueError('Manifest must contain contiguous source ordinals')
        sha = digest(frame.image_path)
        with Image.open(frame.image_path) as im:
            current = im.size
            im.verify()
        if sha != digest(frame.image_path) or (size is not None and size != current):
            raise ValueError('Image changed during inspection or geometry differs')
        size = current
        result.append({'frame': index, 'source_frame_index': frame.frame_index,
                       'image_path': str(frame.image_path.resolve()), 'image_sha256': sha})
    if not result:
        raise ValueError('No input frames')
    return result, size


def screen_target_duplicates(videos, training):
    """Screen seq1/7 against all sequences using bytes and decoded pixels.

    Training and validation bytes must still match their frozen capture
    preflight. This is not a near-duplicate or patient-identity audit.
    """
    targets={'seq1','seq7'}
    if set(videos)!=ALL_SEQUENCES:raise ValueError('Duplicate-screen sequence coverage differs')
    if training.get('manifest_sha256') != digest(DATA):
        raise ValueError('Training data manifest differs')
    frozen_sequences=EXPECTED_TRAIN | EXPECTED_VALIDATION
    if not frozen_sequences.issubset(videos):raise ValueError('Training/validation sequence is missing')
    raw,decoded,rows,counts={},{},[],{}
    for sequence in sorted(videos,key=lambda s:(s not in targets,s)):
        frames=videos[sequence];counts[sequence]=len(frames)
        old=training['sequences'].get(sequence) if sequence in frozen_sequences else None
        if sequence in frozen_sequences and old is None:
            raise ValueError('Frozen training/validation sequence is missing')
        if old is not None and (old.get('frames')!=len(frames) or len(old.get('image_sha256',[]))!=len(frames)):
            raise ValueError('Frozen training/validation image count differs')
        for index,frame in enumerate(frames):
            if frame.frame_index!=index:raise ValueError('Image ordinal differs')
            sha=digest(frame.image_path)
            if old is not None and sha!=old['image_sha256'][index]:
                raise ValueError('Frozen training/validation image differs')
            with Image.open(frame.image_path) as image:
                rgb=image.convert('RGB')
                rgb_sha=hashlib.sha256(str(rgb.size).encode('ascii')+rgb.tobytes()).hexdigest()
            if digest(frame.image_path)!=sha:raise ValueError('Image changed during duplicate screening')
            for owners,key in ((raw,sha),(decoded,rgb_sha)):
                prior=owners.get(key,set())
                if any(owner!=sequence and (owner in targets or sequence in targets) for owner in prior):
                    raise ValueError('Target image duplicates a different sequence')
                owners.setdefault(key,set()).add(sequence)
            rows.append({'sequence':sequence,'frame':index,'image_sha256':sha,'decoded_rgb_sha256':rgb_sha})
    if set(counts)!=set(videos) or any(n<=0 for n in counts.values()):
        raise ValueError('Duplicate-screen coverage differs')
    return {'passed':True,'target_sequences':sorted(targets),
            'scope':'All manifest sequences, target-versus-other exact-byte and decoded-RGB checks only.',
            'training_preflight_sha256':digest(TRAIN_PREFLIGHT/'manifest.json'),
            'counts':counts,'rows':rows,'target_cross_sequence_exact_or_decoded_matches':0,
            'limitation':'Does not exclude near duplicates, unseen extra data or patient metadata errors.'}


def validate_detection_trace(attempts,inventory,prompt):
    """Bind every detection call to its chronological input and terminal box."""
    first=prompt.get('first_detection_index')
    if first is not None and (type(first) is not int or not 0<=first<len(inventory)):
        raise ValueError('Detection frame is invalid')
    expected=len(inventory) if first is None else first+1
    if not isinstance(attempts,list) or len(attempts)!=expected:
        raise ValueError('Detection trace coverage differs')
    for index,(attempt,item) in enumerate(zip(attempts,inventory,strict=False)):
        if attempt.get('frame')!=index or attempt.get('image_sha256')!=item['image_sha256']:
            raise ValueError('Detection trace input order differs')
        if index!=first and (attempt.get('box') is not None or attempt.get('confidence') is not None):
            raise ValueError('Detection trace continued after an earlier valid box')
    if first is None:
        if prompt.get('box') is not None or prompt.get('confidence') is not None:
            raise ValueError('No-detection prompt has a box')
    else:
        box,confidence=prompt.get('box'),prompt.get('confidence')
        if (not isinstance(box,list) or len(box)!=4 or confidence is None or
                not all(math.isfinite(float(v)) for v in [*box,confidence]) or box[0]>=box[2] or box[1]>=box[3] or
                attempts[-1].get('box')!=box or attempts[-1].get('confidence')!=confidence):
            raise ValueError('Detection trace terminal box differs from prompt')


def scan_first(frames, inventory, detector, detect=detect_box):
    """Scan chronologically through first valid detector box, never after it."""
    attempts = []
    for index, frame in enumerate(frames):
        if digest(frame.image_path) != inventory[index]['image_sha256']:
            raise ValueError('Detector image differs from frozen input')
        with Image.open(frame.image_path) as im:
            image = np.asarray(im.convert('RGB'))
        if digest(frame.image_path) != inventory[index]['image_sha256']:
            raise ValueError('Detector image changed during decoding')
        box, confidence = detect(detector, image)
        if (box is None) != (confidence is None):
            raise ValueError('Detector returned a partial detection')
        if box is not None:
            h,w = image.shape[:2]
            if len(box) != 4 or not all(math.isfinite(float(v)) for v in [*box, confidence]):
                raise ValueError('Detector returned nonfinite geometry')
            if not (0 <= box[0] < box[2] <= w and 0 <= box[1] < box[3] <= h):
                raise ValueError('Detector box is invalid')
        attempts.append({'frame': index, 'image_sha256': inventory[index]['image_sha256'],
                         'box': box, 'confidence': confidence})
        if box is not None:
            return index, box, confidence, attempts
    return None, None, None, attempts


def state_fingerprints(model):
    result = {}
    for name, value in model.state_dict().items():
        raw = value.detach().cpu().contiguous().reshape(-1).view(torch.uint8).numpy().tobytes()
        result[name] = {'shape': list(value.shape), 'dtype': str(value.dtype),
                        'sha256': hashlib.sha256(raw).hexdigest()}
    return result


def binary_mask(logits, expected_hw):
    if not torch.is_tensor(logits) or logits.ndim != 4 or logits.shape[:2] != (1,1):
        raise ValueError('Predictor output shape differs')
    if tuple(logits.shape[-2:]) != tuple(expected_hw) or not torch.isfinite(logits).all():
        raise ValueError('Predictor output is nonfinite or misaligned')
    return logits[0,0].detach().cpu().numpy() > 0


def save_mask(mask, path):
    Image.fromarray(np.asarray(mask, dtype=np.uint8)*255).save(path)
    return digest(path)


def generate_prompt(model, frames, first, box, folder, size):
    """Use the historical box-to-early-points prompt route, without propagation."""
    before = state_fingerprints(model)
    selections=[]
    install_multiscale_anchor(model,.5,selection_events=selections)
    with tempfile.TemporaryDirectory(prefix='ordered_frames_', dir=folder) as temporary:
        stage_frames(frames, Path(temporary))
        staged = sorted(Path(temporary).glob('*.jpg'), key=lambda x:int(x.stem))
        if len(staged) != len(frames) or any(x.resolve() != f.image_path.resolve() for x,f in zip(staged,frames)):
            raise ValueError('Staged frame order differs')
        state = model.init_state(video_path=temporary, offload_video_to_cpu=True, offload_state_to_cpu=False)
        if state['num_frames'] != len(frames):
            raise ValueError('Predictor frame count differs')
        try:
            with torch.inference_mode():
                _, ids, initial = model.add_new_points_or_box(state,frame_idx=first,obj_id=1,
                                                               box=np.asarray(box,dtype=np.float32))
                if ids != [1]: raise ValueError('Unexpected prompt object')
                provisional = binary_mask(initial, (size[1],size[0]))
                provisional_sha = save_mask(provisional,folder/'prompt_provisional.png')
                _, backbone, _, _, _ = model._get_image_feature(state,first,batch_size=1)
                early = backbone['backbone_fpn'][0].detach().float().cpu().contiguous()
                points,status = early_feature_points(backbone,provisional,box,seed=20261114)
                point_data = None
                if points is not None:
                    point_data = points['feature_points']
                    if (len(point_data) != 2 or any(len(p) != 2 or not all(math.isfinite(float(v)) for v in p) for p in point_data)
                            or any(not (0<=p[0]<size[0] and 0<=p[1]<size[1]) for p in point_data)):
                        raise ValueError('Automatic feature points have invalid geometry')
                    _, ids, initial = model.add_new_points_or_box(state,frame_idx=first,obj_id=1,
                        points=np.asarray(point_data,dtype=np.float32),labels=np.array([1,0],dtype=np.int32),
                        clear_old_points=False)
                    if ids != [1]: raise ValueError('Unexpected point object')
                refined = binary_mask(initial, (size[1],size[0]))
                refined_sha = save_mask(refined,folder/'prompt_refined.png')
        finally:
            model.reset_state(state)
    after = state_fingerprints(model)
    if before != after:
        raise ValueError('Predictor parameters or buffers changed')
    (folder/'model_state.json').write_text(json.dumps(before,indent=2)+'\n')
    return {'point_status':status, 'points':point_data,
            'feature_shape':list(early.shape), 'early_feature_sha256':hashlib.sha256(early.numpy().tobytes()).hexdigest(),
            'provisional_mask_sha256':provisional_sha, 'refined_mask_sha256':refined_sha,
            'lean_mask_status':'automatic_mask' if refined.any() else 'empty_prediction_box_only',
            'status':'detected_nonempty' if refined.any() else 'detected_empty_box_only',
            'model_state_sha256':digest(folder/'model_state.json')}


def save_no_detection_outputs(folder, inventory, size):
    """Explicitly freeze all four arm masks when no object was initialized."""
    zero = np.zeros((size[1],size[0]),dtype=bool)
    arms = {}
    for arm in ('A','B','C','D'):
        path=folder/arm;path.mkdir()
        rows=[]
        for item in inventory:
            file=path/f"{item['frame']:06d}.png"
            rows.append({'frame':item['frame'],'sha256':save_mask(zero,file)})
        arms[arm]=rows
    return arms


def verify_reference_matches(result, folder):
    old = read(ARCHIVE/'seq6/prompt.json')
    keys=('first_detection_index','box','confidence','point_status','points','lean_mask_status',
          'provisional_mask_sha256','feature_shape','early_feature_sha256','refined_mask_sha256')
    if any(result.get(k) != old.get(k) for k in keys):
        raise ValueError('Regenerated seq6 prompt differs from archived automatic prompt')
    if digest(folder/'prompt_refined.png') != digest(ARCHIVE/'seq6/prompt_refined.png'):
        raise ValueError('Regenerated seq6 prompt mask differs')


def verify_reference_pair(a,b,current_sig):
    a,b=Path(a),Path(b)
    if not a.resolve().is_relative_to(BASE.resolve()) or not b.resolve().is_relative_to(BASE.resolve()):
        raise ValueError('Reference replays must remain inside the Drive experiment folder')
    if a.resolve() == b.resolve() or os.path.samefile(a,b):
        raise ValueError('Reference replays must be separate directories')
    ra,rb=read(a/'completion.json'),read(b/'completion.json')
    if (type(ra.get('pid')) is not int or type(rb.get('pid')) is not int or ra['pid']==rb['pid'] or
            not ra.get('run_uuid') or not rb.get('run_uuid') or ra['run_uuid']==rb['run_uuid']):
        raise ValueError('Reference replays must use distinct process identities')
    for folder,rec in ((a,ra),(b,rb)):
        if not rec.get('passed') or rec.get('mode') != 'reference' or rec.get('sequence') != 'seq6' or rec.get('signature') != current_sig:
            raise ValueError('Reference replay identity differs')
        if rec.get('label_guard',{}).get('open_attempts') != 0:
            raise ValueError('Reference replay attempted label access')
        if (digest(folder/'prompt.json') != rec.get('prompt_sha256') or
                digest(folder/'frames.json') != rec.get('frames_sha256') or
                digest(folder/'detections.json') != rec.get('detections_sha256')):
            raise ValueError('Reference artifacts changed')
        result=read(folder/'prompt.json');inventory=read(folder/'frames.json')
        validate_detection_trace(read(folder/'detections.json'),inventory,result)
        verify_reference_matches(result,folder)
        if digest(folder/'prompt_provisional.png') != result['provisional_mask_sha256']:
            raise ValueError('Reference provisional mask differs')
        if digest(folder/'model_state.json') != result['model_state_sha256']:
            raise ValueError('Reference predictor state differs')
    common_a={k:v for k,v in ra.items() if k not in ('pid','run_uuid')}
    common_b={k:v for k,v in rb.items() if k not in ('pid','run_uuid')}
    if common_a != common_b or read(a/'prompt.json') != read(b/'prompt.json') or read(a/'frames.json') != read(b/'frames.json'):
        raise ValueError('Independent reference replays differ')
    return {'passed':True,'reference_a':str(a.resolve()),'reference_b':str(b.resolve()),
            'reference_a_sha256':digest(a/'completion.json'),'reference_b_sha256':digest(b/'completion.json'),
            'signature':current_sig,'scope':'Two separate CPU prompt replays; not full-video inference.'}


def load_gate(path,current_sig):
    path=Path(path); rec=read(path/'gate.json'); done=read(path/'completion.json')
    if not path.resolve().is_relative_to(BASE.resolve()):
        raise ValueError('Reference gate must remain inside the Drive experiment folder')
    if not done.get('passed') or done.get('gate_sha256') != digest(path/'gate.json') or not rec.get('passed'):
        raise ValueError('Reference gate is missing or changed')
    fresh=verify_reference_pair(rec['reference_a'],rec['reference_b'],current_sig)
    if fresh != rec:
        raise ValueError('Reference gate identity changed')
    return rec


def verify_independent_audit(path,gate_folder,gate):
    """Require the separately checked saved-artifact audit before target YOLO."""
    path,gate_folder=Path(path),Path(gate_folder)
    if not path.resolve().is_relative_to(BASE.resolve()):
        raise ValueError('Independent audit must remain inside the Drive experiment folder')
    audit=read(path);a,b=Path(gate['reference_a']),Path(gate['reference_b'])
    expected={'passed':True,'gate_sha256':digest(gate_folder/'gate.json'),
              'audit_source_sha256':digest(ROOT/'scripts/audit_unarchived_prompt_gate_v1.py'),
              'input_frames_checked':len(read(a/'frames.json')),
              'detector_attempts_checked':len(read(a/'detections.json')),'reference_runs_checked':2}
    if (any(audit.get(k)!=v for k,v in expected.items()) or
            audit.get('label_guard',{}).get('open_attempts')!=0 or
            type(audit.get('label_guard',{}).get('blocked_paths')) is not int or audit['label_guard']['blocked_paths']<=0):
        raise ValueError('Independent prompt gate audit differs')
    runs=audit.get('runs',[])
    if len(runs)!=2:raise ValueError('Independent reference audit coverage differs')
    for folder,row in zip((a,b),runs,strict=True):
        completion=read(folder/'completion.json');prompt=read(folder/'prompt.json')
        checks={'completion_sha256':digest(folder/'completion.json'),'pid':completion['pid'],
                'run_uuid':completion['run_uuid'],'frames_sha256':completion['frames_sha256'],
                'detections_sha256':completion['detections_sha256'],
                'model_state_sha256':prompt['model_state_sha256'],'prompt_sha256':completion['prompt_sha256']}
        if any(row.get(k)!=v for k,v in checks.items()):
            raise ValueError('Independent audited reference identity differs')
    return audit


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('reference','seal-gate','target'),required=True)
    p.add_argument('--sequence',choices=('seq1','seq7','seq6'))
    p.add_argument('--checkpoint',type=Path,required=True);p.add_argument('--detector',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--reference-a',type=Path);p.add_argument('--reference-b',type=Path);p.add_argument('--gate',type=Path)
    p.add_argument('--independent-audit',type=Path)
    args=p.parse_args()
    if (args.mode=='reference' and args.sequence!='seq6') or (args.mode=='target' and args.sequence not in ('seq1','seq7')):
        raise ValueError('Sequence differs from the declared mode')
    if args.mode=='seal-gate' and (args.sequence is not None or not args.reference_a or not args.reference_b):
        raise ValueError('Gate requires exactly two reference directories')
    if args.mode=='target' and (args.gate is None or args.independent_audit is None):
        raise ValueError('Target requires the sealed seq6 reference gate and independent audit')
    if os.environ.get('PYTHONHASHSEED') != '0':
        raise ValueError('Launch with PYTHONHASHSEED=0')
    guard=install_mask_open_guard(DATA)
    torch.set_num_threads(2)
    sig,repo=signature(args.checkpoint,args.detector)
    if args.mode=='seal-gate':
        gate=verify_reference_pair(args.reference_a,args.reference_b,sig)
        if guard['open_attempts'] != 0: raise ValueError('Label-mask access attempted')
        out=output_folder(args.output);(out/'gate.json').write_text(json.dumps(gate,indent=2)+'\n')
        (out/'completion.json').write_text(json.dumps({'passed':True,'gate_sha256':digest(out/'gate.json')},indent=2)+'\n')
        print(json.dumps({'passed':True,'mode':'seal-gate'}));return
    if args.mode=='target':
        gate=load_gate(args.gate,sig)
        verify_independent_audit(args.independent_audit,args.gate,gate)
    videos={**load_manifest(DATA,'dev'),**load_manifest(DATA,'test')};frames=videos[args.sequence]
    inventory,size=image_inventory(frames)
    if args.mode=='reference':
        inds,hashes=image_metadata_only(ARCHIVE/'seq6/frames.json')
        if [x['source_frame_index'] for x in inventory] != inds or [x['image_sha256'] for x in inventory] != hashes:
            raise ValueError('Seq6 archived source images differ')
    screen=None
    if args.mode=='target':
        screen=screen_target_duplicates(videos,checked_training_preflight(TRAIN_PREFLIGHT))
    out=output_folder(args.output)
    (out/'frames.json').write_text(json.dumps(inventory,indent=2)+'\n')
    if screen is not None:(out/'duplicate_screen.json').write_text(json.dumps(screen,indent=2)+'\n')
    settings=out/'ultralytics';settings.mkdir()
    os.environ['YOLO_CONFIG_DIR']=str(settings);os.environ['WANDB_MODE']='disabled'
    from ultralytics import YOLO
    detector=YOLO(str(args.detector))
    if detector.names != {0:'polyp'}:raise ValueError('Detector class differs')
    first,box,confidence,attempts=scan_first(frames,inventory,detector)
    (out/'detections.json').write_text(json.dumps(attempts,indent=2)+'\n')
    result={'first_detection_index':first,'box':box,'confidence':confidence,'point_status':'no_detection',
            'points':None,'lean_mask_status':'no_detection','status':'no_detection',
            'post_prompt_frames': 0 if first is None else len(frames)-first-1}
    validate_detection_trace(attempts,inventory,result)
    if first is None:
        result['empty_arm_masks']=save_no_detection_outputs(out,inventory,size)
    else:
        model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=args.checkpoint,device='cpu'),repo)
        result.update(generate_prompt(model,frames,first,box,out,size))
        del model
    if args.mode=='reference':verify_reference_matches(result,out)
    if guard['open_attempts'] != 0:raise ValueError('Label-mask access attempted')
    if any(digest(frame.image_path) != item['image_sha256'] for frame,item in zip(frames,inventory,strict=True)):
        raise ValueError('An input image changed during prompt generation')
    if signature(args.checkpoint,args.detector)[0] != sig:raise ValueError('Sources or environment changed')
    (out/'prompt.json').write_text(json.dumps(result,indent=2)+'\n')
    completion={'passed':True,'mode':args.mode,'sequence':args.sequence,'signature':sig,
                'pid':os.getpid(),'run_uuid':str(uuid.uuid4()),
                'prompt_sha256':digest(out/'prompt.json'),'frames_sha256':digest(out/'frames.json'),
                'detections_sha256':digest(out/'detections.json'),'label_guard':dict(guard),
                'duplicate_screen_sha256':digest(out/'duplicate_screen.json') if screen is not None else None,
                'reference_gate_sha256':digest(args.gate/'gate.json') if args.gate else None,
                'independent_audit_sha256':digest(args.independent_audit) if args.independent_audit else None,
                'limitation':'Annotation guard covers CPython opens, not arbitrary native I/O; historical detector-helper SHA was not recorded.'}
    (out/'completion.json').write_text(json.dumps(completion,indent=2)+'\n')
    print(json.dumps({'passed':True,'mode':args.mode,'sequence':args.sequence,'status':result['status'],
                      'detection_frame':first,'label_open_attempts':guard['open_attempts']}))


if __name__=='__main__':main()
