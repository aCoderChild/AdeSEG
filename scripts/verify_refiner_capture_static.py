#!/usr/bin/env python3
"""Read-only static check of the historical raw-window capture pipeline.

This does not construct or authorize a live predictor. It checks only recorded
capture metadata against actual local source, checkpoint and software versions.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys
import importlib.metadata

import numpy as np
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.train_multilevel_refiner_fixed_read_pilot import check_manifest
from datasets.video import load_manifest

FORK=ROOT/'external/Lean-SAM2-token-window'
RAW_FILES={
    'runner_sha256':ROOT/'scripts/run_lean_token_window.py',
    'protocol_sha256':ROOT/'protocols/lean_token_window_candidate_v1.json',
    'policy_sha256':ROOT/'modeling/token_memory_window.py',
    'runtime_sha256':ROOT/'modeling/lean_token_window_storage.py',
    'observer_sha256':ROOT/'modeling/lean_token_storage_observer.py',
}
SOURCE_FILES={
    'manifest_sha256':ROOT/'data/polypgen_sequence.jsonl',
    'protocol_sha256':ROOT/'protocols/lean_auto_multiscale_early_prompt_v1.json',
    'memory_protocol_sha256':ROOT/'protocols/lean_multiscale_anchor_v1.json',
    'early_module_sha256':ROOT/'modeling/early_feature_prompt.py',
    'prompt_module_sha256':ROOT/'modeling/prompt_feature_refinement.py',
    'memory_module_sha256':ROOT/'modeling/lean_multiscale_anchor.py',
    'lean_runner_sha256':ROOT/'scripts/run_author_baseline.py',
    'runner_sha256':ROOT/'scripts/run_lean_auto_multiscale.py',
    'bypass_sha256':ROOT/'external/Lean-SAM2/checkpoints/bypass_bottleneck_base.pth',
    'lean_config_sha256':ROOT/'external/Lean-SAM2/configs/lean_sam2/lean_sam2.yaml',
}


def check_files(expected, files):
    for key,path in files.items():
        if digest(path)!=expected[key]:
            raise ValueError(f'Static capture input differs: {key}')


def check_canonical_pipeline(pipeline, expected_sha):
    raw=json.dumps(pipeline,sort_keys=True,separators=(',',':')).encode()
    if hashlib.sha256(raw).hexdigest()!=expected_sha:
        raise ValueError('Capture pipeline hash differs')


def check_detector(detector, expected_sha):
    detector=detector.resolve()
    if detector!=(ROOT/'checkpoints/polypgen_yolov8n.pt').resolve() or digest(detector)!=expected_sha:
        raise ValueError('Captured detector path or hash differs')
    return detector


def verify(preflight, split_preflight, capture, checkpoint, detector):
    manifest, manifest_sha=check_manifest(preflight)
    channel=json.loads((capture/'channel_feature_receipt.json').read_text())
    utility=json.loads((capture/'utility_capture_receipt.json').read_text())
    if not channel['passed'] or not utility['passed'] or digest(capture/'channel_feature_receipt.json')!=utility['feature_receipt_sha256']:
        raise ValueError('Capture receipts differ')
    pipeline=channel['pipeline']; check_canonical_pipeline(pipeline,manifest['feature_pipeline_sha256'])
    if channel['feature_pipeline_sha256']!=manifest['feature_pipeline_sha256']:
        raise ValueError('Recorded pipeline differs')
    raw=pipeline['raw_identity']; signature=pipeline['cpu_identity']['source_signature']
    # The historical signature records the detector digest, not its path.
    # Explicitly require the known project artifact, and report this distinction.
    detector=check_detector(detector,signature['detector_sha256'])
    if digest(checkpoint)!=manifest['checkpoint_sha256'] or digest(checkpoint)!=raw['checkpoint_sha256']:
        raise ValueError('Captured checkpoint differs')
    check_files(raw, RAW_FILES)
    check_files(signature, SOURCE_FILES)
    check_files(raw['patch_sha256'],{name:FORK/name for name in raw['patch_sha256']})
    if digest(ROOT/'modeling/lean_channel_feature_capture.py')!=pipeline['feature_observer_sha256'] or digest(FORK/'sam2/utils/misc.py')!=pipeline['image_loader_sha256']:
        raise ValueError('Captured feature observer or image loader differs')
    revision=subprocess.check_output(['git','-C',str(FORK),'rev-parse','HEAD'],text=True).strip()
    changed=subprocess.check_output(['git','-C',str(FORK),'diff','--name-only','HEAD'],text=True).splitlines()
    if revision!=raw['revision'] or revision!=signature['revision'] or sorted(changed)!=sorted(raw['patch_sha256']):
        raise ValueError('Token-window fork revision or patch set differs')
    if (pipeline['torch']!=str(torch.__version__) or signature['torch']!=str(torch.__version__) or
            signature['numpy']!=np.__version__ or signature['python']!=sys.version or
            signature['ultralytics']!=importlib.metadata.version('ultralytics') or
            pipeline['device']!='cpu' or pipeline['dtype']!='float32' or pipeline['resolution']!=1024):
        raise ValueError('Static CPU capture environment differs')
    split_file=split_preflight/'manifest.json';split_done=json.loads((split_preflight/'completion.json').read_text())
    if digest(split_file)!=manifest['split_sha256'] or not split_done['passed'] or split_done['manifest_sha256']!=digest(split_file):
        raise ValueError('Training split preflight differs')
    split=json.loads(split_file.read_text());sequence=channel['sequence'];meta=split['sequences'][sequence]
    source=Path(meta['source']); archive=json.loads((source/'records.json').read_text()); done=json.loads((source/'completion.json').read_text())
    if not done['passed'] or digest(source/'records.json')!=done['records_sha256'] or len(archive)!=1 or archive[0]['sequence']!=sequence:
        raise ValueError('Archived prompt source differs')
    folder=source/sequence;prompt_path=folder/'prompt.json';frames_path=folder/'frames.json'
    if (digest(prompt_path)!=archive[0]['prompt_sha256'] or digest(prompt_path)!=meta['prompt_sha256'] or
            digest(frames_path)!=archive[0]['frames_sha256'] or digest(frames_path)!=meta['frames_metadata_sha256']):
        raise ValueError('Archived prompt or frame metadata differs')
    prompt=json.loads(prompt_path.read_text());rows=json.loads(frames_path.read_text())
    if (prompt['first_detection_index']!=meta['first_prompt'] or prompt['refined_mask_sha256']!=meta['prompt_mask_sha256'] or
            digest(folder/'prompt_refined.png')!=prompt['refined_mask_sha256'] or
            digest(folder/'prompt_provisional.png')!=prompt['provisional_mask_sha256']):
        raise ValueError('Archived prompt masks or frame differ')
    videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
    frames=videos[sequence]
    if len(frames)!=len(rows) or len(frames)!=meta['frame_count']:
        raise ValueError('Archived frame count differs')
    for frame,row in zip(frames,rows,strict=True):
        if digest(frame.image_path)!=row['image_sha256']:
            raise ValueError('Archived original frame bytes differ')
    return {'passed':True,'manifest_sha256':manifest_sha,'feature_pipeline_sha256':manifest['feature_pipeline_sha256'],
            'revision':revision,'device':'cpu','dtype':'float32',
            'detector_path':str(detector),'detector_sha256':signature['detector_sha256'],
            'detector_path_captured_in_original_signature':False,
            'sequence':sequence,'prompt_source':str(source),'prompt_sha256':meta['prompt_sha256'],
            'prompt_frame':meta['first_prompt'],'archived_input_frames_checked':len(frames),
            'archived_view':'Original input frame bytes; no seq23 no-inset crop claim.',
            'scope':'Static source and archived inputs match only; live model weights, buffers, hooks, input transforms, prompts and memory traces have not been verified. This receipt does not authorize video inference.'}


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    for key in ('preflight','split-preflight','capture','checkpoint','detector'):p.add_argument('--'+key,type=Path,required=True)
    a=p.parse_args();print(json.dumps(verify(a.preflight,a.split_preflight,a.capture,a.checkpoint,a.detector),indent=2))
