#!/usr/bin/env python3
"""Label-free author comparator on the frozen additional automatic prompts.

This is an adapter around the existing native/SAMURAI/SAM2Long runners and
author predictors, not a new segmentation implementation. Each method must run
in a fresh process. Full videos require a matching per-sequence replay gate.
"""
import argparse
from contextlib import nullcontext
import json
import os
from pathlib import Path
import platform
import sys
import tempfile
from types import SimpleNamespace
import uuid

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import stage_frames
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.samurai_mps_kalman import portable_kalman_iou
from scripts.memory_token_io import digest, validate_output
from scripts.preflight_author_trainval_v1 import SEQUENCES as TRAINVAL_SEQUENCES, TRAIN_PREFLIGHT, trainval_inputs
from scripts.preflight_extended_boundary_video_v1 import extended_priority_inputs as extended_inputs
from scripts.preflight_remaining_boundary_video_v1 import extended_priority_inputs as remaining_inputs
from scripts.run_author_baseline import build_author_predictor, source_identity
from scripts.run_native_sam2_automatic_priority import REPO as NATIVE_REPO, native_source_identity
from scripts.run_samurai_automatic_priority import model_state_hash
from scripts.run_sam2long_automatic_priority import PATHWAY_CONFIG, audit_pathway_state
from scripts.source_group_label_guard_v1 import install_mask_open_guard

DATA = ROOT / 'data/polypgen_sequence.jsonl'
EXPERIMENT_ROOT = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment')
PROTOCOL = ROOT / 'protocols/EXTENDED_AUTHOR_COMPARATORS_20261115.md'
SEQUENCES = ('seq2', 'seq3', 'seq4', 'seq5', 'seq6', 'seq8', 'seq9', 'seq10', 'seq12',
             'seq13', 'seq14', 'seq15', 'seq16', 'seq17', 'seq18', 'seq19', 'seq21', 'seq22')
METHODS = ('native', 'samurai', 'sam2long')
REPRESENTATIVE = {'seq2': 'seq6', 'seq3': 'seq8', 'seq4': 'seq6', 'seq5': 'seq5',
                  'seq6': 'seq6', 'seq8': 'seq8', 'seq9': 'seq6', 'seq10': 'seq6',
                  'seq12': 'seq12', 'seq13': 'seq5', 'seq14': 'seq5', 'seq15': 'seq15',
                  'seq16': 'seq17', 'seq17': 'seq17', 'seq18': 'seq17',
                  'seq19': 'seq17', 'seq21': 'seq17', 'seq22': 'seq17'}
CHECKPOINT_SHA = 'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
DETECTOR_SHA = '858dfcffa13408382f612c7b57498908e11c0b28c8b24a948685806ad3c3c58e'
SOURCES = (
    'scripts/run_additional_author_comparator_v1.py',
    'scripts/verify_additional_author_comparator_gate_v1.py',
    'scripts/run_author_baseline.py', 'scripts/run_native_sam2_automatic_priority.py',
    'scripts/run_samurai_automatic_priority.py', 'scripts/run_sam2long_automatic_priority.py',
    'scripts/preflight_author_trainval_v1.py', 'scripts/preflight_source_group_capture_v1.py',
    'scripts/preflight_extended_boundary_video_v1.py', 'scripts/preflight_remaining_boundary_video_v1.py',
    'scripts/source_group_label_guard_v1.py', 'scripts/memory_token_io.py',
    'datasets/video.py', 'datasets/io.py', 'evaluation/davis2017/metrics.py',
    'evaluation/segmentation.py', 'training/tracking.py', 'modeling/seq23_viewport.py',
    'modeling/lean_mps_blocking_pointer_position.py',
    'modeling/samurai_mps_kalman.py', 'modeling/samurai_auto_prompt.py',
    'protocols/EXTENDED_AUTHOR_COMPARATORS_20261115.md',
)
CONFIGS = {
    'native': NATIVE_REPO / 'sam2/configs/sam2.1/sam2.1_hiera_b+.yaml',
    'samurai': ROOT / 'external/SAMURAI/sam2/sam2/configs/samurai/sam2.1_hiera_b+.yaml',
    'sam2long': ROOT / 'external/SAM2Long/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml',
}


def sources():
    return {name: digest(ROOT / name) for name in SOURCES}


def model_repo(method):
    if method == 'native':
        return NATIVE_REPO, native_source_identity()
    return source_identity(method)


def input_data(sequence, extended_preflight, remaining_preflight, checkpoint, detector):
    if sequence not in SEQUENCES:
        raise ValueError('Sequence not in frozen extension')
    if sequence in TRAINVAL_SEQUENCES:
        return trainval_inputs(sequence, checkpoint, detector)
    loader = remaining_inputs if sequence in ('seq6', 'seq10') else extended_inputs
    folder = remaining_preflight if sequence in ('seq6', 'seq10') else extended_preflight
    frames, prompt, _, view, mask, provenance = loader(sequence, folder, checkpoint, detector)
    start = prompt['first_detection_index']
    if view != {'kind': 'full_source_frame'} or not mask.any() or not 0 <= start < len(frames) - 1:
        raise ValueError('Unsupported prompt, view, or video coverage')
    return frames, start, mask, provenance


def identity(method, sequence, checkpoint, detector, extended_preflight, remaining_preflight, device):
    if method not in METHODS or sequence not in SEQUENCES:
        raise ValueError('Unknown comparator or sequence')
    if digest(checkpoint) != CHECKPOINT_SHA or digest(detector) != DETECTOR_SHA:
        raise ValueError('Author backbone or archived detector differs')
    repo, revision = model_repo(method)
    return {'method': method, 'sequence': sequence, 'repo': str(repo.resolve()), 'revision': revision,
            'checkpoint_sha256': digest(checkpoint), 'detector_sha256': digest(detector),
            'config_sha256': digest(CONFIGS[method]), 'data_sha256': digest(DATA),
            'extended_preflight_sha256': digest(Path(extended_preflight) / 'manifest.json'),
            'remaining_preflight_sha256': digest(Path(remaining_preflight) / 'manifest.json'),
            'train_preflight_sha256': digest(TRAIN_PREFLIGHT / 'manifest.json'),
            'sources': sources(), 'torch': str(torch.__version__), 'numpy': np.__version__,
            'python': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'device': device, 'dtype': 'float32', 'seed': 20261021,
            'hash_seed': os.environ.get('PYTHONHASHSEED'),
            'fallback': os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK'),
            'fast_math': os.environ.get('PYTORCH_MPS_FAST_MATH')}


def new_output(path):
    output = validate_output(path)
    if not output.is_relative_to(EXPERIMENT_ROOT.resolve()) or output.exists():
        raise ValueError('Output must be new and within the requested experiment folder')
    return output


def verify_gate(path, expected, representative_prompt, checkpoint, detector, extended_preflight, remaining_preflight):
    path = Path(path).resolve(strict=True)
    if not path.is_relative_to(EXPERIMENT_ROOT.resolve()):
        raise ValueError('Short replay gate lies outside experiment folder')
    gate = json.loads(path.read_text())
    representative = REPRESENTATIVE[expected['sequence']]
    gate_identity = dict(expected, sequence=representative)
    if (not gate.get('passed') or gate.get('identity') != gate_identity or gate.get('prompt') != representative_prompt
            or gate.get('verifier_sha256') != digest(ROOT / 'scripts/verify_additional_author_comparator_gate_v1.py')):
        raise ValueError('Matching audited short replay is required')
    receipt_paths = []
    for item in gate.get('receipt_paths', []):
        receipt_path = Path(item['path']).resolve(strict=True)
        if not receipt_path.is_relative_to(EXPERIMENT_ROOT.resolve()) or digest(receipt_path) != item['sha256']:
            raise ValueError('Short replay receipt changed')
        receipt_paths.append(receipt_path.parent)
        receipt = json.loads(receipt_path.read_text())
        if not receipt.get('passed') or receipt.get('full_video') is not False:
            raise ValueError('Short replay is incomplete')
        for record in receipt.get('records', []):
            if digest(receipt_path.parent / f"{record['frame']:06d}.png") != record['mask_sha256']:
                raise ValueError('Short replay prediction changed')
    if len(gate.get('receipt_paths', [])) != 3:
        raise ValueError('Short replay receipt coverage differs')
    # Reconstruct rather than trusting a self-asserted passed flag. This also
    # verifies process independence, frame identities and the repeat tolerance.
    from scripts.verify_additional_author_comparator_gate_v1 import audit
    regenerated = audit(*receipt_paths, checkpoint, detector, extended_preflight, remaining_preflight)
    if ({k: v for k, v in gate.items() if k != 'label_guard'} != regenerated
            or gate.get('label_guard', {}).get('open_attempts') != 0):
        raise ValueError('Reconstructed short replay gate differs')
    return gate


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--method', choices=METHODS, required=True)
    p.add_argument('--sequence', choices=SEQUENCES, required=True)
    p.add_argument('--device', choices=('cpu', 'mps'), required=True)
    p.add_argument('--limit', type=int, help='Number of staged post-prompt frames for a short gate')
    p.add_argument('--gate', type=Path)
    for name in ('checkpoint', 'detector', 'extended-preflight', 'remaining-preflight', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args()
    if (os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0' or
            os.environ.get('PYTORCH_MPS_FAST_MATH') != '0' or os.environ.get('PYTHONHASHSEED') != '0'):
        raise ValueError('Frozen numerical environment differs')
    if a.device == 'mps' and not torch.backends.mps.is_available():
        raise ValueError('MPS unavailable')
    if a.limit is not None and a.limit < 5:
        raise ValueError('A short engineering clip needs at least five staged frames')
    if a.limit is None and (a.device != 'mps' or a.gate is None):
        raise ValueError('Full video requires an MPS replay gate')
    guard = install_mask_open_guard(DATA)
    frames, start, prompt, provenance = input_data(a.sequence, a.extended_preflight, a.remaining_preflight,
                                                   a.checkpoint, a.detector)
    current = identity(a.method, a.sequence, a.checkpoint, a.detector,
                       a.extended_preflight, a.remaining_preflight, a.device)
    if a.limit is None:
        representative = REPRESENTATIVE[a.sequence]
        _, _, _, representative_provenance = input_data(representative, a.extended_preflight,
            a.remaining_preflight, a.checkpoint, a.detector)
        gate = verify_gate(a.gate, current, representative_provenance, a.checkpoint, a.detector,
                           a.extended_preflight, a.remaining_preflight)
    else:
        gate = None
    end = min(len(frames), start + a.limit) if a.limit is not None else len(frames)
    post = frames[start:end]
    if len(post) < 5:
        raise ValueError('Insufficient post-prompt frames for gate')
    out = new_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    torch.manual_seed(20261021)
    torch.use_deterministic_algorithms(True)
    if a.device == 'mps':
        torch.mps.manual_seed(20261021)
    repo, _ = model_repo(a.method)
    sys.path.insert(0, str(repo / 'sam2') if a.method == 'samurai' else str(repo))
    method = 'sam2' if a.method == 'native' else a.method
    model = build_author_predictor(SimpleNamespace(method=method, checkpoint=a.checkpoint, device=a.device), repo)
    import sam2
    if not Path(sam2.__file__).resolve().is_relative_to(repo.resolve()) or model.fill_hole_area != 0:
        raise ValueError('Wrong author package or postprocessing')
    if a.method == 'sam2long' and (model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1
                                  or not model.iou_prediction_use_sigmoid):
        raise ValueError('SAM2Long author configuration differs')
    before = model_state_hash(model)
    if gate is not None and gate.get('state_sha256') != before:
        raise ValueError('Loaded model state differs from the short gate')
    records = []
    for i in range(start):
        with Image.open(frames[i].image_path) as image:
            width, height = image.size
        path = out / f'{i:06d}.png'
        Image.fromarray(np.zeros((height, width), np.uint8)).save(path)
        records.append({'frame': i, 'source_frame_index': frames[i].frame_index,
                        'image_sha256': digest(frames[i].image_path), 'mask_sha256': digest(path), 'post_prompt': False})
    kalman_events = []
    with tempfile.TemporaryDirectory(prefix='frames_', dir=out) as tmp:
        staged = frames[:end] if a.method == 'sam2long' else post
        stage_frames(staged, Path(tmp))
        state = model.init_state(video_path=tmp, offload_video_to_cpu=True, offload_state_to_cpu=False)
        if state['num_frames'] != len(staged):
            raise ValueError('Staged coverage differs')
        if a.method == 'sam2long':
            state.update(PATHWAY_CONFIG)
        adapter = (blocking_mps_pointer_position(model) if a.method == 'native' and a.device == 'mps'
                   else portable_kalman_iou(model, a.device, kalman_events) if a.method == 'samurai'
                   else nullcontext())
        try:
            with torch.inference_mode(), adapter:
                local_start = start if a.method == 'sam2long' else 0
                if a.method == 'sam2long':
                    # Author SAM2Long differs from SAM2's add_new_mask return
                    # contract; validate object IDs from propagation instead.
                    model.add_new_mask(state, frame_idx=local_start, obj_id=1, mask=prompt)
                    ids, logits_list = model.propagate_in_video(state, start_frame_idx=start, reverse=False)
                    if ids != [1] or len(logits_list) != len(post):
                        raise ValueError('SAM2Long pathway coverage differs')
                    pathway = audit_pathway_state(state, start, end - 1)
                    outputs = ((i, ids, logits) for i, logits in enumerate(logits_list))
                else:
                    _, ids, _ = model.add_new_mask(state, frame_idx=local_start, obj_id=1, mask=prompt)
                    if ids != [1]:
                        raise ValueError('Unexpected object ID')
                    pathway = None
                    outputs = model.propagate_in_video(state, start_frame_idx=0, reverse=False)
                for local, ids, logits in outputs:
                    if local != len(records) - start or ids != [1] or logits.ndim != 4 or not bool(torch.isfinite(logits).all()):
                        raise ValueError('Frame order, identity or logits differ')
                    image_size = provenance['roi_xyxy'][3], provenance['roi_xyxy'][2]
                    mask = logits[0, 0].detach().float().cpu().numpy() > 0
                    if mask.shape != image_size:
                        raise ValueError('Mask geometry differs from original full frame')
                    frame = start + local
                    path = out / f'{frame:06d}.png'
                    Image.fromarray(mask.astype(np.uint8) * 255).save(path)
                    records.append({'frame': frame, 'source_frame_index': frames[frame].frame_index,
                                    'image_sha256': digest(frames[frame].image_path),
                                    'mask_sha256': digest(path), 'post_prompt': True})
        finally:
            model.reset_state(state)
    if len(records) != end or model_state_hash(model) != before or guard['open_attempts']:
        raise ValueError('Prediction coverage, model state, or label guard differs')
    receipt = {'passed': True, 'scope': 'Previously inspected automatic-mask comparator; not independent validation.',
               'method': a.method, 'sequence': a.sequence, 'prompt_frame': start, 'prompt': provenance,
               'full_video': a.limit is None, 'noncausal': a.method == 'sam2long',
               'device': a.device, 'identity': current, 'state_sha256': before,
               'process_id': os.getpid(), 'run_nonce': uuid.uuid4().hex,
               'deterministic': torch.are_deterministic_algorithms_enabled(),
               'gate_sha256': digest(a.gate) if a.gate else None, 'label_guard': dict(guard),
               'representative_gate_sequence': REPRESENTATIVE[a.sequence] if gate is not None else None,
               'pathway_audit': pathway, 'kalman_events': kalman_events, 'records': records}
    (out / 'receipt.json').write_text(json.dumps(receipt, indent=2) + '\n')
    print(json.dumps({'passed': True, 'method': a.method, 'sequence': a.sequence, 'frames': len(records)}))


if __name__ == '__main__':
    main()
