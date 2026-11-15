#!/usr/bin/env python3
"""Run released SASVi video loop with saved CMSA masks as a polyp overseer."""
import argparse
import hashlib
import importlib.util
import inspect
import json
import os
from pathlib import Path
import subprocess
import sys
import types

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from scripts.memory_token_io import digest  # noqa: E402

AUTHOR = ROOT / 'external/SASVi'
REVISION = 'ba87d22267da6fd05155829aee35da102c1a6567'
PATCHED_EVAL_SHA256 = 'ec66a305b9d39d4322ae157f203a04689d2628dcb57114d336dceeec53e9aa2d'
PATCHED_PREDICTOR_SHA256 = '2f551650d9cb943d0125a9c99dfaf0013e9c1320a627b8d206fff45a060de42f'
PATCHED_MISC_SHA256 = '672479f6e7251d45af739b85b4cd038f80d01c2767e2a800d8168dceb4548e27'
MANIFEST = ROOT / 'data/polypgen_sequence.jsonl'
CONFIG = ROOT / 'external/sam2_diagnostic_reference/sam2/configs/sam2.1/sam2.1_hiera_b+.yaml'


def load_author():
    """Load author inference without unrelated dataset/model training imports."""
    sys.path.insert(0, str(AUTHOR))
    sys.path.insert(0, str(AUTHOR / 'src/sam2'))
    for name, attrs in {
        'albumentations': (),
        'transformers': ('DetrImageProcessor', 'DetrForSegmentation', 'Mask2FormerForUniversalSegmentation', 'Mask2FormerImageProcessor'),
        'src.data': ('remap_labels', 'insert_component_masks'),
        'src.utils': ('process_detr_outputs', 'process_mask2former_outputs'),
        'src.model': ('get_model_instance_segmentation',),
        'sds_playground.datasets.cholecseg8k.cholecseg8k_visualisation': ('get_cholecseg8k_colormap',),
        'sds_playground.datasets.cadisv2.cadisv2_visualisation': ('get_cadis_colormap',),
        'sds_playground.datasets.cataract1k.cataract1ksegm_visualisation': ('get_cataract1k_colormap',),
    }.items():
        module = types.ModuleType(name)
        for attr in attrs:
            setattr(module, attr, None)
        sys.modules[name] = module
    path = AUTHOR / 'src/sam2/eval_sasvi.py'
    spec = importlib.util.spec_from_file_location('sasvi_author_eval', path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--cmsa-run', type=Path, required=True)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--sequence', required=True)
    p.add_argument('--max-frames', type=int)
    p.add_argument('--device', choices=('cpu', 'mps'), default='cpu')
    args = p.parse_args()
    if args.output.exists():
        raise FileExistsError(args.output)
    if subprocess.check_output(['git', '-C', str(AUTHOR), 'rev-parse', 'HEAD'], text=True).strip() != REVISION:
        raise ValueError('Unexpected SASVi revision')
    subprocess.run(['git', '-C', str(AUTHOR), 'diff', '--quiet', 'HEAD', '--', '.',
                    ':(exclude)src/sam2/eval_sasvi.py', ':(exclude)src/sam2/sam2/sam2_video_predictor.py',
                    ':(exclude)src/sam2/sam2/utils/misc.py'], check=True)
    if digest(AUTHOR / 'src/sam2/eval_sasvi.py') != PATCHED_EVAL_SHA256:
        raise ValueError('Unexpected SASVi boundary fix')
    if digest(AUTHOR / 'src/sam2/sam2/sam2_video_predictor.py') != PATCHED_PREDICTOR_SHA256:
        raise ValueError('Unexpected SASVi predictor fix')
    if digest(AUTHOR / 'src/sam2/sam2/utils/misc.py') != PATCHED_MISC_SHA256:
        raise ValueError('Unexpected SASVi video-loader fix')
    data = {**load_manifest(MANIFEST, 'dev'), **load_manifest(MANIFEST, 'test')}
    frames = data[args.sequence][:args.max_frames]
    if not frames:
        raise ValueError('No frames')
    cmsa_setup = json.loads((args.cmsa_run / 'setup.json').read_text())
    if cmsa_setup['manifest_sha256'] != digest(MANIFEST):
        raise ValueError('CMSA manifest differs from current frame ordering')
    # The original full run predates the normalization field; its signed
    # setup explicitly records the author normalization used by that run.
    original_released = (cmsa_setup.get('runner_sha256') == '871b7548ad43328e8961d4aff698a0896b661be6502181a2428db63f66b20a55'
                         and 'Author normalization retained' in cmsa_setup.get('note', ''))
    if cmsa_setup.get('normalization') != 'released' and not original_released:
        raise ValueError('Expected released CMSA preprocessing')
    records = json.loads((args.cmsa_run / 'frames.json').read_text())
    expected = {(r['sequence'], r['frame']): r for r in records}
    masks = []
    for i, frame in enumerate(frames):
        relative = frame.image_path.resolve().relative_to((ROOT / 'data').resolve()).with_suffix('.png')
        path = args.cmsa_run / 'predictions' / relative
        if digest(path) != expected[(args.sequence, i)]['prediction_sha256']:
            raise ValueError(f'CMSA prediction provenance mismatch: {i}')
        masks.append(path)
    author = load_author()
    from hydra import initialize_config_dir
    from hydra.core.global_hydra import GlobalHydra
    GlobalHydra.instance().clear()
    torch.set_num_threads(2)
    with initialize_config_dir(config_dir=str(CONFIG.parent), version_base=None):
        predictor = author.build_sam2_video_predictor(CONFIG.name, str(args.checkpoint), device=args.device, apply_postprocessing=False)
    args.output.mkdir(parents=True)
    video_dir = args.output / 'input' / args.sequence
    video_dir.mkdir(parents=True)
    for i, frame in enumerate(frames):
        (video_dir / f'{i:06d}.jpg').symlink_to(frame.image_path.resolve())
    original_init_state = predictor.init_state
    def checked_init_state(*init_args, **init_kwargs):
        state = original_init_state(*init_args, **init_kwargs)
        paths = getattr(state['images'], 'img_paths', None)
        expected_names = [f'{i:06d}.jpg' for i in range(len(frames))]
        if paths is None or [Path(path).name for path in paths] != expected_names:
            raise ValueError('Video predictor frame order differs from overseer frame order')
        return state
    predictor.init_state = checked_init_state
    class CMSAOverseer:
        def get_prediction(self, paths, reshape_size):
            idx = int(Path(paths[0]).stem)
            mask = np.asarray(Image.open(masks[idx]).convert('L')) >= 128
            if mask.shape != tuple(reshape_size):
                raise ValueError('Overseer mask dimensions differ')
            return {1: mask} if mask.any() else {}
    def save_masks_to_dir(output_mask_dir, video_name, frame_name, per_obj_output_mask,
                          height, width, *unused, **kwargs):
        raw = per_obj_output_mask.get(1)
        pred = np.zeros((height, width), bool) if raw is None else np.asarray(raw).reshape(height, width)
        dst = Path(output_mask_dir) / video_name / f'{frame_name}.png'
        dst.parent.mkdir(parents=True, exist_ok=True)
        Image.fromarray(pred.astype(np.uint8) * 255).save(dst)
    author.save_masks_to_dir = save_masks_to_dir
    original_points = author.get_points_from_mask
    point_events = []
    def record_points(mask, label_ids=None, num_points=20):
        result = original_points(mask, label_ids=label_ids, num_points=num_points)
        point_events.append({'labels': list(label_ids), 'points': result,
            'mask_sha256': {str(key): hashlib.sha256(np.asarray(value, dtype=np.uint8).tobytes()).hexdigest()
                            for key, value in mask.items()}})
        return result
    author.get_points_from_mask = record_points
    setup = {'method': 'author SASVi inference with adapted saved CMSA binary mask overseer',
             'author_revision': REVISION, 'author_source_sha256': digest(AUTHOR / 'src/sam2/eval_sasvi.py'),
             'predictor_sha256': digest(AUTHOR / 'src/sam2/sam2/sam2_video_predictor.py'),
             'loader_sha256': digest(AUTHOR / 'src/sam2/sam2/utils/misc.py'),
             'sam2_checkpoint_sha256': digest(args.checkpoint), 'sam2_config_sha256': digest(CONFIG),
             'cmsa_setup_sha256': digest(args.cmsa_run / 'setup.json'), 'cmsa_frames_sha256': digest(args.cmsa_run / 'frames.json'),
             'manifest_sha256': digest(MANIFEST), 'runner_sha256': digest(Path(__file__)),
             'sequence': args.sequence, 'frames': len(frames), 'device': f'{args.device} fp32',
             'adaptation': 'CMSA >=128 as single polyp class; FP32; full resolution output; B+ config; SASVi loop with full 3-frame confirmation, small-mask prompts and numeric frame sorting',
             'limitations': 'CMSA uses multi-frame clips including future images; SASVi looks ahead as released. Retrospective evaluation, not causal streaming.'}
    (args.output / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
    with torch.inference_mode():
        inspect.unwrap(author.sasvi_inference)(predictor, str(args.output / 'input'), str(args.output / 'predictions'), args.sequence,
            'CMSA', None, CMSAOverseer(), None, 2, [0], False, [], 'POLYPGEN', save_binary_mask=False)
    (args.output / 'prompt_events.json').write_text(json.dumps(point_events, indent=2) + '\n')
    rows = []
    for i, frame in enumerate(frames):
        path = args.output / 'predictions' / args.sequence / f'{i:06d}.png'
        pred = np.asarray(Image.open(path)) > 0
        truth = load_label_mask(frame.mask_path, [1]) == 1
        denom = int(pred.sum() + truth.sum())
        rows.append({'index': i, 'present': bool(truth.any()), 'dice': 2 * int(np.count_nonzero(pred & truth)) / denom if denom else 1.,
                     'false_alarm': bool(pred.any()) if not truth.any() else None, 'mask_sha256': digest(path),
                     'cmsa_mask_sha256': digest(masks[i]), 'image_sha256': digest(frame.image_path)})
    (args.output / 'records.json').write_text(json.dumps(rows, indent=2) + '\n')
    (args.output / 'completion.json').write_text(json.dumps({'passed': True, 'records_sha256': digest(args.output / 'records.json'), 'frames': len(rows)}, indent=2) + '\n')
    print('Completed', args.sequence, len(rows), 'frames', flush=True)


if __name__ == '__main__':
    main()
