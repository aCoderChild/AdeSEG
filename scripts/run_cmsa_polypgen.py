#!/usr/bin/env python3
"""Run the pinned CMSA-Net author test loop on manifest-ordered PolypGen videos."""
import argparse
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import subprocess
import sys
from unittest.mock import patch

import numpy as np
from PIL import Image
import torch
from torch.utils.data import DataLoader

ROOT = Path(__file__).resolve().parents[1]
AUTHOR = ROOT / 'external/CMSA-Net/cmsa_project'
REVISION = '4e4602e93211b5d0006a8a7415e8477de413b79d'
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest, assert_disjoint_videos
from evaluation.segmentation import calculate_scores
from datasets.io import load_label_mask


def sha(path):
    h = hashlib.sha256()
    with open(path, 'rb') as f:
        for block in iter(lambda: f.read(1024 * 1024), b''):
            h.update(block)
    return h.hexdigest()


def clips(frames, length=4):
    if len(frames) < length:
        raise ValueError('Author test requires at least four frames per sequence')
    pairs = [(str(f.image_path.resolve()), str(f.mask_path.resolve())) for f in frames]
    return [[pairs[0], pairs[1], *pairs[i:i + length]] for i in range(len(pairs) - length + 1)]


def load_author():
    revision = subprocess.check_output(['git', '-C', str(AUTHOR.parent), 'rev-parse', 'HEAD'], text=True).strip()
    if revision != REVISION:
        raise RuntimeError(f'Unexpected CMSA-Net revision: {revision}')
    subprocess.run(['git', '-C', str(AUTHOR.parent), 'diff', '--exit-code', 'HEAD', '--', 'cmsa_project'], check=True, stdout=subprocess.DEVNULL)
    sys.path.insert(0, str(AUTHOR))
    sys.path.insert(0, str(AUTHOR / 'scripts'))
    # The upstream config parses argv at import time.
    with patch.object(sys, 'argv', ['cmsa-author']):
        config_module = importlib.import_module('config')
        sys.modules['scripts.config'] = config_module
        spec = importlib.util.spec_from_file_location('cmsa_author_test', AUTHOR / 'scripts/my_test.py')
        module = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(module)
    return module


class ChannelNormalizeVideo:
    """Apply the released statistics to each channel of a CHW tensor."""

    def __init__(self, mean, std):
        from torchvision.transforms import Normalize
        self.normalize = Normalize(mean=[float(x) for x in mean], std=[float(x) for x in std])

    def __call__(self, images, labels=None):
        images = [self.normalize(image) for image in images]
        return (images, labels) if labels is not None else images


def author_dataset(videos, normalization='released'):
    from lib.dataloader.dataloader import CMSAVideoDataset
    from lib.dataloader.preprocess import Compose_imglabel, Resize_video, toTensor_video, Normalize_video
    stats = torch.load(AUTHOR / 'lib/dataloader/statistics.pth', map_location='cpu', weights_only=False)
    dataset = CMSAVideoDataset.__new__(CMSAVideoDataset)
    dataset.video_train_list = [clip for frames in videos.values() for clip in clips(frames)]
    dataset.img_only = False
    if normalization == 'channel':
        normalize = ChannelNormalizeVideo(stats['mean'], stats['std'])
    elif normalization == 'released':
        normalize = Normalize_video(stats['mean'], stats['std'])
    else:
        raise ValueError(f'Unknown normalization: {normalization}')
    dataset.img_label_transform = Compose_imglabel([
        Resize_video(352, 352), toTensor_video(), normalize
    ])
    return dataset


def run(args):
    if args.output.exists():
        raise FileExistsError(f'Output directory already exists: {args.output}')
    if not args.checkpoint.is_file():
        raise FileNotFoundError(args.checkpoint)
    data_root = (ROOT / 'data').resolve()
    splits = [load_manifest(args.manifest, split) for split in ('dev', 'test')]
    assert_disjoint_videos(*splits)
    videos = {**splits[0], **splits[1]}
    if not args.sequence:
        args.sequence = [f'seq{i}' for i in range(2, 24) if i != 7]
    if args.sequence:
        missing = set(args.sequence) - videos.keys()
        if missing:
            raise ValueError(f'Unknown sequences: {sorted(missing)}')
        videos = {name: videos[name] for name in args.sequence}
    for frames in videos.values():
        clips(frames)
        for f in frames:
            f.image_path.resolve().relative_to(data_root)
    author = load_author()
    dataset = author_dataset(videos, args.normalization)
    # Construct the exact author model and enforce a fully matching checkpoint.
    model = author.CMSANet()
    state = torch.load(args.checkpoint, map_location='cpu', weights_only=True)
    model.load_state_dict(state, strict=True)
    model = model.to(args.device).eval()
    test = author.VPSTest_WithTestingTimeReferUpdate_2Ref.__new__(author.VPSTest_WithTestingTimeReferUpdate_2Ref)
    test.data_root = str(data_root)
    test.save_dir = str(args.output / 'predictions')
    test.test_dataset = ['PolypGen']
    test.dataloader = {'PolypGen': DataLoader(dataset, batch_size=1, shuffle=False, num_workers=0)}
    test.model = model
    args.output.mkdir(parents=True)
    setup = {'author_revision': REVISION, 'author_test_sha256': sha(AUTHOR / 'scripts/my_test.py'),
             'checkpoint_sha256': sha(args.checkpoint), 'checkpoint': str(args.checkpoint),
             'manifest_sha256': sha(args.manifest), 'runner_sha256': sha(Path(__file__)), 'torch': torch.__version__, 'device': args.device,
             'sequences': list(videos), 'clip_length': 4, 'probability_threshold': 128,
             'normalization': args.normalization,
             'note': 'Released overlapping clip and first-save policy. Frame 0 can use frame 1 as an anchor and is excluded from matched post-prompt evaluation. Channel normalization is an inference-only change to a pretrained checkpoint.'}
    (args.output / 'setup.json').write_text(json.dumps(setup, indent=2) + '\n')
    # The upstream loop hardcodes Tensor.cuda(); adapt only device placement.
    def move(tensor, *unused, **kwargs):
        return tensor.to(args.device)
    with patch.object(torch.Tensor, 'cuda', move):
        test.test()
    rows, summary = [], {}
    for name, frames in videos.items():
        sequence_rows = []
        for i, frame in enumerate(frames):
            relative = frame.image_path.resolve().relative_to(data_root).with_suffix('.png')
            pred_path = args.output / 'predictions' / relative
            if not pred_path.is_file():
                raise RuntimeError(f'Missing author prediction: {pred_path}')
            gt = load_label_mask(frame.mask_path, [1]) == 1
            pred = np.asarray(Image.open(pred_path).convert('L')) >= 128
            if pred.shape != gt.shape:
                raise RuntimeError(f'Prediction shape mismatch: {pred_path}')
            scores = calculate_scores(pred, gt)
            row = {'sequence': name, 'frame': i, 'present': bool(gt.any()), 'dice': scores['dice'], 'iou': scores['iou'], 'prediction_sha256': sha(pred_path)}
            rows.append(row)
            sequence_rows.append(row)
        # Native SAM2's first GT-present prompt and all following frames.
        prompt = next(i for i, row in enumerate(sequence_rows) if row['present'])
        matched = sequence_rows[prompt + 1:]
        present = [row['dice'] for row in matched if row['present']]
        empty = [row for row in matched if not row['present']]
        summary[name] = {'prompt_index': prompt, 'matched_targets': len(matched),
                         'present_targets': len(present), 'mean_present_dice': float(np.mean(present)) if present else None,
                         'gt_empty_targets': len(empty), 'false_alarms': sum(row['dice'] == 0 for row in empty)}
    (args.output / 'frames.json').write_text(json.dumps(rows, indent=2) + '\n')
    means = [v['mean_present_dice'] for v in summary.values() if v['mean_present_dice'] is not None]
    result = {'per_sequence': summary, 'equal_sequence_mean_present_dice': float(np.mean(means)),
              'comparison_scope': 'Same post-prompt frame indices as native; CMSA is unprompted. Frame 0 may use frame 1 as an anchor and is excluded.'}
    (args.output / 'results.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--checkpoint', type=Path, required=True)
    p.add_argument('--output', type=Path, required=True)
    p.add_argument('--manifest', type=Path, default=ROOT / 'data/polypgen_sequence.jsonl')
    p.add_argument('--device', choices=('cpu', 'mps', 'cuda'), default='cpu')
    p.add_argument('--normalization', choices=('channel', 'released'), default='released',
                   help='Use released column-wise preprocessing (default) or corrected channel-wise normalization.')
    p.add_argument('--sequence', action='append')
    args = p.parse_args()
    run(args)


if __name__ == '__main__':
    main()
