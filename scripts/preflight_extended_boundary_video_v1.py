#!/usr/bin/env python3
"""Freeze image-only provenance for the seven-patient exploratory extension.

No annotation-mask files are opened and no archived outcome fields are parsed.
The raw archived frame metadata is hash-checked and searched only for source
indices and image hashes; its path/hash are retained for later scoring audit.
"""
import argparse
import hashlib
import json
import math
from pathlib import Path
import re
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.samurai_auto_prompt import prepare_prompt
from scripts.memory_token_io import digest, validate_output
from scripts.source_group_label_guard_v1 import install_mask_open_guard

BASE = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment')
ARCHIVE = BASE / 'lean_auto_multiscale_v1'
LOCATIONS = ARCHIVE / 'cohort_combined_attempt01/locations.json'
TRAIN_PREFLIGHT = BASE / 'contrast_refinement_v1/source_group_capture_preflight_attempt03'
TRAIN_CONFIG = ROOT / 'protocols/source_group_capture_v1.json'
DATA = ROOT / 'data/polypgen_sequence.jsonl'
PROTOCOL = ROOT / 'protocols/EXTENDED_BOUNDARY_FACTORIAL_V1.md'
SEQUENCES = ('seq8', 'seq16', 'seq17', 'seq18', 'seq19', 'seq21', 'seq22')
EXPECTED_TRAIN = {'seq2', 'seq3', 'seq4', 'seq5', 'seq12', 'seq13', 'seq14'}
EXPECTED_VALIDATION = {'seq9', 'seq15'}
CHECKPOINT_SHA = 'a2345aede8715ab1d5d31b4a509fb160c5a4af1970f199d9054ccfb746c004c5'
DETECTOR_SHA = '858dfcffa13408382f612c7b57498908e11c0b28c8b24a948685806ad3c3c58e'
SOURCE_FILES = ('scripts/preflight_extended_boundary_video_v1.py', 'scripts/memory_token_io.py',
                'scripts/source_group_label_guard_v1.py', 'datasets/video.py',
                'modeling/samurai_auto_prompt.py', 'modeling/seq23_viewport.py',
                'protocols/EXTENDED_BOUNDARY_FACTORIAL_V1.md', 'protocols/source_group_capture_v1.json')


def read(path):
    return json.loads(Path(path).read_text())


def source_hashes():
    return {name:digest(ROOT/name) for name in SOURCE_FILES}


def register_cross_sequence_image(seen, key, sequence):
    if key in seen and seen[key] != sequence:
        raise ValueError(f'{sequence}: exact or decoded image duplicates a different selected sequence')
    seen[key]=sequence


def image_metadata_only(path):
    """Extract and validate only image indices and hashes from archived JSON.

    This intentionally does not deserialize any label-derived outcome field.
    Inputs are matched by exact complete field-line syntax; unexpected formats
    fail closed instead of accepting guessed row boundaries.
    """
    index = re.compile(rb'^\s*"source_frame_index"\s*:\s*(\d+)\s*,?\s*$')
    image = re.compile(rb'^\s*"image_sha256"\s*:\s*"([a-f0-9]{64})"\s*,?\s*$')
    indices, hashes = [], []
    with Path(path).open('rb') as stream:
        for line in stream:
            if b'"source_frame_index"' in line:
                m = index.fullmatch(line.rstrip(b'\r\n'))
                if not m:
                    raise ValueError('Unexpected source frame index syntax')
                indices.append(int(m.group(1)))
            elif b'"image_sha256"' in line:
                m = image.fullmatch(line.rstrip(b'\r\n'))
                if not m:
                    raise ValueError('Unexpected source image hash syntax')
                hashes.append(m.group(1).decode('ascii'))
    if not indices or len(indices) != len(hashes):
        raise ValueError('Archived image-only metadata is incomplete')
    return indices, hashes


def validate_prompt(prompt, mask_path, image_size):
    """Check archived automatic prompt geometry without opening label masks."""
    width, height = image_size
    start = prompt.get('first_detection_index')
    box, points = prompt.get('box'), prompt.get('points')
    if type(start) is not int or not isinstance(box, list) or len(box) != 4:
        raise ValueError('Archived automatic prompt is incomplete')
    if not isinstance(points, list) or len(points) != 2 or any(not isinstance(p, list) or len(p) != 2 for p in points):
        raise ValueError('Archived automatic point geometry differs')
    if not all(isinstance(v, (int, float)) and math.isfinite(v) for v in [*box, *points[0], *points[1]]):
        raise ValueError('Automatic prompt coordinates are nonfinite')
    if not (0 <= box[0] < box[2] <= width and 0 <= box[1] < box[3] <= height and
            all(0 <= p[0] < width and 0 <= p[1] < height for p in points)):
        raise ValueError('Automatic prompt coordinates lie outside the frame')
    with Image.open(mask_path) as image:
        mask = np.asarray(image.convert('L'))
    result, viewport = prepare_prompt(mask, 'not_seq23', (height, width))
    if not result.any() or tuple(viewport) != (0, 0, width, height):
        raise ValueError('Automatic prompt mask is empty or viewport differs')
    return result


def checked_training_preflight(path):
    path = Path(path)
    p, done = path/'manifest.json', read(path/'completion.json')
    if not done.get('passed') or done.get('manifest_sha256') != digest(p):
        raise ValueError('Frozen source-group preflight differs')
    data = read(p)
    config=read(TRAIN_CONFIG)
    if (data.get('config_sha256') != digest(TRAIN_CONFIG) or
            not set(SEQUENCES).issubset(config.get('duplicate_screen_sequences', [])) or
            not data.get('source_sha256') or
            any(digest(ROOT/name) != value for name,value in data['source_sha256'].items())):
        raise ValueError('Training preflight config, duplicate-screen scope or source differs')
    train = {s for s, row in data['sequences'].items() if row['split'] == 'train'}
    val = {s for s, row in data['sequences'].items() if row['split'] == 'validation'}
    duplicates = data['duplicates']
    if (train != EXPECTED_TRAIN or val != EXPECTED_VALIDATION or set(SEQUENCES) & (train | val) or
            duplicates.get('cross_group_exact_byte') != 0 or duplicates.get('cross_group_decoded_rgb') != 0):
        raise ValueError('Training/validation split or duplicate preflight differs')
    return data


def build_manifest(checkpoint, detector, training_preflight=TRAIN_PREFLIGHT):
    checkpoint, detector = Path(checkpoint), Path(detector)
    if digest(checkpoint) != CHECKPOINT_SHA or digest(detector) != DETECTOR_SHA:
        raise ValueError('Native checkpoint or archived detector differs')
    train = checked_training_preflight(training_preflight)
    if train['manifest_sha256'] != digest(DATA):
        raise ValueError('Current data manifest differs from frozen training preflight')
    locations = read(LOCATIONS)
    videos = {**load_manifest(DATA, 'dev'), **load_manifest(DATA, 'test')}
    out, byte_owners, rgb_owners = {}, {}, {}
    for seq in SEQUENCES:
        source = Path(locations[seq]).resolve()
        if not source.is_relative_to(ARCHIVE.resolve()):
            raise ValueError('Archived automatic source outside expected cohort')
        folder = source/seq
        records, done, setup = read(source/'records.json'), read(source/'completion.json'), read(source/'setup.json')
        if (not done.get('passed') or done.get('records_sha256') != digest(source/'records.json') or
                len(records) != 1 or records[0].get('sequence') != seq or setup.get('sequences') != [seq]):
            raise ValueError(f'{seq}: original automatic cohort receipt differs')
        signature = setup['signature']
        if (signature.get('checkpoint_sha256') != CHECKPOINT_SHA or signature.get('detector_sha256') != DETECTOR_SHA or
                signature.get('manifest_sha256') != digest(DATA) or
                setup.get('initialization') != 'first chronological YOLO detection; no GT prompt'):
            raise ValueError(f'{seq}: original detector/model lineage differs')
        rec = records[0]
        frame_file, prompt_file, mask_file = folder/'frames.json', folder/'prompt.json', folder/'prompt_refined.png'
        if digest(frame_file) != rec['frames_sha256'] or digest(prompt_file) != rec['prompt_sha256']:
            raise ValueError(f'{seq}: archived source metadata changed')
        prompt = read(prompt_file)
        if (prompt.get('refined_mask_sha256') != digest(mask_file) or
                prompt.get('first_detection_index') != rec.get('first_detection_index') or
                rec.get('point_status') != 'available' or rec.get('lean_mask_status') != 'automatic_mask'):
            raise ValueError(f'{seq}: archived automatic prompt identity differs')
        frames = videos[seq]
        indices, hashes = image_metadata_only(frame_file)
        if len(frames) != len(indices) or len(frames) != rec['frames']:
            raise ValueError(f'{seq}: source frame count differs')
        rows, size = [], None
        for i, (frame, old_index, old_sha) in enumerate(zip(frames, indices, hashes, strict=True)):
            if frame.frame_index != old_index or digest(frame.image_path) != old_sha:
                raise ValueError(f'{seq}: source image/ordinal differs at {i}')
            with Image.open(frame.image_path) as image:
                actual_size = image.size
                rgb=image.convert('RGB')
                rgb_sha=hashlib.sha256(str(rgb.size).encode('ascii')+rgb.tobytes()).hexdigest()
            if size is None: size = actual_size
            if actual_size != size:
                raise ValueError(f'{seq}: source geometry is not constant')
            for seen, key in ((byte_owners,old_sha),(rgb_owners,rgb_sha)):
                register_cross_sequence_image(seen,key,seq)
            rows.append({'frame': i, 'source_frame_index': frame.frame_index,
                         'image_path': str(frame.image_path.resolve()), 'image_sha256': old_sha})
        start = prompt['first_detection_index']
        if not 0 <= start < len(frames)-1:
            raise ValueError(f'{seq}: prompt frame is invalid')
        validate_prompt(prompt, mask_file, size)
        out[seq] = {'frame_count': len(frames), 'frames': rows,
                    'archive_frames_path': str(frame_file), 'archive_frames_sha256': digest(frame_file),
                    'archive_records_path': str(source/'records.json'), 'archive_records_sha256': digest(source/'records.json'),
                    'archive_completion_path': str(source/'completion.json'), 'archive_completion_sha256': digest(source/'completion.json'),
                    'archive_setup_path': str(source/'setup.json'), 'archive_setup_sha256': digest(source/'setup.json'),
                    'prompt_metadata_path': str(prompt_file), 'prompt_metadata_sha256': digest(prompt_file),
                    'prompt_mask_path': str(mask_file), 'prompt_mask_sha256': digest(mask_file),
                    'first_detection_index': start, 'box': prompt['box'], 'points': prompt['points'],
                    'image_size_wh': list(size), 'view': {'kind': 'full_source_frame'}}
    return {'passed': True,
            'scope': 'Previously inspected, patient-disjoint by user assertion; not a blinded evaluation.',
            'patient_mapping': 'User states each numbered sequence represents a different patient; no patient-ID table independently verified.',
            'selection': list(SEQUENCES), 'train_sequences': sorted(EXPECTED_TRAIN),
            'validation_sequences': sorted(EXPECTED_VALIDATION), 'data_manifest_sha256': digest(DATA),
            'locations_sha256': digest(LOCATIONS), 'training_preflight_path': str(Path(training_preflight).resolve()),
            'training_preflight_sha256': digest(Path(training_preflight)/'manifest.json'),
            'checkpoint_sha256': CHECKPOINT_SHA, 'detector_sha256': DETECTOR_SHA,
            'preflight_source_sha256': digest(Path(__file__)), 'protocol_sha256': digest(PROTOCOL), 'source_sha256':source_hashes(),
            'sequences': out,
            'within_extension_duplicates': {'cross_sequence_exact_byte':0,'cross_sequence_decoded_rgb':0},
            'limitation': 'Exact/decoded duplicate preflight does not exclude near duplicates, metadata errors or detector training exposure.'}


def load_extended_preflight(folder, checkpoint, detector):
    folder = Path(folder)
    manifest, completion = folder/'manifest.json', read(folder/'completion.json')
    if not completion.get('passed') or completion.get('manifest_sha256') != digest(manifest):
        raise ValueError('Extended preflight receipt differs')
    data = read(manifest)
    if (not data.get('passed') or data.get('selection') != list(SEQUENCES) or set(data.get('sequences', {})) != set(SEQUENCES) or
            data.get('preflight_source_sha256') != digest(Path(__file__)) or data.get('protocol_sha256') != digest(PROTOCOL) or
            data.get('source_sha256') != source_hashes() or
            data.get('within_extension_duplicates') != {'cross_sequence_exact_byte':0,'cross_sequence_decoded_rgb':0} or
            data.get('data_manifest_sha256') != digest(DATA) or data.get('checkpoint_sha256') != digest(checkpoint) or
            data.get('detector_sha256') != digest(detector)):
        raise ValueError('Extended preflight identity differs')
    training_path=Path(data['training_preflight_path'])
    checked_training_preflight(training_path)
    if data['training_preflight_sha256'] != digest(training_path/'manifest.json'):
        raise ValueError('Training preflight changed after extension freeze')
    return data


def extended_priority_inputs(sequence, preflight, checkpoint, detector):
    """Load sanitized, frozen input records; no archived outcome metadata read."""
    data = load_extended_preflight(preflight, checkpoint, detector)
    if sequence not in SEQUENCES:
        raise ValueError('Sequence not in frozen extension')
    item = data['sequences'][sequence]
    videos = {**load_manifest(DATA, 'dev'), **load_manifest(DATA, 'test')}
    frames = videos[sequence]
    if len(frames) != item['frame_count'] or len(item['frames']) != len(frames):
        raise ValueError('Frozen frame count differs')
    for i, (frame, row) in enumerate(zip(frames, item['frames'], strict=True)):
        if (row['frame'] != i or row['source_frame_index'] != frame.frame_index or
                row['image_path'] != str(frame.image_path.resolve()) or row['image_sha256'] != digest(frame.image_path)):
            raise ValueError('Frozen image identity differs')
    prompt_file, mask_file = Path(item['prompt_metadata_path']), Path(item['prompt_mask_path'])
    if digest(prompt_file) != item['prompt_metadata_sha256'] or digest(mask_file) != item['prompt_mask_sha256']:
        raise ValueError('Frozen automatic prompt differs')
    prompt = read(prompt_file)
    if (prompt.get('first_detection_index') != item['first_detection_index'] or prompt.get('box') != item['box'] or
            prompt.get('points') != item['points'] or prompt.get('refined_mask_sha256') != item['prompt_mask_sha256']):
        raise ValueError('Frozen automatic prompt metadata differs')
    size = tuple(item['image_size_wh'])
    prompt_mask = validate_prompt(prompt, mask_file, size)
    provenance = {'prompt_file': str(mask_file), 'prompt_mask_sha256': item['prompt_mask_sha256'],
                  'prompt_metadata_sha256': item['prompt_metadata_sha256'],
                  'reference_receipt_sha256': item['archive_completion_sha256'],
                  'preflight_sha256': digest(Path(preflight)/'manifest.json'),
                  'roi_xyxy': [0, 0, *size]}
    return frames, prompt, item['prompt_metadata_sha256'], item['view'], prompt_mask, provenance


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--checkpoint', type=Path, required=True)
    parser.add_argument('--detector', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True)
    args = parser.parse_args()
    guard = install_mask_open_guard(DATA)
    result = build_manifest(args.checkpoint, args.detector)
    if guard['open_attempts'] != 0:
        raise ValueError('Annotation-mask open was attempted')
    result['label_guard'] = dict(guard)
    out = validate_output(args.output)
    out.mkdir(parents=True, exist_ok=False)
    path = out/'manifest.json'
    path.write_text(json.dumps(result, indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed': True, 'manifest_sha256': digest(path)}, indent=2)+'\n')
    print(json.dumps({'passed': True, 'sequences': list(SEQUENCES), 'frames': sum(x['frame_count'] for x in result['sequences'].values()), 'label_open_attempts': guard['open_attempts']}))


if __name__ == '__main__':
    main()
