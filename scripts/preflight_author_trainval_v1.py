"""Load image-only archived automatic prompts for refiner train/val videos.

These sequences were used to train or select the refiner: no video results
from this group may be described as independent or held out. The existing
source-group preflight is checked, not modified, and annotation masks are not
opened by this loader.
"""
import json
from pathlib import Path

import numpy as np
from PIL import Image

from datasets.video import load_manifest
from modeling.samurai_auto_prompt import prepare_prompt
from scripts.memory_token_io import digest
from scripts.preflight_extended_boundary_video_v1 import image_metadata_only

ROOT = Path(__file__).resolve().parents[1]
DATA = ROOT/'data/polypgen_sequence.jsonl'
CONFIG = ROOT/'protocols/source_group_capture_v1.json'
TRAIN_PREFLIGHT = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/contrast_refinement_v1/source_group_capture_preflight_attempt03')
ARCHIVE = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment/lean_auto_multiscale_v1')
SEQUENCES = ('seq2', 'seq3', 'seq4', 'seq5', 'seq9', 'seq12', 'seq13', 'seq14', 'seq15')
TRAIN = {'seq2', 'seq3', 'seq4', 'seq5', 'seq12', 'seq13', 'seq14'}
VALIDATION = {'seq9', 'seq15'}


def read(path):
    return json.loads(Path(path).read_text())


def trainval_inputs(sequence, checkpoint, detector, preflight=TRAIN_PREFLIGHT):
    if sequence not in SEQUENCES:
        raise ValueError('Sequence is outside frozen train/validation set')
    preflight = Path(preflight)
    manifest_path = preflight/'manifest.json'
    done, data = read(preflight/'completion.json'), read(manifest_path)
    if (not done.get('passed') or done.get('manifest_sha256') != digest(manifest_path)
            or data.get('config_sha256') != digest(CONFIG) or data.get('manifest_sha256') != digest(DATA)
            or data.get('preflight_sha256') != digest(ROOT/'scripts/preflight_source_group_capture_v1.py')
            or any(digest(ROOT/name) != sha for name, sha in data['source_sha256'].items())
            or data.get('checkpoint_sha256') != digest(checkpoint)):
        raise ValueError('Frozen source-group preflight differs')
    specs = data['sequences']
    if (set(specs) != set(SEQUENCES) or {s for s in specs if specs[s]['split'] == 'train'} != TRAIN
            or {s for s in specs if specs[s]['split'] == 'validation'} != VALIDATION):
        raise ValueError('Source-group training or validation sequence sets differ')
    spec = specs[sequence]
    source = Path(spec['source']).resolve(strict=True)
    if not source.is_relative_to(ARCHIVE.resolve(strict=True)):
        raise ValueError('Archived source is outside automatic-prompt outputs')
    folder = source/sequence
    # The historical records file contains outcome fields. Bind its bytes but
    # never deserialize it during label-free inference preparation.
    completion = read(source/'completion.json')
    setup = read(source/'setup.json')
    prompt_file, mask_file, frame_file = folder/'prompt.json', folder/'prompt_refined.png', folder/'frames.json'
    if (not completion.get('passed') or digest(source/'completion.json') != spec['archived_completion_sha256']
            or completion.get('records_sha256') != digest(source/'records.json')
            or digest(source/'records.json') != spec['archived_records_sha256']
            or digest(frame_file) != spec['archived_frames_sha256']
            or digest(prompt_file) != spec['archived_prompt_sha256']
            or digest(mask_file) != spec['archived_prompt_mask_sha256']
            or source.name != sequence
            or setup.get('sequences') != [sequence]
            or setup.get('initialization') != 'first chronological YOLO detection; no GT prompt'
            or setup['signature'].get('checkpoint_sha256') != digest(checkpoint)
            or setup['signature'].get('detector_sha256') != digest(detector)
            or setup['signature'].get('manifest_sha256') != digest(DATA)):
        raise ValueError('Archived automatic prompt lineage differs')
    prompt = read(prompt_file)
    if (prompt.get('first_detection_index') != spec['prompt_frame']
            or prompt.get('refined_mask_sha256') != spec['archived_prompt_mask_sha256']):
        raise ValueError('Archived prompt mask or frame differs')
    videos = {**load_manifest(DATA, 'dev'), **load_manifest(DATA, 'test')}
    frames = videos[sequence]
    indices, image_shas = image_metadata_only(frame_file)
    if len(frames) != spec['frames'] or len(indices) != len(frames) or len(spec['image_sha256']) != len(frames):
        raise ValueError('Archived image coverage differs')
    for frame, source_index, archived, frozen in zip(frames, indices, image_shas, spec['image_sha256'], strict=True):
        if frame.frame_index != source_index or archived != frozen or digest(frame.image_path) != frozen:
            raise ValueError('Archived input image differs')
    start = spec['prompt_frame']
    if type(start) is not int or not 0 <= start < len(frames)-1:
        raise ValueError('Archived prompt frame invalid')
    with Image.open(frames[start].image_path) as image:
        width, height = image.size
    with Image.open(mask_file) as image:
        mask, viewport = prepare_prompt(np.asarray(image.convert('L')), sequence, (height, width))
    if not mask.any() or tuple(viewport) != (0, 0, width, height):
        raise ValueError('Archived prompt is empty or its viewport differs')
    provenance = {'prompt_file': str(mask_file), 'prompt_mask_sha256': digest(mask_file),
                  'prompt_metadata_sha256': digest(prompt_file),
                  'reference_receipt_sha256': digest(source/'completion.json'),
                  'preflight_sha256': digest(manifest_path), 'roi_xyxy': [0, 0, width, height]}
    return frames, start, mask, provenance
