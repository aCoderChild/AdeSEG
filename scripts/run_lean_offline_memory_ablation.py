#!/usr/bin/env python3
"""Replay frozen model reads in isolation; fail closed on baseline mismatch."""
import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from modeling.lean_memory_capture import clone_data, zero_memory_segment
from scripts.memory_token_io import digest, validate_output
from scripts.run_author_baseline import source_identity, build_author_predictor, parameter_fingerprints, assert_parameter_integrity

def equal(a, b):
    if torch.is_tensor(a):
        return torch.is_tensor(b) and torch.equal(a, b)
    if isinstance(a, (tuple, list)):
        return type(a) is type(b) and len(a) == len(b) and all(equal(x, y) for x, y in zip(a, b))
    if isinstance(a, dict):
        return isinstance(b, dict) and a.keys() == b.keys() and all(equal(a[k], b[k]) for k in a)
    return a == b

def main():
    p = argparse.ArgumentParser()
    for k in ('checkpoint', 'capture', 'indexed-reference', 'output'):
        p.add_argument('--' + k, type=Path, required=True)
    a = p.parse_args()
    receipt = json.loads((a.capture / 'capture_observation.json').read_text())
    refdone = json.loads((a.indexed_reference / 'completion.json').read_text())
    setup = json.loads((a.capture / 'setup.json').read_text())
    if not receipt['passed'] or not refdone['passed'] or receipt['reference_records_sha256'] != refdone['records_sha256']:
        raise ValueError('Capture and reference do not match')
    if digest(a.capture / 'records.json') != receipt['records_sha256'] or digest(a.indexed_reference / 'records.json') != refdone['records_sha256']:
        raise ValueError('Records changed')
    if digest(a.checkpoint) != setup['identity']['cpu_identity']['checkpoint_sha256']:
        raise ValueError('Checkpoint changed')
    if receipt['capture_sha256'] != digest(ROOT / 'modeling/lean_memory_capture.py'):
        raise ValueError('Capture implementation changed')
    out = validate_output(a.output)
    out.mkdir(parents=True, exist_ok=False)
    torch.set_num_threads(2)
    repo, revision = source_identity('lean')
    if revision != setup['identity']['cpu_identity']['source_signature']['revision']:
        raise ValueError('Lean source revision changed')
    model = build_author_predictor(SimpleNamespace(method='lean', checkpoint=a.checkpoint, device='cpu'), repo)
    before = parameter_fingerprints(model)
    events = []
    with torch.inference_mode(), indexed_spatial_rope(model, []):
        for event in receipt['snapshots']:
            path = a.capture / 'snapshots' / event['snapshot_file']
            if digest(path) != event['sha256']:
                raise ValueError('Snapshot changed')
            data = torch.load(path, map_location='cpu', weights_only=True)
            if data['frame'] != event['frame']:
                raise ValueError('Snapshot frame changed')
            baseline = model.memory_attention(**data['attention_inputs'])
            if not equal(baseline, data['attention_outputs']):
                raise ValueError('Attention baseline does not reproduce frozen output')
            head = model._forward_sam_heads(**data['decoder_inputs'])
            if not equal(head, data['decoder_outputs']):
                raise ValueError('Decoder baseline does not reproduce frozen output')
            masks = {}
            for name in ('noop', 'conditioning', 'other_spatial'):
                attn_args = clone_data(data['attention_inputs'])
                attn_args['memory'] = zero_memory_segment(attn_args['memory'], int(attn_args['num_obj_ptr_tokens']), name)
                features = model.memory_attention(**attn_args)[0]
                head_args = clone_data(data['decoder_inputs'])
                head_args['backbone_features'] = features.permute(1, 2, 0).view(1, features.shape[2], 64, 64)
                result = model._forward_sam_heads(**head_args)
                if name == 'noop' and not equal(result, data['decoder_outputs']):
                    raise ValueError('No-op does not reproduce frozen decoder')
                state = {'device': torch.device('cpu'), 'video_height': data['video_height'], 'video_width': data['video_width']}
                _, video = model._get_orig_video_res_output(state, result[3])
                image = (video[0, 0].cpu().numpy() > 0).astype(np.uint8) * 255
                path = out / f"{event['frame']:06d}_{name}.png"
                Image.fromarray(image).save(path)
                masks[name] = digest(path)
                if name == 'noop' and not np.array_equal(image > 0, np.asarray(Image.open(a.indexed_reference / f"{event['frame']:06d}.png")) > 0):
                    raise ValueError('No-op does not reproduce original mask')
            events.append({'frame': event['frame'], 'masks': masks})
    assert_parameter_integrity(model, before, out, 'after')
    result = {'passed': True, 'sequence': receipt['sequence'], 'events': events,
              'capture_receipt_sha256': digest(a.capture / 'capture_observation.json'),
              'source_sha256': digest(Path(__file__))}
    (out / 'completion.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps(result, indent=2))

if __name__ == '__main__':
    main()
