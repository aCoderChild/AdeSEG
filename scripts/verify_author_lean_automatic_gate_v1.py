#!/usr/bin/env python3
"""Verify two independent-geometry MPS smoke trajectories without labels."""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest
from scripts.run_author_lean_automatic_v1 import DATA, identity, inputs, new_output
from scripts.source_group_label_guard_v1 import install_mask_open_guard

REPRESENTATIVES = ('seq6', 'seq23')


def audit(smokes, checkpoint, detector, extended, remaining):
    if set(smokes) != set(REPRESENTATIVES): raise ValueError('Both representative smokes are required')
    rows, state, common, processes = [], None, None, set()
    for seq in REPRESENTATIVES:
        root = Path(smokes[seq]).resolve(strict=True); path = root / 'receipt.json'
        sha = digest(path); rec = json.loads(path.read_text())
        if digest(path) != sha: raise ValueError('Smoke receipt changed during parsing')
        frames, start, _, prompt = inputs(seq, extended, remaining, checkpoint, detector)
        expected = identity(seq, checkpoint, detector)
        records = rec.get('records', [])
        if (rec.get('passed') is not True or rec.get('full_video') is not False or
                rec.get('identity') != expected or rec.get('prompt_frame') != start or rec.get('prompt') != prompt or
                rec.get('label_guard', {}).get('open_attempts') != 0 or
                len(records) != min(len(frames), start + 5) or not rec.get('state_sha256') or
                not isinstance(rec.get('process_id'), int) or rec['process_id'] in processes or
                not rec.get('positions')):
            raise ValueError('Smoke identity, guard or coverage differs')
        processes.add(rec['process_id'])
        for i, (frame, row) in enumerate(zip(frames, records, strict=False)):
            file = root / f'{i:06d}.png'; expected_sha = row['mask_sha256']
            if (row.get('frame') != i or row.get('source_frame_index') != frame.frame_index or
                    row.get('image_sha256') != digest(frame.image_path) or row.get('post_prompt') != (i >= start) or
                    digest(file) != expected_sha): raise ValueError('Smoke frame identity differs')
            with Image.open(frame.image_path) as src, Image.open(file) as image:
                values = np.asarray(image.convert('L'))
                if image.format != 'PNG' or image.size != src.size or not np.isin(values, (0, 255)).all() or (i < start and values.any()):
                    raise ValueError('Smoke mask geometry or binary encoding differs')
            if digest(file) != expected_sha: raise ValueError('Smoke mask changed during reading')
        shared = {k: v for k, v in expected.items() if k != 'sequence'}
        if state is not None and (state != rec['state_sha256'] or common != shared):
            raise ValueError('Smoke model states or source identities differ')
        state, common = rec['state_sha256'], shared
        rows.append({'sequence': seq, 'receipt_path': str(path), 'receipt_sha256': sha})
    return {'passed': True, 'scope': 'Two MPS smoke trajectories; no CPU comparison or repeated-run equivalence.',
            'state_sha256': state, 'common_identity': common, 'smokes': rows,
            'verifier_sha256': digest(Path(__file__))}


def verify_gate(path, expected, checkpoint, detector, extended, remaining):
    path = Path(path).resolve(strict=True)
    from scripts.run_additional_author_comparator_v2 import EXPERIMENT_ROOT
    if not path.is_relative_to(EXPERIMENT_ROOT.resolve(strict=True)):
        raise ValueError('Gate is outside the experiment root')
    gate_sha = digest(path); gate = json.loads(path.read_text())
    if digest(path) != gate_sha: raise ValueError('Gate changed during parsing')
    if gate.get('label_guard', {}).get('open_attempts') != 0:
        raise ValueError('Gate label guard differs')
    smokes = {r['sequence']: Path(r['receipt_path']).parent for r in gate.get('smokes', [])}
    if any(root.resolve() != path.parent.parent/'smoke'/seq for seq, root in smokes.items()):
        raise ValueError('Gate smokes are outside this batch')
    checked = audit(smokes, checkpoint, detector, extended, remaining)
    if ({k: v for k, v in gate.items() if k != 'label_guard'} != checked or
            checked['common_identity'] != {k: v for k, v in expected.items() if k != 'sequence'}):
        raise ValueError('Gate reconstruction or identity differs')
    return checked


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('seq6', 'seq23', 'checkpoint', 'detector', 'extended-preflight', 'remaining-preflight', 'output'):
        p.add_argument('--' + name, type=Path, required=True)
    a = p.parse_args(); out = new_output(a.output); guard = install_mask_open_guard(DATA)
    result = audit({'seq6': a.seq6, 'seq23': a.seq23}, a.checkpoint, a.detector,
                   a.extended_preflight, a.remaining_preflight)
    if guard['open_attempts']: raise ValueError('Annotation open attempted')
    result['label_guard'] = dict(guard); out.mkdir(parents=True, exist_ok=False)
    (out / 'gate.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'representatives': REPRESENTATIVES}))


if __name__ == '__main__': main()
