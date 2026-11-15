import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from scripts import audit_warm_crossfit_video_smoke_v1 as smoke
from scripts.memory_token_io import digest


def test_full_smoke_structure_and_native_mask_tamper(tmp_path, monkeypatch):
    monkeypatch.setattr(smoke, 'OUT', tmp_path)
    monkeypatch.setattr(smoke, 'phase_artifact', lambda root, name, path: digest(path))
    certificate = {'passed': True, 'label_guard': {'open_attempts': 0}, 'references': {}, 'checkpoints': {}}
    for seq, fold in [('seq3', 2), ('seq23', 1)]:
        shape = (720, 1280) if seq == 'seq23' else (4, 5)
        refdir = tmp_path / ('ref_' + seq); refdir.mkdir()
        rows = []
        for i in range(2):
            file = refdir / f'{i:06d}.png'; Image.fromarray(np.zeros(shape, np.uint8)).save(file)
            rows.append({'frame': i, 'mask_sha256': digest(file)})
        reference = {'rows': rows, 'positions': [], 'selection': [], 'anchor_events': [], 'token_window_events': []}
        ref = refdir / 'receipt.json'; ref.write_text(json.dumps(reference))
        roi = [446, 0, 1280, 720] if seq == 'seq23' else None
        certificate['references'][seq] = {'prompt_frame': 0, 'frames': 2, 'reference_path': str(ref),
            'reference_sha256': digest(ref), 'roi_xyxy': roi, 'fold': fold,
            'image_sizes': [[shape[1], shape[0]]] * 2}
        checkpoint = {'fold': fold, 'sha256': f'fold{fold}'}; certificate['checkpoints'][str(fold)] = checkpoint
        for arm in smoke.ARMS:
            folder = tmp_path / 'smoke' / seq / arm; (folder / 'verification').mkdir(parents=True)
            runtime = folder / 'verification/runtime_verification.json'
            runtime.write_text(json.dumps({'passed': True, 'certificate_sha256': 'CERT', 'sequence': seq,
                'device': 'mps', 'fold_checkpoint': checkpoint}))
            masks = []
            for i in range(2):
                file = folder / f'{i:06d}.png'; Image.fromarray(np.zeros(shape, np.uint8)).save(file)
                masks.append({'frame': i, 'sha256': digest(file)})
            traces = {}
            for name in ('positions', 'selection', 'anchor_events', 'token_window_events', 'telemetry'):
                file = folder / f'{name}.json'; file.write_text(json.dumps([{'frame': 1}] if name == 'telemetry' else []))
                traces[name] = digest(file)
            native = []
            if arm == 'B':
                (folder / 'native').mkdir()
                for i in range(2):
                    file = folder / 'native' / f'{i:06d}.png'; Image.fromarray(np.zeros(shape, np.uint8)).save(file)
                    native.append({'frame': i, 'sha256': digest(file)})
            rec = {'passed': True, 'sequence': seq, 'arm': arm, 'limit': 4, 'certificate_sha256': 'CERT',
                'reference_sha256': digest(ref), 'prompt_frame': 0, 'roi_xyxy': roi,
                'label_guard': {'open_attempts': 0}, 'rows': [{'frame': 0}, {'frame': 1}],
                'masks': masks, 'runtime_verification_sha256': digest(runtime), 'trace_sha256': traces,
                'native_masks': native}
            (folder / 'receipt.json').write_text(json.dumps(rec))
    certfile = tmp_path / 'certificate.json'; certfile.write_text(json.dumps(certificate))
    monkeypatch.setattr(smoke, 'read_bound', lambda path, sha: json.loads(path.read_text())
                        if sha == 'CERT' else _read_bound(path, sha))
    result = smoke.audit(tmp_path, certfile, 'CERT'); assert result['passed']
    file = tmp_path / 'smoke/seq3/B/native/000001.png'
    Image.fromarray(np.ones((4, 5), np.uint8) * 255).save(file)
    with pytest.raises(ValueError, match='native or reference mask changed'):
        smoke.audit(tmp_path, certfile, 'CERT')


def _read_bound(path, sha):
    assert digest(path) == sha
    return json.loads(Path(path).read_text())
