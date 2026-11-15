import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from scripts.memory_token_io import digest
from scripts import verify_samurai_automatic_gate as verify
from scripts.score_samurai_automatic_priority import score, summary, verify_frozen
from modeling.samurai_mps_kalman import portable_kalman_iou


def write_replay(folder, device, monkeypatch):
    folder.mkdir()
    rows = []
    for i in range(4):
        path = folder / f'{13+i:06d}.png'
        Image.fromarray(np.zeros((3, 5), dtype=np.uint8)).save(path)
        rows.append({'frame': 13+i, 'local_index': i, 'source_frame_index': 13+i,
                     'image_sha256': str(i), 'mask_sha256': digest(path)})
    r = {'passed': True, 'full_video': False, 'device': device, 'source_revision': 'rev',
         'torch': str(torch.__version__), 'config_sha256': digest(verify.CONFIG),
         'manifest_sha256': digest(verify.MANIFEST), 'protocol_sha256': digest(verify.PROTOCOL),
         'runner_sha256': digest(verify.ROOT / 'scripts/run_samurai_automatic_priority.py'),
         'kalman_adapter_sha256': digest(verify.ROOT / 'modeling/samurai_mps_kalman.py'),
         'kalman_iou_events': [{'count':1, 'max_rounding_error':0.0}],
         'fallback': '0', 'fast_math': '0', 'frames': 4, 'records': rows,
         'sequence': 'seq20', 'prompt_frame': 13, 'prompt': {'prompt_mask_sha256':'abc'},
         'rebased_start': 0, 'checkpoint_sha256': 'ckpt', 'state_sha256': 'state',
         'seed': 20261021, 'dtype': 'float32', 'deterministic': True}
    (folder / 'receipt.json').write_text(json.dumps(r))


def test_gate_rejects_changed_mask_and_changed_prompt(tmp_path, monkeypatch):
    monkeypatch.setattr(verify, 'source_identity', lambda _: (Path('/unused'), 'rev'))
    cpu, a, b = [tmp_path / x for x in ('cpu', 'a', 'b')]
    for path, dev in ((cpu, 'cpu'), (a, 'mps'), (b, 'mps')):
        write_replay(path, dev, monkeypatch)
    receipt = verify.audit(cpu, a, b)
    assert receipt['passed'] and sum(x['mps_repeat_pixels'] for x in receipt['rows']) == 0
    changed = json.loads((b / 'receipt.json').read_text())
    changed['kalman_iou_events'] = []
    (b / 'receipt.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='stable-scoring branch'):
        verify.audit(cpu, a, b)
    changed['kalman_iou_events'] = [{'count':1, 'max_rounding_error':0.0}]
    (b / 'receipt.json').write_text(json.dumps(changed))
    changed = json.loads((b / 'receipt.json').read_text())
    changed['prompt']['prompt_mask_sha256'] = 'different'
    (b / 'receipt.json').write_text(json.dumps(changed))
    with pytest.raises(ValueError, match='Prompt'):
        verify.audit(cpu, a, b)
    write = np.ones((3, 5), dtype=np.uint8) * 255
    changed['prompt']['prompt_mask_sha256'] = 'abc'
    (b / 'receipt.json').write_text(json.dumps(changed))
    Image.fromarray(write).save(b / '000013.png')
    with pytest.raises(ValueError, match='mask changed'):
        verify.audit(cpu, a, b)


def test_score_distinguishes_empty_alarms_and_excludes_prompt_when_requested():
    empty = np.zeros((2, 3), dtype=bool)
    one = empty.copy(); one[0, 0] = True
    assert score(empty, empty)['dice'] == 1.0
    assert score(one, empty)['false_alarm'] is True
    assert score(empty, one)['dice'] == 0.0
    rows = [{'post_prompt': False, 'samurai': score(one, one)},
            {'post_prompt': True, 'samurai': score(empty, one)},
            {'post_prompt': True, 'samurai': score(one, empty)}]
    report = summary(rows, 'samurai', post_only=True)
    assert report == {'frames':2, 'present_frames':1, 'mean_present_frame_dice':0.0,
                      'empty_frames':1, 'false_alarm_frames':1, 'mean_empty_predicted_pixels':1.0}


def test_portable_kalman_preserves_cpu_values_and_casts_only_mps():
    class KF:
        def compute_iou(self, *_): return [np.float64(0.7), np.float64(0.25)]
    class Model:
        def __init__(self): self.kf = KF()
    model=Model(); observations=[]
    with portable_kalman_iou(model, 'cpu', observations):
        values=model.kf.compute_iou(None, None)
        assert isinstance(values[0], np.float64)
    assert 'compute_iou' not in vars(model.kf)
    with portable_kalman_iou(model, 'mps', observations):
        values=model.kf.compute_iou(None, None)
        assert values.dtype==np.float32
        np.testing.assert_allclose(values, [0.7, 0.25], rtol=1e-6)
    assert 'compute_iou' not in vars(model.kf)
    assert len(observations)==2 and observations[1]['max_rounding_error']>0


def test_frozen_full_video_covers_original_frame_indices_and_rejects_mask_tamper(tmp_path):
    from types import SimpleNamespace
    from scripts.score_samurai_automatic_priority import ROOT
    samurai=tmp_path/'samurai';lean=tmp_path/'lean';prompt=tmp_path/'prompt'
    for p in (samurai,lean,prompt):p.mkdir()
    frames=[];original=[];lean_hashes=[];samurai_rows=[]
    for i in range(3):
        image=tmp_path/f'image{i}.png'; label=tmp_path/f'label{i}.png'
        for p in (image,label,lean/f'{i:06d}.png'):
            Image.fromarray(np.zeros((3,5),dtype=np.uint8)).save(p)
        frames.append(SimpleNamespace(image_path=image,mask_path=label,frame_index=i))
        original.append({'image_sha256':digest(image),'label_sha256':digest(label)})
        lean_hashes.append(digest(lean/f'{i:06d}.png'))
        if i:
            path=samurai/f'{i:06d}.png';Image.fromarray(np.zeros((3,5),dtype=np.uint8)).save(path)
            samurai_rows.append({'local_index':i-1,'frame':i,'source_frame_index':i,
                                 'image_sha256':digest(image),'mask_sha256':digest(path)})
    (prompt/'frames.json').write_text(json.dumps(original))
    (lean/'frames.json').write_text(json.dumps([{} for _ in frames]))
    record={'sequence':'seq20','frames':3,'frames_sha256':digest(lean/'frames.json'),'mask_sha256':lean_hashes}
    (lean/'records.json').write_text(json.dumps(record))
    (lean/'completion.json').write_text(json.dumps({'passed':True,'records_sha256':digest(lean/'records.json')}))
    sr={'passed':True,'full_video':True,'sequence':'seq20','prompt_frame':1,'frames':2,
        'runner_sha256':digest(ROOT/'scripts/run_samurai_automatic_priority.py'),
        'records':samurai_rows,'prompt':{'prompt_file':str(prompt/'prompt.png')}}
    (samurai/'receipt.json').write_text(json.dumps(sr))
    start, labels, _=verify_frozen('seq20',samurai,lean,frames)
    assert start==1 and labels==[x['label_sha256'] for x in original]
    Image.fromarray(np.ones((3,5),dtype=np.uint8)).save(samurai/'000002.png')
    with pytest.raises(ValueError,match='saved prediction differs'):
        verify_frozen('seq20',samurai,lean,frames)
