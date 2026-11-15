import json
from pathlib import Path
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts.memory_token_io import digest
import scripts.verify_native_sam2_automatic_gate as gate
import scripts.report_native_lean_samurai_automatic as report


def make_run(folder, identity, *, changed=None):
    folder.mkdir()
    rows=[]
    for frame in range(12):
        mask=np.zeros((20,20),dtype=np.uint8)
        if frame>=2: mask[3:8,3:8]=255
        if changed==frame: mask[10:13,10:13]=255
        path=folder/f'{frame:06d}.png';Image.fromarray(mask).save(path)
        rows.append({'frame':frame,'source_frame_index':frame,'image_sha256':str(frame),
                     'mask_sha256':digest(path),'post_prompt':frame>=2})
    receipt={'passed':True,'full_video':False,'identity':identity,'deterministic':True,
             'device':identity['device'],'blocking_pointer_transfer':identity['device']=='mps',
             'sequence':'seq20','prompt_frame':2,'prompt':{'mask':'same'},'rebased_start':0,
             'state_sha256':'state','post_prompt_frames':10,'frames':12,'records':rows}
    (folder/'receipt.json').write_text(json.dumps(receipt))


def test_gate_verifies_prompt_inputs_and_repeat_variability(tmp_path,monkeypatch):
    cpu={'device':'cpu'};mps={'device':'mps'}
    monkeypatch.setattr(gate,'identity',lambda checkpoint,device: cpu if device=='cpu' else mps)
    a,b,c=(tmp_path/x for x in ('cpu','mps1','mps2'))
    make_run(a,cpu);make_run(b,mps);make_run(c,mps)
    result=gate.audit(a,b,c,tmp_path/'unused')
    assert result['passed'] and len(result['rows'])==12
    bad=json.loads((c/'receipt.json').read_text());bad['prompt']={'mask':'different'}
    (c/'receipt.json').write_text(json.dumps(bad))
    with pytest.raises(ValueError,match='prompt'):
        gate.audit(a,b,c,tmp_path/'unused')


def test_gate_rejects_changed_masks_and_excess_mps_variation(tmp_path,monkeypatch):
    monkeypatch.setattr(gate,'identity',lambda checkpoint,device:{'device':device})
    a,b,c=(tmp_path/x for x in ('cpu','mps1','mps2'))
    make_run(a,{'device':'cpu'});make_run(b,{'device':'mps'});make_run(c,{'device':'mps'},changed=5)
    with pytest.raises(ValueError,match='variability'):
        gate.audit(a,b,c,tmp_path/'unused')
    (b/'000004.png').write_bytes(b'corrupt')
    with pytest.raises(ValueError,match='mask changed'):
        gate.audit(a,b,c,tmp_path/'unused')


def test_native_scoring_rejects_changed_predictions(tmp_path, monkeypatch):
    native = tmp_path/'native'; native.mkdir()
    gate_path = tmp_path/'gate.json'; gate_path.write_text('{}')
    monkeypatch.setattr(report, 'identity', lambda checkpoint, device: {'device': device})
    monkeypatch.setattr(report, 'verify_gate', lambda *args: {'state_sha256': 'state'})
    frames, records = [], []
    for i in range(3):
        source = tmp_path/f'source{i}.png'
        mask = native/f'{i:06d}.png'
        Image.fromarray(np.zeros((20, 20), dtype=np.uint8)).save(source)
        Image.fromarray(np.full((20, 20), 255 if i >= 1 else 0, dtype=np.uint8)).save(mask)
        frames.append(SimpleNamespace(frame_index=i, image_path=source))
        records.append({'frame': i, 'source_frame_index': i, 'image_sha256': digest(source),
                        'mask_sha256': digest(mask), 'post_prompt': i >= 1})
    receipt = {'passed': True, 'full_video': True, 'sequence': 'seq11', 'prompt_frame': 1,
               'prompt': {}, 'identity': {'device': 'mps'}, 'device': 'mps', 'deterministic': True,
               'blocking_pointer_transfer': True, 'state_sha256': 'state', 'gate_sha256': digest(gate_path),
               'rebased_start': 0, 'frames': 3, 'post_prompt_frames': 2, 'records': records}
    (native/'receipt.json').write_text(json.dumps(receipt))
    assert report.check_native('seq11', frames, native, gate_path, {}, 1, tmp_path/'checkpoint') == receipt
    (native/'000002.png').write_bytes(b'changed')
    with pytest.raises(ValueError, match='prediction changed'):
        report.check_native('seq11', frames, native, gate_path, {}, 1, tmp_path/'checkpoint')
