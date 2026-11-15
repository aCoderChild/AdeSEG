import json

import pytest
import torch

from scripts.memory_token_io import digest
from scripts.source_group_refiner_shared_v3 import ARMS, PARAMETERS, SEED, initial_hash, model_for
from scripts.train_source_group_refiner_v3 import PARITY_KEYS, build_plan, require_parity, require_plan
from scripts.run_source_group_refiner_study_v3 import make_phases


def test_six_matched_controls_identical_initial_parameters_and_unique_control():
    assert ARMS == ('reference', 'early_only', 'deep_only', 'foreground_only', 'rgb_contrast', 'spatial_shuffle')
    hashes = []
    for arm in ARMS:
        torch.manual_seed(SEED)
        model = model_for(arm)
        assert model.control == arm
        assert sum(p.numel() for p in model.parameters()) == PARAMETERS
        hashes.append(initial_hash(model))
    assert len(set(hashes)) == 1
    with pytest.raises(ValueError):
        model_for('region')


def test_v3_plan_frozen_roundtrip_and_training_subset(tmp_path):
    samples = [{'sequence': 'seq2', 'frame': 4, 'split': 'train'},
               {'sequence': 'seq9', 'frame': 7, 'split': 'validation'}]
    manifest = {'samples': samples, 'feature_pipeline_sha256': 'pipeline',
                'feature_pipeline': {'runtime_environment': {'device': 'mps'}}}
    plan = build_plan(manifest, 'manifest-sha')
    path = tmp_path/'plan.json'; path.write_text(json.dumps(plan))
    assert require_plan(path, manifest, 'manifest-sha') == plan
    assert plan['arms'] == list(ARMS)
    assert all(epoch == [['seq2', 4]] for epoch in plan['training_order'])
    plan['arms'].pop(); path.write_text(json.dumps(plan))
    with pytest.raises(ValueError):
        require_plan(path, manifest, 'manifest-sha')


def test_parity_requires_all_six_arms_and_all_three_training_reads(tmp_path):
    plan_path = tmp_path/'plan.json'; plan_path.write_text('{}')
    plan = {'environment': {'device': 'mps'}}
    checks = [{'arm': arm, 'sequence': seq, 'frame': frame, 'pixels': 1_000_000,
               'mask_difference': 0, 'max_logit_difference': 0., 'cpu_loss': 1., 'mps_loss': 1.,
               'loss_difference': 0., 'loss_tolerance_ratio': 0., 'max_gradient_difference': 0.,
               'max_gradient_tolerance_ratio': 0.} for arm in ARMS for seq, frame in PARITY_KEYS]
    receipt = {'passed': True, 'manifest_sha256': 'm', 'plan_sha256': digest(plan_path),
               'environment': plan['environment'], 'logit_atol': 0.001, 'gradient_atol_rtol': 0.001,
               'mask_max_pixels': 20, 'mask_max_fraction': 1e-5, 'checks': checks}
    path = tmp_path/'parity.json'; path.write_text(json.dumps(receipt))
    require_parity(path, plan, plan_path, 'm')
    receipt['checks'].pop(); path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError):
        require_parity(path, plan, plan_path, 'm')


def test_separate_v3_launcher_orders_all_arms_before_replay_and_scoring(tmp_path):
    phases = make_phases(tmp_path)
    assert [name for name, _ in phases] == (['manifest', 'plan', 'parity']
        + [f'train_{a}' for a in ARMS] + ['verify', 'score'])
    assert all('v3.py' in ' '.join(cmd) for name, cmd in phases if name != 'manifest')


def _payload_for(model, arm):
    from pathlib import Path
    from modeling.lean_channel_attention import author_sha256
    root = Path(__file__).resolve().parents[1]
    return {'format': 'foreground_conditioned_refiner_v1', 'research_only': True,
            'training_steps': 1, 'control': arm, 'author_sha256': author_sha256(),
            'conditioned_source_sha256': digest(root/'modeling/foreground_conditioned_refiner.py'),
            'base_source_sha256': digest(root/'modeling/multilevel_contrast_refiner.py'),
            'gate_mode': 'region', 'contrast_head': True, 'arm': 'multilevel',
            'backbone_sha256': 'a'*64, 'feature_pipeline_sha256': 'b'*64,
            'train_manifest_sha256': 'c'*64, 'state_dict': model.state_dict()}


def test_v3_checkpoint_reload_preserves_each_control_and_rejects_mismatch():
    import copy
    for arm in ARMS:
        model = model_for(arm)
        payload = _payload_for(model, arm)
        kwargs = {'backbone_sha256': 'a'*64, 'feature_pipeline_sha256': 'b'*64,
                  'train_manifest_sha256': 'c'*64}
        restored = model_for(arm)
        restored.load_trained_state(payload, **kwargs)
        assert restored._trained_checkpoint_loaded
        assert initial_hash(restored) == initial_hash(model)
        wrong = copy.copy(payload); wrong['control'] = 'not-the-arm'
        with pytest.raises(ValueError, match='control'):
            model_for(arm).load_trained_state(wrong, **kwargs)
        wrong = copy.copy(payload); wrong['train_manifest_sha256'] = 'd'*64
        with pytest.raises(ValueError, match='provenance'):
            model_for(arm).load_trained_state(wrong, **kwargs)


def test_v3_rgb_inputs_and_checkpoint_replay_use_hash_bound_image(tmp_path, monkeypatch):
    import numpy as np
    from PIL import Image
    from scripts import source_group_ablation_inputs_v1 as rgb_inputs
    torch.manual_seed(5)
    early, deep, logits = torch.randn(1,32,8,8), torch.randn(1,256,4,4), torch.randn(1,1,8,8)
    monkeypatch.setattr(rgb_inputs, 'read_capture', lambda item: (early, deep, logits, (6,8)))
    path = tmp_path/'image.png'
    Image.fromarray(np.arange(6*8*3, dtype=np.uint8).reshape(6,8,3)).save(path)
    item = {'image': str(path), 'image_sha256': digest(path)}
    args, shape = rgb_inputs.inputs(item, 'cpu', 'rgb_contrast')
    assert shape == (6,8) and len(args) == 4 and tuple(args[3].shape) == (1,3,1024,1024)
    model = model_for('rgb_contrast').train()
    torch.nn.init.normal_(model.head[-1].weight, std=0.01)
    with torch.enable_grad():
        original = model(*args).detach()
    restored = model_for('rgb_contrast')
    restored.load_trained_state(_payload_for(model, 'rgb_contrast'), backbone_sha256='a'*64,
                                feature_pipeline_sha256='b'*64, train_manifest_sha256='c'*64)
    restored.eval()
    with torch.no_grad():
        replay = restored(*args)
    torch.testing.assert_close(original, replay, atol=0, rtol=0)
    item['image_sha256'] = '0'*64
    with pytest.raises(ValueError, match='frozen manifest'):
        rgb_inputs.inputs(item, 'cpu', 'rgb_contrast')


def test_frozen_arm_rejects_missing_and_bad_completion_before_loading_checkpoint(tmp_path, monkeypatch):
    from scripts import verify_source_group_refiner_v3 as verifier
    monkeypatch.setattr(verifier, 'mask_path_sets', lambda path: (set(), set()))
    with pytest.raises(FileNotFoundError):
        verifier.check_frozen_arm(tmp_path, 'reference', {}, 'm', tmp_path/'plan', tmp_path/'parity', {})
    (tmp_path/'completion.json').write_text(json.dumps({'passed': False}))
    (tmp_path/'frozen_predictions.json').write_text('{}')
    with pytest.raises(ValueError, match='incomplete training'):
        verifier.check_frozen_arm(tmp_path, 'reference', {}, 'm', tmp_path/'plan', tmp_path/'parity', {})


def test_frozen_arm_detects_tampered_prediction_and_wrong_control(tmp_path, monkeypatch):
    from scripts import verify_source_group_refiner_v3 as v
    monkeypatch.setattr(v, 'mask_path_sets', lambda path: (set(), set()))
    monkeypatch.setattr(v, 'dependencies', lambda: {'test_source': 'hash'})
    plan_path = tmp_path/'plan.json'; plan_path.write_text('{}')
    parity_path = tmp_path/'parity.json'; parity_path.write_text('{}')
    checkpoint = tmp_path/'checkpoint.pt'; checkpoint.write_bytes(b'checkpoint')
    samples = [{'sequence': 'seq9', 'frame': i, 'split': 'validation'} for i in range(50)]
    manifest = {'samples': samples, 'checkpoint_sha256': 'backbone', 'feature_pipeline_sha256': 'pipeline'}
    order = [[['seq2', i] for i in range(164)] for _ in range(v.EPOCHS)]
    plan = {'initial_state_sha256': 'initial', 'training_order': order, 'environment': {'device':'mps'}}
    rows = []
    for i in range(50):
        row = {'sequence':'seq9', 'frame':i, 'shape':[1,1]}
        for name in ('baseline', 'refined'):
            path = tmp_path/f'{i}_{name}.png'; path.write_bytes(f'{i}-{name}'.encode())
            row[name], row[name+'_sha256'] = str(path), digest(path)
        rows.append(row)
    frozen = tmp_path/'frozen_predictions.json'
    frozen.write_text(json.dumps({'checkpoint_sha256': digest(checkpoint), 'rows': rows}))
    history = [{'epoch':i+1, 'samples':164, 'sample_order':order[i], 'mean_training_loss':1.0}
               for i in range(v.EPOCHS)]
    completion = {'passed':True, 'arm':'reference', 'manifest_sha256':'manifest',
                  'plan_sha256':digest(plan_path), 'parity_sha256':digest(parity_path),
                  'checkpoint_sha256':digest(checkpoint), 'frozen_predictions_sha256':digest(frozen),
                  'initial_state_sha256':'initial', 'training_order':order, 'history':history,
                  'label_guard':{'blocked_open_attempts':0, 'train_paths':0, 'blocked_paths':0}}
    (tmp_path/'completion.json').write_text(json.dumps(completion))
    payload = {'research_only':True, 'training_steps':164*v.EPOCHS, 'study_arm':'reference',
               'train_manifest_sha256':'manifest', 'backbone_sha256':'backbone', 'feature_pipeline_sha256':'pipeline',
               'plan_sha256':digest(plan_path), 'parity_sha256':digest(parity_path), 'initial_state_sha256':'initial',
               'dependency_sha256':{'test_source':'hash'}, 'environment':plan['environment'], 'seed':v.SEED,
               'epochs':v.EPOCHS, 'parameter_count':v.PARAMETERS, 'loss':v.LOSS, 'optimizer':'Adam',
               'learning_rate':0.0003, 'weight_decay':0.0001, 'gate_mode':'region', 'contrast_head':True,
               'arm':'multilevel', 'control':'reference'}
    monkeypatch.setattr(v.torch, 'load', lambda *a, **kw: payload)
    check = lambda: v.check_frozen_arm(tmp_path, 'reference', manifest, 'manifest', plan_path, parity_path, plan)
    assert len(check()['mapping']) == 50
    payload['control'] = 'rgb_contrast'
    with pytest.raises(ValueError, match='checkpoint identity'):
        check()
    payload['control'] = 'reference'
    Path = __import__('pathlib').Path
    Path(rows[0]['refined']).write_bytes(b'tampered')
    with pytest.raises(ValueError, match='frozen prediction changed'):
        check()
