import json
import subprocess
import sys

import torch

from scripts.score_source_group_refiner_v1 import metric, summary
from scripts.source_group_refiner_shared_v1 import ARMS, PARAMETERS, SEED, initial_hash, model_for, schedule


def test_matched_arms_have_identical_parameters_and_mask_control():
    hashes = []
    for arm in ARMS:
        torch.manual_seed(SEED); model = model_for(arm)
        hashes.append(initial_hash(model))
        assert sum(x.numel() for x in model.parameters()) == PARAMETERS
        assert model.gate_mode == ('global' if arm == 'global' else 'region')
        assert model._mask_only == (arm == 'mask_only')
        assert model.contrast_head == (arm != 'no_contrast')
    assert len(set(hashes)) == 1


def test_training_order_is_deterministic_and_excludes_validation():
    samples = [{'sequence': 'seq2', 'frame': x, 'split': 'train'} for x in (2, 3, 4)]
    samples += [{'sequence': 'seq9', 'frame': 7, 'split': 'validation'}]
    one = schedule(samples)
    assert one == schedule(list(reversed(samples)))
    assert all(sorted(epoch) == [['seq2', 2], ['seq2', 3], ['seq2', 4]] for epoch in one)
    assert json.loads(json.dumps(one)) == one


def test_full_frozen_plan_json_roundtrip(tmp_path):
    from scripts.train_source_group_refiner_v1 import build_plan, require_plan
    manifest = {'samples': [{'sequence': 'seq2', 'frame': 4, 'split': 'train'}],
                'feature_pipeline_sha256': 'pipeline', 'feature_pipeline': {'runtime_environment': {'device':'mps'}}}
    plan = build_plan(manifest, 'manifest-sha')
    path = tmp_path/'plan.json'; path.write_text(json.dumps(plan))
    assert require_plan(path, manifest, 'manifest-sha') == plan


def test_parity_receipt_requires_all_arms_and_bounded_finite_diagnostics(tmp_path):
    import pytest
    from scripts.memory_token_io import digest
    from scripts.train_source_group_refiner_v1 import PARITY_KEYS, require_parity
    plan_path = tmp_path/'plan.json'; plan_path.write_text('{}')
    plan = {'environment': {'device': 'mps'}}
    checks = [{'arm': arm, 'sequence': seq, 'frame': frame, 'pixels': 1000000,
               'mask_difference': 0, 'max_logit_difference': 0., 'cpu_loss': 1., 'mps_loss': 1.,
               'loss_difference': 0., 'loss_tolerance_ratio': 0., 'max_gradient_difference': 0.,
               'max_gradient_tolerance_ratio': 0.} for arm in ARMS for seq, frame in PARITY_KEYS]
    receipt = {'passed': True, 'manifest_sha256': 'm', 'plan_sha256': digest(plan_path),
               'environment': plan['environment'], 'logit_atol': 0.001, 'gradient_atol_rtol': 0.001,
               'mask_max_pixels': 20, 'mask_max_fraction': 1e-5, 'checks': checks}
    path = tmp_path/'parity.json'; path.write_text(json.dumps(receipt))
    require_parity(path, plan, plan_path, 'm')
    receipt['checks'][0]['max_gradient_tolerance_ratio'] = 1.01; path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError): require_parity(path, plan, plan_path, 'm')
    receipt['checks'].pop(); path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError): require_parity(path, plan, plan_path, 'm')


def test_undefined_empty_and_precision_are_reported_explicitly():
    import numpy as np
    truth = np.array([[True, False]]); pred = np.array([[False, False]])
    m = metric(pred, truth)
    assert m['dice'] == 0 and m['precision'] is None and m['recall'] == 0
    s = summary([{'present': True, 'arm': m}], 'arm')
    assert s['empty_frames'] == 0 and s['mean_empty_predicted_pixels'] is None
    assert s['precision_defined_frames'] == 0 and s['mean_present_precision_when_defined'] is None


def test_training_guard_allows_training_only_and_rejects_path_overlap(tmp_path):
    train = tmp_path/'train.jpg'; val = tmp_path/'val.jpg'
    train.write_bytes(b'train'); val.write_bytes(b'val')
    manifest = tmp_path/'manifest.jsonl'
    manifest.write_text('\n'.join(json.dumps(x) for x in
        ({'video_id':'seq2', 'mask':str(train)}, {'video_id':'seq9', 'mask':str(val)})))
    code = '''from pathlib import Path\nimport sys\nfrom scripts.source_group_training_label_guard_v1 import install_training_mask_guard\ng=install_training_mask_guard(Path(sys.argv[1]))\nassert Path(sys.argv[2]).read_bytes()==b"train"\ntry: Path(sys.argv[3]).read_bytes()\nexcept PermissionError: pass\nelse: raise AssertionError("validation label opened")\nassert g["blocked_open_attempts"]==1\n'''
    subprocess.run([sys.executable, '-c', code, str(manifest), str(train), str(val)], check=True)
    manifest.write_text('\n'.join(json.dumps(x) for x in
        ({'video_id':'seq2', 'mask':str(train)}, {'video_id':'seq9', 'mask':str(train)})))
    code = '''from pathlib import Path\nimport sys\nfrom scripts.source_group_training_label_guard_v1 import install_training_mask_guard\ntry: install_training_mask_guard(Path(sys.argv[1]))\nexcept ValueError: pass\nelse: raise AssertionError("train/validation alias accepted")\n'''
    subprocess.run([sys.executable, '-c', code, str(manifest)], check=True)
