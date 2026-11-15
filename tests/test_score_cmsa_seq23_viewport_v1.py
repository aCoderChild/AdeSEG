import json
from types import SimpleNamespace

import numpy as np
from PIL import Image
import pytest

from scripts.memory_token_io import digest
from scripts import score_cmsa_seq23_viewport_v1 as scorer


def test_full_and_viewport_truth_and_existing_project_128_threshold(tmp_path, monkeypatch):
    run = tmp_path/'run'; predictions = run/'predictions'/'seq23'; predictions.mkdir(parents=True)
    frames, metadata, image_hashes, pred_hashes, prior_rows = [], [], [], [], []
    for i in range(2):
        source = tmp_path/f'image{i}.png'; label = tmp_path/f'label{i}.png'
        Image.new('RGB', (1280, 720)).save(source)
        truth = np.zeros((720, 1280), np.uint8)
        truth[1, 10 if i == 0 else 446] = 1
        Image.fromarray(truth).save(label)
        mask = np.zeros((720, 834), np.uint8)
        if i == 1: mask[1, 0], mask[1, 1] = 128, 127
        path = predictions/f'{i:06d}.png'; Image.fromarray(mask).save(path)
        frames.append(SimpleNamespace(frame_index=i*2, image_path=source, mask_path=label))
        metadata.append({'label_sha256': digest(label)}); image_hashes.append(digest(source)); pred_hashes.append(digest(path))
        prior_rows.append({'A': {'present': True}})
    meta = tmp_path/'metadata.json'; meta.write_text(json.dumps(metadata))
    prior = tmp_path/'prior.json'; prior.write_text(json.dumps({'rows': {'seq23': prior_rows}}))
    monkeypatch.setattr(scorer, 'PRIOR', prior)
    monkeypatch.setattr(scorer, 'validate_plan', lambda: {})
    monkeypatch.setattr(scorer, 'sequence_inputs', lambda *_: (frames, None, 0, None, None, None, None, None))
    bound = {'frame_metadata': {'path': str(meta), 'sha256': digest(meta)}, 'prior_sha256': digest(prior),
             'run': str(run), 'images': image_hashes, 'prediction_sha256': pred_hashes}
    rows = scorer.scored_rows(bound)
    assert rows[0]['full_frame']['present'] and not rows[0]['viewport']['present']
    assert rows[1]['full_frame']['dice'] == 1.0 and rows[1]['viewport']['dice'] == 1.0
    assert rows[1]['full_frame']['predicted_pixels'] == 1
    assert not rows[0]['post_prompt'] and rows[1]['post_prompt']


def test_scoring_fails_before_label_read_if_verification_fails(tmp_path, monkeypatch):
    verification = tmp_path/'verification.json'; verification.write_text('{}')
    audit = tmp_path/'prediction_audit.json'; audit.write_text('{}')
    monkeypatch.setattr(scorer, 'new_output', lambda p: p)
    monkeypatch.setattr(scorer, 'install_phase_mask_guard', lambda _: {'open_attempts': 0, 'active': True})
    monkeypatch.setattr(scorer, 'read_bound', lambda *args: {'passed': True,
        'label_guard': {'open_attempts': 0}, 'scorer_sha256': digest(scorer.Path(scorer.__file__)),
        'prediction_audit_path': str(audit.resolve())})
    monkeypatch.setattr(scorer, 'validate_inputs', lambda *_: (_ for _ in ()).throw(ValueError('not frozen')))
    monkeypatch.setattr(scorer, 'load_label_mask', lambda *_: pytest.fail('target opened early'))
    monkeypatch.setattr(scorer.sys, 'argv', ['score', '--mode', 'score', '--prediction-audit', str(audit),
        '--verification', str(verification), '--output', str(tmp_path/'out')])
    with pytest.raises(ValueError, match='not frozen'):
        scorer.main()


def test_prior_audit_rejects_changed_manifest(tmp_path, monkeypatch):
    run = tmp_path/'run'; scores = run/'scores'; scores.mkdir(parents=True)
    prior_path = scores/'results.json'; prior_path.write_text('{}')
    manifest = run/'run_manifest.json'; manifest.write_text('manifest')
    mapping = run/'run_map.json'; mapping.write_text('mapping')
    baseline = run/'completion.json'; baseline.write_text('baseline')
    prior = {'passed': True, 'run_map_sha256': digest(mapping), 'launch_manifest_sha256': digest(manifest),
             'run_receipt_sha256': {'seq23': {'A': digest(baseline)}}}
    audit = {'passed': True, 'error_count': 0, 'score_sha256': 'score', 'run': str(run),
             'audit_source_sha256': digest(scorer.ROOT/'scripts/audit_boundary_video_factorial_v1.py'),
             'run_map_sha256': digest(mapping), 'run_manifest_sha256': digest(manifest)}
    monkeypatch.setattr(scorer, 'PRIOR', prior_path)
    scorer.verify_prior(prior, audit, 'score', baseline)
    manifest.write_text('tampered')
    with pytest.raises(ValueError, match='Prior independently'):
        scorer.verify_prior(prior, audit, 'score', baseline)
    manifest.write_text('manifest'); audit['run'] = str(tmp_path/'wrong')
    with pytest.raises(ValueError, match='Prior independently'):
        scorer.verify_prior(prior, audit, 'score', baseline)


def test_scoring_rejects_changed_source_before_label_read(tmp_path, monkeypatch):
    verification = tmp_path/'verification.json'; verification.write_text('{}')
    audit = tmp_path/'prediction_audit.json'; audit.write_text('{}')
    monkeypatch.setattr(scorer, 'new_output', lambda p: p)
    monkeypatch.setattr(scorer, 'install_phase_mask_guard', lambda _: {'open_attempts': 0, 'active': True})
    monkeypatch.setattr(scorer, 'read_bound', lambda *args: {'passed': True,
        'label_guard': {'open_attempts': 0}, 'scorer_sha256': digest(scorer.Path(scorer.__file__)),
        'prediction_audit_path': str(audit.resolve()), 'source_sha256': {'metric': 'old'}})
    monkeypatch.setattr(scorer, 'validate_inputs', lambda *_: {'source_sha256': {'metric': 'tampered'}})
    monkeypatch.setattr(scorer, 'load_label_mask', lambda *_: pytest.fail('target opened early'))
    monkeypatch.setattr(scorer.sys, 'argv', ['score', '--mode', 'score', '--prediction-audit', str(audit),
        '--verification', str(verification), '--output', str(tmp_path/'out')])
    with pytest.raises(ValueError, match='Verified inputs changed'):
        scorer.main()


def test_static_source_closure_includes_transitive_metric_and_prompt_dependencies():
    bound = scorer.source_hashes()
    assert 'scripts/local_import_closure_v1.py' in bound
    assert 'evaluation/davis2017/metrics.py' in bound
    assert 'modeling/samurai_auto_prompt.py' in bound
