import numpy as np
import pytest
import json
from pathlib import Path

from scripts.audit_source_group_preservation_scores_v1 import metrics as independent_metric, matched_contrasts
from scripts.audit_source_group_preservation_scores_v1 import summarize as independent_summary
from scripts.score_source_group_preservation_v1 import metric, summary
import scripts.audit_source_group_preservation_scores_v1 as audit


@pytest.mark.parametrize('case', ['empty', 'one_empty', 'touch_edge', 'perturbed', 'full'])
def test_independent_metric_matches_scorer_for_mask_geometries(case):
    rng = np.random.default_rng(17)
    truth = np.zeros((43, 48), dtype=bool); pred = np.zeros_like(truth)
    if case == 'one_empty':
        truth[14:24, 12:28] = True
    elif case == 'touch_edge':
        truth[0:25, 0:14] = True; pred[0:23, 2:16] = True
    elif case == 'perturbed':
        truth[5:30, 9:34] = True; pred[7:32, 11:36] = True
        pred[rng.integers(0, 43, 10), rng.integers(0, 48, 10)] = True
    elif case == 'full':
        truth.fill(True); pred.fill(True)
    assert independent_metric(pred, truth) == pytest.approx(metric(pred, truth))


def test_independent_summary_and_preservation_pairing():
    truth = np.zeros((32, 32), dtype=bool); truth[8:15, 7:16] = True
    empty = np.zeros_like(truth); pred = np.zeros_like(truth); pred[9:16, 7:16] = True
    rows = [{'present': True, 'arm': metric(pred, truth)}, {'present': False, 'arm': metric(pred, empty)}]
    assert independent_summary(rows, 'arm') == pytest.approx(summary(rows, 'arm'))
    base = {'pooled_frames': {'mean_present_dice': .6, 'mean_present_boundary_f_5px': .5, 'empty_false_alarms': 2},
            'per_video': {'seq9': {'mean_present_dice': .4}}, 'equal_video_mean_present_dice': .4}
    improved = {'pooled_frames': {'mean_present_dice': .7, 'mean_present_boundary_f_5px': .8, 'empty_false_alarms': 1},
                'per_video': {'seq9': {'mean_present_dice': .6}}, 'equal_video_mean_present_dice': .6}
    data = {f'{m}_{o}': (base if o == 'base' else improved)
            for m in ('region_clean', 'region_aug', 'no_contrast_aug', 'mask_only_aug') for o in ('base', 'preserve')}
    for contrast in matched_contrasts(data).values():
        assert contrast['pooled_present_dice_delta'] == pytest.approx(.1)
        assert contrast['pooled_boundary_f_delta'] == pytest.approx(.3)
        assert contrast['empty_false_alarm_delta'] == -1
        assert contrast['per_video_present_dice_delta']['seq9'] == pytest.approx(.2)


def test_independent_metadata_rejects_wrong_role_and_width(tmp_path, monkeypatch):
    (tmp_path/'data').mkdir()
    (tmp_path/'data/polypgen_sequence.jsonl').write_text('\n'.join(json.dumps(x) for x in [
        {'video_id': 'seq2', 'mask': 'train.png'}, {'video_id': 'seq9', 'mask': 'held.png'}]))
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    arm = 'region_aug_preserve'
    order = [[['seq2', f] for f in range(164)] for _ in range(10)]
    plan = {'training_order': order, 'dependency_sha256': {}, 'environment': {}, 'initial_state_sha256': 'i'}
    manifest = {'checkpoint_sha256': 'c', 'feature_pipeline_sha256': 'f'}
    checkpoint = {'research_only': True, 'study_arm': arm, 'seed': 20261115, 'epochs': 10,
                  'parameter_count': 48005, 'optimizer': 'Adam', 'learning_rate': .0003,
                  'weight_decay': .0001, 'loss': audit.LOSS[arm], 'gate_mode': 'region',
                  'arm': 'multilevel', 'contrast_head': True, 'backbone_sha256': 'c',
                  'feature_pipeline_sha256': 'f', 'dependency_sha256': {}, 'environment': {},
                  'initial_state_sha256': 'i', 'training_steps': 1640}
    history = [{'epoch': e+1, 'samples': 164, 'sample_order': order[e], 'mean_training_loss': 1.0,
                'widths': [audit.expected_width(arm, s, f, e) for s, f in order[e]]} for e in range(10)]
    completion = {'history': history, 'training_order': order,
                  'label_guard': {'train_paths': 1, 'blocked_paths': 1, 'blocked_open_attempts': 0}}
    audit.verify_training_metadata(arm, checkpoint, completion, plan, manifest)
    bad = dict(checkpoint); bad['contrast_head'] = False
    with pytest.raises(ValueError, match='role'):
        audit.verify_training_metadata(arm, bad, completion, plan, manifest)
    completion['history'][0]['widths'][0] = 2
    with pytest.raises(ValueError, match='augmentation'):
        audit.verify_training_metadata(arm, checkpoint, completion, plan, manifest)


def test_launch_record_rejects_tampering_before_any_labels(tmp_path, monkeypatch):
    root = tmp_path/'root'; (root/'scripts').mkdir(parents=True); (root/'modeling').mkdir()
    launcher = root/'scripts/run_source_group_preservation_study_v1.py'; launcher.write_text('launcher')
    discovery = root/'scripts/resume_source_group_capture_v1.py'; discovery.write_text('discovery')
    dependency = root/'modeling/a.py'; dependency.write_text('model')
    study = tmp_path/'study'; study.mkdir(); frozen = tmp_path/'manifest.json'; original_map = tmp_path/'old_map.json'
    mapping = {'seq2': {'capture': 'frozen'}}; original_map.write_text(json.dumps(mapping))
    (study/'capture_map.json').write_text(json.dumps(mapping))
    manifest = {'capture_map_path': str(original_map), 'preflight_sha256': 'p', 'source_freeze_sha256': 's',
                'short_gate_sha256': 'g', 'checkpoint_sha256': 'c'}
    frozen.write_text(json.dumps(manifest))
    monkeypatch.setattr(audit, 'ROOT', root); monkeypatch.setattr(audit, 'MANIFEST', frozen)
    deps = {'modeling/a.py': audit.digest(dependency)}
    record = {'frozen_manifest_path': str(frozen), 'frozen_manifest_sha256': audit.digest(frozen),
              'launcher_sha256': audit.digest(launcher), 'discovery_sha256': audit.digest(discovery),
              'capture_map_sha256': audit.digest(study/'capture_map.json'), 'min_free_bytes': audit.GUARD_BYTES,
              'child_environment': {'PYTHONHASHSEED': '0', 'PYTORCH_ENABLE_MPS_FALLBACK': '0', 'PYTORCH_MPS_FAST_MATH': '0'},
              'dependency_sha256': deps, 'preflight_sha256': 'p', 'source_freeze_sha256': 's',
              'short_gate_sha256': 'g', 'checkpoint_sha256': 'c'}
    record_path = study/'run_manifest.json'; record_path.write_text(json.dumps(record))
    phases = ['plan', 'parity', *['train_'+arm for arm in audit.ARMS], 'verify', 'score']
    events = []
    for phase in phases:
        for status in ('started', 'passed'):
            event = {'phase': phase, 'status': status, 'free_bytes': audit.GUARD_BYTES,
                     'launcher_sha256': record['launcher_sha256'], 'run_manifest_sha256': audit.digest(record_path)}
            if status == 'passed': event['returncode'] = 0
            events.append(event)
    (study/'events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
    plan = {'dependency_sha256': deps}
    assert audit.verify_launch_record(study, plan)[0] == frozen
    (study/'capture_map.json').write_text('{}')
    with pytest.raises(ValueError, match='capture map'):
        audit.verify_launch_record(study, plan)
    (study/'capture_map.json').write_text(json.dumps(mapping))
    events[-1]['status'] = 'failed'
    (study/'events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
    with pytest.raises(ValueError, match='phase order'):
        audit.verify_launch_record(study, plan)
    events[-1]['status'] = 'passed'
    (study/'events.jsonl').write_text(''.join(json.dumps(x)+'\n' for x in events))
    bad = dict(record); bad['child_environment'] = dict(record['child_environment'], PYTHONHASHSEED='7')
    record_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='environment'):
        audit.verify_launch_record(study, plan)
    bad = dict(record); bad['frozen_manifest_sha256'] = 'wrong'; record_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='frozen manifest'):
        audit.verify_launch_record(study, plan)
    alternate = tmp_path/'alternate_manifest.json'; alternate.write_text(frozen.read_text())
    bad = dict(record); bad['frozen_manifest_path'] = str(alternate); record_path.write_text(json.dumps(bad))
    with pytest.raises(ValueError, match='frozen manifest'):
        audit.verify_launch_record(study, plan)
