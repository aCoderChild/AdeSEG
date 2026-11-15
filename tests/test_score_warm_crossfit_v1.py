"""Synthetic scorer/order tests; no experimental labels are used."""
import json
from pathlib import Path

import numpy as np
from PIL import Image
import pytest
import torch

from scripts import run_warm_crossfit_study_v1 as launcher
from scripts import score_warm_crossfit_v1 as scorer
from scripts.memory_token_io import digest


def test_all_folds_and_controls_freeze_before_score(tmp_path):
    tasks = launcher.phases(tmp_path, tmp_path/'preflight', cache_mib=17)
    names = [x[0] for x in tasks]
    assert len(names) == 64 and names[0] == 'probe'
    assert all(x.startswith('train_') for x in names[1:31])
    assert all(x.startswith('predict_') for x in names[31:61])
    assert names[-3:] == ['verify', 'score', 'audit']
    assert all(cmd[cmd.index('--cache-mib') + 1] == '17' for _, cmd, _ in tasks[1:31])
    assert all('--cache-mib' not in cmd for _, cmd, _ in tasks[31:61])


def test_patient_macro_frame_weighted_and_full_canvas_scoring(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, 'ARMS', ('mask_only',))
    samples = {}; records = {(2, 'mask_only'): {'mapping': {}}, (1, 'mask_only'): {'mapping': {}}}
    # Seq2 contributes two present reads and one empty; seq23 contributes one
    # present read with one target pixel outside the evaluated viewport.
    for seq, frame, fold, foreground in [('seq2', 0, 2, True), ('seq2', 1, 2, True),
                                         ('seq2', 2, 2, False), ('seq23', 0, 1, True)]:
        shape = (720, 1280) if seq == 'seq23' else (8, 8)
        truth = np.zeros(shape, np.uint8)
        if foreground:
            if seq == 'seq23':
                truth[5, 100] = 1; truth[5, 500] = 1
            else:
                truth[2:4, 2:4] = 1
        label = tmp_path/f'{seq}_{frame}_label.png'; Image.fromarray(truth).save(label)
        baseline = truth.copy()
        if seq == 'seq23': baseline[:, :446] = 0
        refined = baseline.copy()
        if seq == 'seq2' and not foreground: refined[0, 0] = 1
        basefile = tmp_path/f'{seq}_{frame}_baseline.png'; Image.fromarray(baseline*255).save(basefile)
        reffile = tmp_path/f'{seq}_{frame}_refined.png'; Image.fromarray(refined*255).save(reffile)
        samples[(seq, frame)] = {'label': str(label)}
        records[(fold, 'mask_only')]['mapping'][(seq, frame)] = {
            'baseline_path': str(basefile), 'baseline_sha256': digest(basefile),
            'refined_path': str(reffile), 'refined_sha256': digest(reffile)}
    monkeypatch.setattr(scorer, 'label_hashes', lambda _, data: {k: digest(Path(v['label'])) for k, v in data.items()})
    primary = scorer.compute({}, samples, records)
    independent = scorer.compute({}, samples, records, independent=True)
    errors = []; scorer.compare(independent, primary, [], errors)
    assert not errors
    base = primary['summaries']['baseline']; refined = primary['summaries']['mask_only']
    assert base['patients_with_present_reads'] == 2
    assert base['patient_macro_present_dice'] == pytest.approx(5/6)
    assert base['pooled']['mean_present_dice'] == pytest.approx(8/9)
    assert primary['seq23_viewport_summaries']['baseline']['mean_present_dice'] == pytest.approx(1.0)
    assert primary['summaries']['baseline']['per_sequence']['seq23']['mean_present_dice'] == pytest.approx(2/3)
    assert refined['pooled']['false_alarms'] == 1
    assert primary['paired_vs_baseline']['mask_only']['pooled']['unchanged'] == 3
    assert primary['paired_vs_baseline']['mask_only']['pooled']['mean_present_dice_difference'] == 0
    assert primary['no_initialization'] == ['seq1', 'seq7']


def test_failed_verification_cannot_open_targets(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, 'install_mask_open_guard', lambda _: {'open_attempts': 0})
    monkeypatch.setattr(scorer, 'load_preflight', lambda _: ({}, 'sha'))
    monkeypatch.setattr(scorer, 'validate_all', lambda *a, **k: (_ for _ in ()).throw(ValueError('not frozen')))
    monkeypatch.setattr(scorer, 'load_label_mask', lambda *_: pytest.fail('target opened'))
    dummy = tmp_path/'preflight'; dummy.mkdir()
    monkeypatch.setattr(scorer.sys, 'argv', ['score', '--mode', 'verify', '--preflight', str(dummy),
        '--batch-root', str(tmp_path), '--output', str(tmp_path/'out')])
    with pytest.raises(ValueError, match='not frozen'):
        scorer.main()
    assert not (tmp_path/'out').exists()


def test_resume_preserves_unfinished_phase_without_deleting(tmp_path):
    out = tmp_path/'train/fold1/mask_only'; out.mkdir(parents=True)
    (out/'partial.bin').write_bytes(b'important')
    log = tmp_path/'train_f1_mask_only.log'; log.write_text('partial')
    launcher.preserve_partial(tmp_path, 'train_f1_mask_only', out/'completion.json')
    saved = list((tmp_path/'partial_attempts').iterdir())
    assert len(saved) == 1 and (saved[0]/'output/partial.bin').read_bytes() == b'important'
    assert (saved[0]/'phase.log').read_text() == 'partial'
    assert not out.exists() and not log.exists()


def test_phase_bound_receipt_rejects_rewrite_during_parse(tmp_path, monkeypatch):
    file = tmp_path/'receipt.json'; file.write_text('{"passed":true}')
    expected = digest(file)
    def changing_read(path):
        value = json.loads(path.read_text())
        path.write_text('{"passed":false}')
        return value
    monkeypatch.setattr(scorer, 'read', changing_read)
    with pytest.raises(ValueError, match='changed during parsing'):
        scorer.read_bound(file, expected)


def test_independent_comparison_rejects_changed_paired_statistics():
    actual = {'paired_vs_baseline': {'mask_only': {'mean_present_dice_difference': .1}}}
    expected = {'paired_vs_baseline': {'mask_only': {'mean_present_dice_difference': .2}}}
    errors = []; scorer.compare(actual, expected, [], errors)
    assert errors and 'mean_present_dice_difference' in errors[0]


def test_all_fold_artifact_verifier_accepts_complete_batch_and_rejects_tamper(tmp_path, monkeypatch):
    # One control across all five patient groups exercises the actual artifact
    # verifier without touching a patient label or requiring a GPU.
    monkeypatch.setattr(scorer, 'BASE', tmp_path)
    monkeypatch.setattr(scorer, 'ARMS', ('mask_only',))
    monkeypatch.setattr(launcher, 'ARMS', ('mask_only',))
    monkeypatch.setattr(scorer, 'sources', lambda: {})
    monkeypatch.setattr(scorer, 'trained_model', lambda *a, **k: None)
    fold_sequences = {1: ['seq6', 'seq23'], 2: ['seq2'], 3: ['seq8'], 4: ['seq12'], 5: ['seq18']}
    samples = {(seq, 0): {'sequence': seq, 'frame': 0} for seqs in fold_sequences.values() for seq in seqs}
    index = {'source_frames': {seq: [{'source_frame_index': 5}] for seq, _ in samples}}
    monkeypatch.setattr(scorer, 'samples_for', lambda _: (index, samples))
    monkeypatch.setattr(scorer, 'fold_definition', lambda plan, fold, rows:
                        (*scorer.fold_split(fold), [[['irrelevant', 1]]]))
    preflight_path = tmp_path/'preflight'; preflight_path.mkdir()
    preflight = {'environment': {}, 'initial_state_sha256': {'mask_only': 'initial'}}
    tasks = launcher.phases(tmp_path, preflight_path, cache_mib=0)
    manifest = {'preflight_path': str(preflight_path), 'preflight_sha256': 'p', 'source_sha256': {},
                'cache_mib': 0, 'phases': [x[0] for x in tasks]}
    (tmp_path/'run_manifest.json').write_text(json.dumps(manifest))
    checks = [{'arm': 'mask_only', 'sequence': seq, 'frame': 0,
               'video_shape': [720, 834] if seq == 'seq23' else [4, 4],
               'head_bias_gradient_abs': 1.0} for seq in ('seq6', 'seq23')]
    phase_rows = []
    for name, _, artifact in tasks[:-3]:
        artifact.parent.mkdir(parents=True, exist_ok=True)
        if name == 'probe':
            record = {'passed': True, 'preflight_sha256': 'p', 'device': 'mps', 'environment': {},
                      'label_guard': {'open_attempts': 0}, 'checks': checks}
        else:
            stage, fold_name, arm = name.split('_', 2); fold = int(fold_name[1:])
            if stage == 'train':
                checkpoint = artifact.parent/'checkpoint.pt'; torch.save({'training_steps': 1}, checkpoint)
                record = {'passed': True, 'fold': fold, 'arm': arm, 'preflight_sha256': 'p',
                    'checkpoint_sha256': digest(checkpoint), 'training_steps': 1, 'initial_state_sha256': 'initial',
                    'device': 'mps', 'environment': {}, 'label_guard': {'blocked_open_attempts': 0},
                    'cache': {'limit_bytes': 0, 'peak_bytes': 0, 'hits': 0, 'misses': 1,
                              'unique_reads': 1, 'final_sha256_verified': True}}
            else:
                train = tmp_path/'train'/f'fold{fold}'/arm
                prediction_rows = []
                for seq in fold_sequences[fold]:
                    source_shape = [720, 1280] if seq == 'seq23' else [4, 4]
                    video_shape = [720, 834] if seq == 'seq23' else source_shape
                    row = {'sequence': seq, 'frame': 0, 'source_frame_index': 5,
                           'source_shape': source_shape, 'video_shape': video_shape}
                    for kind in ('baseline', 'refined'):
                        file = artifact.parent/f'{seq}_{kind}.png'
                        Image.fromarray(np.zeros(source_shape, np.uint8)).save(file)
                        row[kind+'_path'] = str(file); row[kind+'_sha256'] = digest(file)
                    prediction_rows.append(row)
                record = {'passed': True, 'fold': fold, 'arm': arm, 'preflight_sha256': 'p',
                    'checkpoint_sha256': digest(train/'checkpoint.pt'),
                    'training_completion_sha256': digest(train/'completion.json'),
                    'device': 'mps', 'environment': {}, 'label_guard': {'open_attempts': 0}, 'rows': prediction_rows}
        artifact.write_text(json.dumps(record))
        log = tmp_path/(name+'.log'); log.write_text('done')
        phase_rows.append({'phase': name, 'passed': True, 'artifact_path': str(artifact),
                           'artifact_sha256': digest(artifact), 'log_sha256': digest(log)})
    (tmp_path/'phases.jsonl').write_text(''.join(json.dumps(row)+'\n' for row in phase_rows))
    _, _, records, proof = scorer.validate_all(tmp_path, preflight_path, preflight, 'p')
    assert len(records) == 5 and proof['patient_test_folds']['seq23'] == 1
    altered = tmp_path/'predict/fold1/mask_only/completion.json'; altered.write_text('{}')
    with pytest.raises(ValueError, match='phase artifact changed'):
        scorer.validate_all(tmp_path, preflight_path, preflight, 'p')
