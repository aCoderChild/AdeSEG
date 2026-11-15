import ast
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

from scripts.memory_token_io import digest
from scripts.run_source_group_boundary_multiseed_v1 import make_phases
from scripts.source_group_boundary_multiseed_shared_v1 import ARMS, SEEDS, dependencies
import scripts.freeze_source_group_boundary_multiseed_v1 as gate
import scripts.score_source_group_boundary_multiseed_v1 as scorer
import scripts.report_source_group_boundary_multiseed_v1 as report
import scripts.audit_source_group_boundary_multiseed_v1 as audit


def test_order_freezes_both_seeds_before_scoring(tmp_path):
    phases = make_phases(tmp_path)
    names = [p[0] for p in phases]
    assert len(phases) == 25
    for seed in SEEDS:
        assert sum(name.startswith(f'{seed}_train_') for name in names) == 6
        assert names.index(f'{seed}_verify') < names.index('joint_gate')
        assert names.index('joint_gate') < names.index(f'{seed}_score')
        assert names.index(f'{seed}_score') < names.index(f'{seed}_independent_audit')
        for name, actual_seed, _ in phases:
            if name.startswith(str(seed)):
                assert actual_seed == seed
    assert names[-1] == 'report'


def test_seed_is_explicit_distinct_and_unapproved_seed_fails():
    script = ("import json,torch; from scripts.source_group_boundary_multiseed_shared_v1 "
              "import SEED,schedule,initial_hash,model_for; torch.manual_seed(SEED); "
              "s=[{'split':'train','sequence':'s','frame':i} for i in range(20)]; "
              "print(json.dumps({'seed':SEED,'order':schedule(s),'state':initial_hash(model_for('region_bce'))}))")
    outputs = []
    for seed in SEEDS:
        env = {**os.environ, 'ADESEG_BOUNDARY_MULTISEED': str(seed), 'PYTHONHASHSEED': '0'}
        outputs.append(json.loads(subprocess.check_output([sys.executable, '-c', script], env=env, text=True)))
    assert [o['seed'] for o in outputs] == list(SEEDS)
    assert outputs[0]['state'] != outputs[1]['state']
    assert all(a != b for a in outputs[0]['order'] for b in outputs[1]['order'])
    env = {**os.environ, 'ADESEG_BOUNDARY_MULTISEED': '20261115'}
    assert subprocess.run([sys.executable, '-c', script], env=env, capture_output=True).returncode != 0


def test_plan_requires_explicit_seed_and_python_hash_seed():
    script = ("from scripts.train_source_group_boundary_multiseed_v1 import build_plan; "
              "m={'feature_pipeline_sha256':'b'*64,'feature_pipeline':{'runtime_environment':{}},"
              "'samples':[{'split':'train','sequence':'s','frame':1}]}; build_plan(m,'a'*64)")
    base = {**os.environ, 'ADESEG_BOUNDARY_MULTISEED': str(SEEDS[0]), 'PYTHONHASHSEED': '0'}
    assert subprocess.run([sys.executable, '-c', script], env=base, capture_output=True).returncode == 0
    for key in ('ADESEG_BOUNDARY_MULTISEED', 'PYTHONHASHSEED'):
        env = dict(base); env.pop(key)
        assert subprocess.run([sys.executable, '-c', script], env=env, capture_output=True).returncode != 0


def _write(path, data):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data))
    return path


def _make_joint(tmp_path, monkeypatch):
    monkeypatch.setattr(gate, 'validate_output', lambda p: Path(p))
    monkeypatch.setattr(gate, 'dependencies', lambda: {'source': 'a'*64})
    monkeypatch.setattr(gate, 'mask_path_sets', lambda _: (set(), set()))
    manifest = _write(tmp_path/'manifest/manifest.json', {'passed': True})
    _write(tmp_path/'manifest/completion.json', {'passed': True, 'manifest_sha256': digest(manifest)})
    for seed in SEEDS:
        base = tmp_path/str(seed)
        plan = _write(base/'plan/plan.json', {'passed': True, 'seed': seed, 'arms': list(ARMS),
            'python_hash_seed': '0', 'study_seed_environment': str(seed), 'training_order': [[seed]],
            'manifest_sha256': digest(manifest), 'dependency_sha256': {'source': 'a'*64}})
        parity = _write(base/'parity/parity.json', {'passed': True, 'manifest_sha256': digest(manifest),
                                                    'plan_sha256': digest(plan)})
        arms = {}
        for arm in ARMS:
            folder = base/'runs'/arm
            completion = _write(folder/'completion.json', {'passed': True})
            checkpoint = _write(folder/'checkpoint.pt', {})
            frozen = _write(folder/'frozen_predictions.json', {})
            arms[arm] = {'completion_path': str(completion.resolve()), 'completion_sha256': digest(completion),
                         'checkpoint_sha256': digest(checkpoint), 'frozen_sha256': digest(frozen)}
        verification = _write(base/'verification/verification.json', {'passed': True,
            'manifest_sha256': digest(manifest), 'plan_sha256': digest(plan), 'parity_sha256': digest(parity),
            'dependency_sha256': {'source': 'a'*64}, 'label_guard': {'open_attempts': 0, 'blocked_paths': 0},
            'runner_sha256': digest(gate.ROOT/'scripts/verify_source_group_boundary_multiseed_v1.py'), 'runs': arms})
        _write(base/'verification/completion.json', {'passed': True, 'verification_sha256': digest(verification)})
    evidence = gate.collect(tmp_path)
    path = _write(tmp_path/'joint_gate/joint.json', {'passed': True, 'runner_sha256': digest(Path(gate.__file__)),
                   'dependency_sha256': {'source': 'a'*64}, 'evidence': evidence})
    _write(path.parent/'completion.json', {'passed': True, 'joint_sha256': digest(path)})
    return path


def test_joint_gate_rejects_other_seed_tampering(tmp_path, monkeypatch):
    path = _make_joint(tmp_path, monkeypatch)
    assert gate.check_joint_gate(path, tmp_path)['passed']
    (tmp_path/str(SEEDS[1])/'runs'/ARMS[0]/'checkpoint.pt').write_text('tampered')
    with pytest.raises(ValueError, match='replayed artifacts'):
        gate.check_joint_gate(path, tmp_path)


def test_joint_gate_rejects_identical_orders(tmp_path, monkeypatch):
    path = _make_joint(tmp_path, monkeypatch)
    first = tmp_path/str(SEEDS[0])/'plan/plan.json'
    second = tmp_path/str(SEEDS[1])/'plan/plan.json'
    data = json.loads(second.read_text())
    data['training_order'] = json.loads(first.read_text())['training_order']
    _write(second, data)
    with pytest.raises(ValueError, match='plan, parity or replay'):
        gate.check_joint_gate(path, tmp_path)


def test_score_refuses_missing_joint_gate_before_label(tmp_path, monkeypatch):
    monkeypatch.setattr(scorer, 'ensure_target', lambda *_: pytest.fail('label opened'))
    monkeypatch.setattr(gate, 'validate_output', lambda p: Path(p))
    argv = ['score']
    for key, value in [('manifest', 'm'), ('plan', 'p'), ('parity', 'q'), ('verification', 'v'),
                       ('run-root', str(tmp_path/str(SEEDS[0])/'runs')), ('output', 'o'),
                       ('joint-gate', str(tmp_path/'joint_gate/joint.json'))]:
        argv.extend(['--'+key, value])
    monkeypatch.setattr(sys, 'argv', argv)
    with pytest.raises(FileNotFoundError):
        scorer.main()


def test_summary_requires_all_audits_and_identical_sample_provenance(tmp_path, monkeypatch):
    monkeypatch.setattr(report, 'validate_output', lambda p: Path(p))
    old, run = tmp_path/'old', tmp_path/'new'
    monkeypatch.setattr(report, 'PRIOR', old)
    monkeypatch.setattr(report, 'PRIOR_AUDIT', old/'audit.json')
    rows = [{'split': 'train' if i < 164 else 'validation', 'sequence': 's', 'frame': i,
             'image_sha256': 'a', 'label_sha256': 'b', 'feature_sha256': 'c', 'snapshot_sha256': 'd'}
            for i in range(214)]
    identities = {k: k for k in ('checkpoint_sha256', 'feature_pipeline_sha256',
                  'source_freeze_sha256', 'preflight_sha256', 'short_gate_sha256')}
    original_manifest = _write(old/'manifest/manifest.json', {'samples': rows, **identities})
    _write(run/'manifest/manifest.json', {'samples': rows, **identities})
    old_score = _write(old/'scores/scores.json', {'passed': True, 'summary': {'baseline': {}}})
    _write(old/'audit.json', {'passed': True, 'scores_sha256': digest(old_score),
                               'actual_manifest_sha256': digest(original_manifest)})
    for seed in SEEDS:
        score = _write(run/str(seed)/'scores/scores.json', {'passed': True, 'summary': {'baseline': {}}})
        audit = _write(run/str(seed)/'independent_audit/audit.json', {'passed': True, 'seed': seed,
            'scores_sha256': digest(score), 'error_count': 0, 'actual_manifest_sha256': digest(run/'manifest/manifest.json'),
            'mask_files_checked': 600, 'validation_frames': 50})
        _write(audit.parent/'completion.json', {'passed': True, 'audit_sha256': digest(audit)})
    assert set(report.collect(run)) == {'20261115', '20261125', '20261135'}
    changed = json.loads((run/'manifest/manifest.json').read_text())
    changed['samples'][0]['feature_sha256'] = 'different'
    _write(run/'manifest/manifest.json', changed)
    with pytest.raises(ValueError, match='sample identities'):
        report.collect(run)


def test_independent_audit_checks_launcher_order_and_environment(tmp_path, monkeypatch):
    monkeypatch.setattr(audit, 'ROOT', tmp_path)
    launcher = _write(tmp_path/'scripts/run_source_group_boundary_multiseed_v1.py', {})
    discovery = _write(tmp_path/'scripts/resume_source_group_capture_v1.py', {})
    map_path = _write(tmp_path/'run/capture_map.json', {})
    fixed = {'PYTHONHASHSEED': '0', 'PYTORCH_ENABLE_MPS_FALLBACK': '0', 'PYTORCH_MPS_FAST_MATH': '0'}
    record = _write(tmp_path/'run/run_manifest.json', {'launcher_sha256': digest(launcher),
        'discovery_sha256': digest(discovery), 'seeds': list(SEEDS), 'fixed_child_environment': fixed,
        'capture_map_sha256': digest(map_path), 'dependency_sha256': {'scripts/resume_source_group_capture_v1.py': digest(discovery)}})
    phases = [('manifest', None)]
    for seed in SEEDS:
        phases += [(f'{seed}_plan', seed), (f'{seed}_parity', seed)]
        phases += [(f'{seed}_train_{arm}', seed) for arm in ARMS]
        phases.append((f'{seed}_verify', seed))
    phases += [('joint_gate', None)] + [(f'{seed}_score', seed) for seed in SEEDS]
    events = []
    for name, seed in phases:
        event = {'phase': name, 'seed': seed, 'launcher_sha256': digest(launcher),
                 'run_manifest_sha256': digest(record), 'child_environment': {**fixed,
                     'ADESEG_BOUNDARY_MULTISEED': str(seed) if seed is not None else None}}
        events += [{**event, 'status': 'started'}, {**event, 'status': 'passed', 'returncode': 0}]
    path = tmp_path/'run/events.jsonl'
    path.write_text(''.join(json.dumps(row)+'\n' for row in events))
    assert audit.check_launcher(path.parent)['run_manifest_sha256'] == digest(record)
    events[2]['child_environment']['PYTHONHASHSEED'] = 'random'
    path.write_text(''.join(json.dumps(row)+'\n' for row in events))
    with pytest.raises(ValueError, match='phase'):
        audit.check_launcher(path.parent)


def test_statically_imported_project_sources_are_bound():
    root = Path(__file__).resolve().parents[1]
    sources = dependencies()
    pending = [Path(k) for k in sources if k.endswith('.py')]
    visited = set()
    while pending:
        current = pending.pop()
        if current in visited:
            continue
        visited.add(current)
        tree = ast.parse((root/current).read_text())
        for node in ast.walk(tree):
            if isinstance(node, ast.Import):
                names = [item.name for item in node.names]
            elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
                names = [node.module]
            else:
                continue
            for name in names:
                if name.split('.')[0] not in {'scripts', 'modeling', 'datasets'}:
                    continue
                path = Path(*name.split('.')).with_suffix('.py')
                if (root/path).is_file():
                    assert str(path) in sources, f'Unbound local dependency: {path}'
                    pending.append(path)
