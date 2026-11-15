"""Exploratory, provenance-bound live runtime for a boundary-study checkpoint.

The user explicitly requested A/B/C/D video inference after observing the
fixed-read study. This is not the old promotion gate, independent validation,
or a claim that the contrast component passed a mechanism screen.
"""
from contextlib import ExitStack, contextmanager
import gc
import json
import os
from pathlib import Path
import sys
from types import SimpleNamespace

import torch

from datasets.video import load_manifest
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_live_state_audit import module_tensor_fingerprints
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.lean_verified_runtime_lease import _FACTORY_ISSUER, _RuntimeLease
from modeling.lean_verified_window_runtime import _same_model_baseline, _mps_hardware_inventory
from scripts.memory_token_io import digest, validate_output
from scripts.freeze_source_group_runtime_v1 import current_identity
from scripts.source_group_boundary_shared_v1 import ARMS, dependencies, load_prepared, model_for
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.train_source_group_boundary_v1 import require_plan, require_parity
from scripts.verify_source_group_boundary_v1 import check_frozen_arm
from scripts.verify_token_window_refiner_routing_mps import build_model, verify_archived_baseline
from scripts.run_author_baseline import build_author_predictor

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / 'external/Lean-SAM2-token-window'


def verify_live_identity(manifest, plan):
    """Capture actual environment/fork evidence and compare to training freeze."""
    live_environment = environment_identity()
    if live_environment != plan['environment']:
        raise ValueError('Live execution environment differs from MPS captures')
    actual = current_identity(Path(manifest['preflight_path']).parent,
                              Path(manifest['reference_capture_path']), Path(manifest['checkpoint_path']))
    saved = json.loads(Path(manifest['source_freeze_path']).read_text())
    if not saved.get('passed') or actual != saved.get('identity'):
        raise ValueError('Live fork/source identity differs from training capture freeze')
    return live_environment, actual


def verify_boundary_study(study, audit, checkpoint):
    """Verify saved, already-scored development artifacts, without label reads."""
    study, audit = Path(study).resolve(), Path(audit).resolve()
    manifest, manifest_sha = load_prepared(study / 'manifest')
    plan_path, parity_path = study / 'plan/plan.json', study / 'parity/parity.json'
    plan = require_plan(plan_path, manifest, manifest_sha)
    require_parity(parity_path, plan, plan_path, manifest_sha)
    if digest(checkpoint) != manifest['checkpoint_sha256']:
        raise ValueError('Backbone checkpoint differs from training pipeline')
    verify_live_identity(manifest, plan)
    runs = {arm: check_frozen_arm(study/'runs'/arm, arm, manifest, manifest_sha,
                                  plan_path, parity_path, plan) for arm in ARMS}
    ver_path = study/'verification/verification.json'
    ver_done = json.loads((study/'verification/completion.json').read_text())
    ver = json.loads(ver_path.read_text())
    score_path = study/'scores/scores.json'
    score = json.loads(score_path.read_text())
    review = json.loads(audit.read_text())
    expected = {'actual_manifest_sha256': manifest_sha, 'recorded_manifest_sha256': manifest_sha,
                'plan_sha256': digest(plan_path), 'parity_sha256': digest(parity_path),
                'verification_sha256': digest(ver_path), 'scores_sha256': digest(score_path)}
    if (not ver_done.get('passed') or ver_done.get('verification_sha256') != digest(ver_path)
            or not ver.get('passed') or ver.get('manifest_sha256') != manifest_sha
            or ver.get('plan_sha256') != digest(plan_path) or ver.get('parity_sha256') != digest(parity_path)
            or ver.get('dependency_sha256') != dependencies() or ver.get('label_guard', {}).get('open_attempts') != 0
            or not score.get('passed') or score.get('manifest_sha256') != manifest_sha
            or score.get('plan_sha256') != digest(plan_path) or score.get('verification_sha256') != digest(ver_path)
            or not review.get('passed') or review.get('error_count') != 0 or review.get('mask_files_checked') != 600
            or Path(review.get('study_path', '')).resolve() != study
            or any(review.get(k) != v for k, v in expected.items())
            or review.get('audit_script_sha256') != digest(ROOT/'scripts/audit_source_group_boundary_scores_v1.py')):
        raise ValueError('Boundary study replay, score or independent audit differs')
    for arm, run in runs.items():
        rv = review.get('arm_provenance', {}).get(arm, {})
        vv = ver.get('runs', {}).get(arm, {})
        expected_arm = {'checkpoint_sha256': digest(run['folder']/'checkpoint.pt'),
                        'completion_sha256': digest(run['folder']/'completion.json'),
                        'frozen_sha256': digest(run['folder']/'frozen_predictions.json')}
        if any(rv.get(k) != v or vv.get(k) != v for k,v in expected_arm.items()):
            raise ValueError(f'{arm}: independent audit does not bind the frozen run')
    return manifest, manifest_sha, plan, runs['region_boundary']['payload'], {
        'study': str(study), 'audit_sha256': digest(audit), 'manifest_sha256': manifest_sha,
        'plan_sha256': digest(plan_path), 'parity_sha256': digest(parity_path),
        'verification_sha256': digest(ver_path), 'scores_sha256': digest(score_path),
        'candidate_checkpoint_sha256': digest(study/'runs/region_boundary/checkpoint.pt'),
        'control_checkpoint_sha256': digest(study/'runs/mask_only_boundary/checkpoint.pt'),
        'same_objective_no_contrast_sha256': digest(study/'runs/no_contrast_boundary/checkpoint.pt'),
        'selection': 'Explicit user-requested exploratory run; old promotion rule not passed or asserted.',
    }


@contextmanager
def verified_boundary_mps_window(*, study, audit, split_preflight, checkpoint, baseline_reference,
                                 seq4_capture, output):
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0' or os.environ.get('PYTORCH_MPS_FAST_MATH') != '0':
        raise ValueError('MPS fallback and fast math must be disabled')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    out = validate_output(Path(output)); out.mkdir(parents=True, exist_ok=False)
    training, manifest_sha, plan, payload, evidence = verify_boundary_study(study, audit, checkpoint)
    live_environment, live_identity = verify_live_identity(training, plan)
    pre_path = Path(split_preflight)/'manifest.json'
    pre = json.loads(pre_path.read_text())
    done = json.loads((Path(split_preflight)/'completion.json').read_text())
    if not done.get('passed') or done.get('manifest_sha256') != digest(pre_path):
        raise ValueError('Split preflight changed')
    source = pre['sequences']['seq4']; prompt_dir = Path(source['source'])/'seq4'
    prompt_path = prompt_dir/'prompt.json'
    if digest(prompt_path) != source['prompt_sha256']:
        raise ValueError('Seq4 prompt changed')
    prompt = json.loads(prompt_path.read_text())
    if prompt['first_detection_index'] != 9 or prompt['points'] is None:
        raise ValueError('Unexpected archived prompt')
    videos = {**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),
              **load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
    frames = videos['seq4']; metadata = json.loads((prompt_dir/'frames.json').read_text())
    if len(frames) != len(metadata) or any(digest(f.image_path) != metadata[i]['image_sha256'] for i,f in enumerate(frames)):
        raise ValueError('Seq4 source images changed')
    sys.path.insert(0, str(FORK))
    cpu = build_author_predictor(SimpleNamespace(method='lean', checkpoint=Path(checkpoint), device='cpu'), FORK)
    expected_model = module_tensor_fingerprints(cpu); del cpu; gc.collect()
    model, runtime, selections = build_model(Path(checkpoint), expected_model)
    refiner = model_for('region_boundary')
    refiner.load_trained_state(payload, backbone_sha256=training['checkpoint_sha256'],
                               feature_pipeline_sha256=training['feature_pipeline_sha256'],
                               train_manifest_sha256=manifest_sha)
    refiner.to('mps').eval().requires_grad_(False)
    expected_refiner = module_tensor_fingerprints(refiner)
    positions = []
    with ExitStack() as stack:
        stack.enter_context(indexed_spatial_rope_mps(model, positions))
        stack.enter_context(blocking_mps_pointer_position(model))
        for attempt in (1,2):
            torch.manual_seed(20261021); torch.mps.manual_seed(20261021)
            replay = _same_model_baseline(model, runtime, frames, prompt, out/f'baseline_{attempt}', 13)
            verify_archived_baseline(replay, Path(baseline_reference), checkpoint=Path(checkpoint),
                                     prompt=prompt_path, seq4_capture=Path(seq4_capture), end=13)
            if module_tensor_fingerprints(model) != expected_model:
                raise RuntimeError('Predictor tensor state changed during baseline replay')
            runtime.events.clear(); positions.clear(); selections.clear()
        lease = _RuntimeLease(model, refiner, runtime, expected_model, expected_refiner,
                              _issuer=_FACTORY_ISSUER, positions=positions, selections=selections)
        lease.validate_live(model, refiner, tensors=True)
        mf, rf = out/'model_tensor_fingerprints.json', out/'refiner_tensor_fingerprints.json'
        mf.write_text(json.dumps(expected_model, indent=2)+'\n'); rf.write_text(json.dumps(expected_refiner, indent=2)+'\n')
        sources = (Path(__file__), ROOT/'modeling/lean_boundary_refinement_routing_v1.py',
                   ROOT/'modeling/lean_verified_runtime_lease.py', ROOT/'modeling/lean_verified_window_runtime.py',
                   ROOT/'scripts/run_boundary_refiner_priority_v1.py', ROOT/'scripts/verify_boundary_video_short_gate_v1.py',
                   ROOT/'scripts/run_verified_refiner_priority_development.py',
                   ROOT/'modeling/lean_contrast_refinement_integration.py', ROOT/'modeling/contrast_refinement_routing.py',
                   ROOT/'modeling/lean_indexed_spatial_rope_mps.py', ROOT/'modeling/lean_mps_blocking_pointer_position.py',
                   ROOT/'modeling/lean_multiscale_anchor.py', ROOT/'modeling/lean_token_window_storage.py',
                   ROOT/'modeling/token_memory_window.py', ROOT/'modeling/lean_token_storage_observer.py',
                   ROOT/'scripts/verify_token_window_refiner_routing_mps.py', ROOT/'scripts/run_author_baseline.py',
                   ROOT/'modeling/seq23_viewport.py', ROOT/'modeling/samurai_auto_prompt.py',
                   ROOT/'scripts/run_samurai_automatic_priority.py', ROOT/'scripts/memory_token_io.py',
                   ROOT/'datasets/video.py', ROOT/'datasets/io.py', ROOT/'data/polypgen_sequence.jsonl')
        record = {'passed': True, 'scope': 'User-requested exploratory video run, not promotion or independent test.',
                  **evidence, 'feature_pipeline_sha256': training['feature_pipeline_sha256'],
                  'checkpoint_sha256': digest(checkpoint), 'split_preflight_sha256': digest(pre_path),
                  'seq4_capture_sha256': digest(Path(seq4_capture)/'channel_feature_receipt.json'),
                  'training_dependencies': dependencies(), 'baseline_reference_sha256': digest(Path(baseline_reference)/'receipt.json'),
                  'model_tensor_fingerprints_sha256': digest(mf), 'refiner_tensor_fingerprints_sha256': digest(rf),
                  'source_sha256': {str(p.relative_to(ROOT)): digest(p) for p in sources},
                  'device': 'mps float32', 'environment': live_environment, 'live_source_identity': live_identity,
                  'gpu_inventory': _mps_hardware_inventory(), 'fallback': 'disabled', 'fast_math': 'disabled',
                  'seed': 20261021, 'baseline_replays': 2, 'memory_budget': 16384, 'insurance_reserve': 3072}
        (out/'runtime_verification.json').write_text(json.dumps(record, indent=2)+'\n')
        try:
            yield model, refiner, lease
            lease.validate_live(model, refiner, tensors=True)
        finally:
            lease.close()
