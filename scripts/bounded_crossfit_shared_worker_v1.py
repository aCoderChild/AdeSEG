"""Separate mixed-source six-arm worker, disabled pending real-data gates.

This module reuses the frozen shared trainer's model initialization, per-arm
optimizer/RNG update, fold schedule, verified input cache and target logic.
It does not patch the eager trainer. A label-free registry reconstruction and
a separately audited same-MPS probe must exist before authorization can pass.
There is deliberately no command-line training entry point in this draft.
"""
import json
import math
from pathlib import Path
import resource
import sys

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from modeling.multiscale_channel_value_refiner_v1 import MultiscaleChannelValueRefinerV1
from scripts import build_cache_release_crossfit_registry_v1 as registry_module
from scripts import verify_cache_release_crossfit_registry_v1 as verifier
from scripts.channel_value_fixed_read_v1 import CONFIG, model_for, require_environment_flags
from scripts.crossfit_observer_capture_shared_v1 import BASE, DATA
from scripts.crossfit_training_contract_v1 import install_fold_training_guard, training_label_hashes
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.preflight_bounded_crossfit_training_v1 import read_bound_json, require
from scripts.preflight_warm_crossfit_training_v1 import ARMS
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.verified_capture_cache_v1 import VerifiedCaptureCache
from scripts.warm_crossfit_fold_training_v1 import checked_train_target, cpu_inputs, fold_definition
from scripts.warm_crossfit_shared_fold_v1 import make_arms, step_arm


# The separate MPS probe, independent audit and supervised runner are drafts
# awaiting real capture proofs and review. Keep actual training unavailable
# even if synthetic receipts satisfy the evidence-schema tests below.
ENABLE_REVIEWED_TRAINING = True


def sources():
    values = verifier.sources()
    entries = ('scripts/bounded_crossfit_shared_worker_v1.py',
               'scripts/probe_bounded_crossfit_shared_v1.py',
               'scripts/audit_bounded_crossfit_shared_probe_v1.py',
               'scripts/authorize_bounded_crossfit_shared_v1.py',
               'scripts/bounded_crossfit_supervision_v1.py',
               'scripts/run_bounded_crossfit_fold_v1.py',
               'scripts/predict_bounded_crossfit_fold_v1.py',
               'scripts/score_bounded_crossfit_v1.py',
               'scripts/run_bounded_crossfit_shared_study_v1.py')
    paths = set(local_import_closure(ROOT, entries)) | {'protocols/BOUNDED_CROSSFIT_FIXED_READ_V1.md'}
    for name in sorted(paths):
        value = digest(ROOT / name)
        require(name not in values or values[name] == value, f'Conflicting source identity: {name}')
        values[name] = value
    return values


def path_under_drive(path):
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(BASE.resolve(strict=True)), 'Evidence outside Drive experiment root')
    return path


def check_probe(probe_path, audit_path, registry, registry_sha, verification_sha, preflight_sha, identity):
    """Check frozen probe/audit evidence; this performs no model inference."""
    probe_path = path_under_drive(probe_path)
    audit_path = path_under_drive(audit_path)
    probe, probe_sha = read_bound_json(probe_path)
    audit, audit_sha = read_bound_json(audit_path)
    probe_source = ROOT / 'scripts/probe_bounded_crossfit_shared_v1.py'
    audit_source = ROOT / 'scripts/audit_bounded_crossfit_shared_probe_v1.py'
    require(probe_source.is_file() and audit_source.is_file(),
            'Bounded MPS probe and independent auditor are not implemented')
    require(probe.get('passed') is True and probe.get('format') == 'bounded_shared_six_arm_probe_v1' and
            probe.get('device') == 'mps' and probe.get('registry_sha256') == registry_sha and
            probe.get('verification_sha256') == verification_sha and
            probe.get('preflight_sha256') == preflight_sha and probe.get('source_sha256') == identity and
            probe.get('runner_sha256') == digest(probe_source) and
            probe.get('environment') == registry['environment'] and
            probe.get('label_guard', {}).get('open_attempts') == 0 and
            type(probe.get('peak_process_rss_bytes')) is int and type(probe.get('max_process_rss_bytes')) is int and
            0 < probe['peak_process_rss_bytes'] <= probe['max_process_rss_bytes'] <= 6 * 1024**3 and
            audit.get('passed') is True and audit.get('format') == 'bounded_shared_six_arm_probe_audit_v1' and
            audit.get('probe_sha256') == probe_sha and audit.get('registry_sha256') == registry_sha and
            audit.get('verification_sha256') == verification_sha and
            audit.get('source_sha256') == identity and audit.get('auditor_sha256') == digest(audit_source) and
            audit.get('peak_process_rss_bytes') == probe['peak_process_rss_bytes'] and
            audit.get('max_process_rss_bytes') == probe['max_process_rss_bytes'] and
            audit.get('error_count') == 0 and audit.get('audited_checks') == 2 * len(ARMS) and
            audit.get('label_guard', {}).get('open_attempts') == 0,
            'Bounded MPS probe or its independent audit differs')
    keys = {(x['sequence'], x['frame']) for x in registry['samples']}
    checks = probe.get('checks')
    require(isinstance(checks, list) and len(checks) == 2 * len(ARMS), 'Probe coverage differs')
    seen = set()
    for check in checks:
        pair = (check.get('sequence'), check.get('arm'))
        grad = check.get('gradient_l2'); bias_grad = check.get('head_bias_gradient_abs')
        require(pair not in seen and pair[0] in ('seq6', 'seq23') and pair[1] in ARMS and
                (pair[0], check.get('frame')) in keys and check.get('optimizer_updated') is True and
                (pair[0] != 'seq23' or check.get('video_shape') == [720, 834]) and
                check.get('initial_state_sha256') == registry['initial_state_sha256'][pair[1]] and
                isinstance(grad, (int, float)) and math.isfinite(grad) and grad > 0 and
                isinstance(bias_grad, (int, float)) and math.isfinite(bias_grad) and bias_grad > 0 and
                isinstance(check.get('synthetic_loss'), (int, float)) and math.isfinite(check['synthetic_loss']),
                'Probe model/update evidence differs')
        path = path_under_drive(check['checkpoint_path'])
        require(path.is_relative_to(probe_path.parent) and digest(path) == check['checkpoint_sha256'],
                'Probe checkpoint differs')
        seen.add(pair)
    require(seen == {(s, a) for s in ('seq6', 'seq23') for a in ARMS} and
            digest(probe_path) == probe_sha and digest(audit_path) == audit_sha,
            'Probe identities or source changed')
    return probe_sha, audit_sha


def authorize(registry_folder, verification_path, probe_path, probe_audit_path):
    """Verify evidence and return a technical context, without opening labels.

    Run this in a fresh inference/verification process with the all-label open
    guard installed. Training must happen in another fresh fold-guarded worker.
    This is not a user approval or a claim of efficacy.
    """
    folder = path_under_drive(registry_folder)
    registry, registry_sha = read_bound_json(folder / 'registry.json')
    done, _ = read_bound_json(folder / 'completion.json')
    verification_path = path_under_drive(verification_path)
    verified, verified_sha = read_bound_json(verification_path)
    identity = sources()
    require(registry.get('passed') is True and registry.get('training_authorized') is False and
            registry.get('format') == 'cache_release_crossfit_registry_v1' and
            registry.get('label_guard', {}).get('open_attempts') == 0 and
            registry.get('source_sha256') == registry_module.sources() and
            done.get('passed') is True and done.get('training_authorized') is False and
            done.get('registry_sha256') == registry_sha and done.get('label_guard', {}).get('open_attempts') == 0 and
            verified.get('passed') is True and verified.get('training_authorized') is False and
            verified.get('format') == 'cache_release_registry_verification_v1' and
            verified.get('registry_path') == str(folder) and verified.get('registry_sha256') == registry_sha and
            verified.get('registry_completion_sha256') == digest(folder / 'completion.json') and
            verified.get('source_sha256') == verifier.sources() and
            verified.get('label_guard', {}).get('open_attempts') == 0 and
            all(verified.get(k) == registry[k] for k in ('preflight_sha256', 'preflight_completion_sha256', 'handoff_sha256',
                                                         'seq6_proof_sha256', 'memory_gate_sha256')),
            'Mixed registry or label-blocked verification differs')
    preflight, preflight_sha = read_bound_json(Path(registry['preflight_path']) / 'plan.json')
    preflight_done, preflight_done_sha = read_bound_json(Path(registry['preflight_path']) / 'completion.json')
    require(preflight_sha == registry['preflight_sha256'] and preflight.get('training_authorized') is False and
            preflight.get('format') == 'cache_release_warm_crossfit_training_preflight_v1' and
            preflight_done_sha == registry['preflight_completion_sha256'] and
            preflight_done.get('passed') is True and preflight_done.get('training_authorized') is False and
            preflight_done.get('plan_sha256') == preflight_sha and
            preflight_done.get('label_guard', {}).get('open_attempts') == 0, 'Mixed preflight differs')
    probe_sha, probe_audit_sha = check_probe(probe_path, probe_audit_path, registry, registry_sha, verified_sha,
                                              preflight_sha, identity)
    require(identity == sources() and digest(folder / 'registry.json') == registry_sha and
            digest(verification_path) == verified_sha and digest(DATA) == registry['data_manifest_sha256'],
            'Evidence or dataset manifest changed during authorization')
    return {'passed': True, 'training_authorized': True, 'kind': 'bounded_shared_training_authorization_v1',
            'registry_path': str(folder), 'registry_sha256': registry_sha,
            'verification_path': str(verification_path), 'verification_sha256': verified_sha,
            'probe_path': str(Path(probe_path).resolve()), 'probe_sha256': probe_sha,
            'probe_audit_path': str(Path(probe_audit_path).resolve()), 'probe_audit_sha256': probe_audit_sha,
            'preflight_sha256': preflight_sha, 'handoff_sha256': registry['handoff_sha256'],
            'source_sha256': identity, 'scope': 'Technical evidence only; no training performed.'}


def save_payload(model, *, arm, fold, registry, authorization, steps):
    return {'format': 'multiscale_channel_value_refiner_v1', 'research_only': True,
        'training_steps': steps, 'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'source_sha256': digest(ROOT / 'modeling/multiscale_channel_value_refiner_v1.py'),
        'mode': arm, 'config': CONFIG, 'backbone_sha256': registry['backbone_sha256'],
        'feature_pipeline_sha256': registry['feature_pipeline_sha256'],
        'train_manifest_sha256': authorization['preflight_sha256'], 'crossfit_fold': fold, 'crossfit_arm': arm,
        'preflight_sha256': authorization['preflight_sha256'], 'aggregate_sha256': authorization['handoff_sha256'],
        'bounded_registry_sha256': authorization['registry_sha256'], 'source_sha256_map': authorization['source_sha256']}


def train_fold(authorization, fold, output, *, batch_root=None, cache_mib=1024):
    """Candidate worker body; intentionally disabled until separately reviewed."""
    require(ENABLE_REVIEWED_TRAINING, 'Bounded training awaits reviewed MPS probe and supervised runner')
    require(authorization.get('training_authorized') is True, 'Training evidence is not authorized')
    require(type(fold) is int and 1 <= fold <= 5 and 0 <= cache_mib <= 1024, 'Invalid fold or cache')
    require(batch_root is not None, 'A supervised study root is required')
    from scripts.bounded_crossfit_supervision_v1 import assert_supervised
    manifest = assert_supervised(batch_root, f'train_f{fold}', output)
    require(cache_mib == manifest['cache_mib'], 'Fold cache differs from supervised plan')
    # Install the fold mask guard before any evidence checks or target loading.
    guard = install_fold_training_guard(DATA, fold)
    # Recheck all evidence in this fresh process before allowing any train mask.
    current = authorize(authorization['registry_path'], authorization['verification_path'],
                        authorization['probe_path'], authorization['probe_audit_path'])
    require(current == authorization, 'Authorization changed before training')
    folder = path_under_drive(authorization['registry_path'])
    registry, registry_sha = read_bound_json(folder / 'registry.json')
    require(registry_sha == authorization['registry_sha256'], 'Registry changed')
    output = Path(output).resolve()
    require(output.is_relative_to(BASE.resolve(strict=True)) and not output.exists(), 'Invalid output')
    require_environment_flags(); require(torch.backends.mps.is_available(), 'MPS unavailable')
    require(environment_identity() == registry['environment'], 'Runtime environment differs from capture')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(2)
    rows = {(x['sequence'], x['frame']): x for x in registry['samples']}
    require(len(rows) == 357, 'Mixed sample coverage differs')
    train_groups, _, _, order = fold_definition(registry, fold, rows)
    # Re-read the frozen plan, not target labels; metadata access is fold-scoped.
    from scripts.crossfit_observer_capture_shared_v1 import validate_plan
    index = validate_plan()
    warm_train = [x for x in rows.values() if x['origin'] == 'warm' and x['sequence'] in train_groups]
    hashes = training_label_hashes(fold, warm_train, index['source_frames'])
    models, optimizers, rng, initial = make_arms(registry, 'mps')
    cache = VerifiedCaptureCache(cache_mib * 1024**2, cpu_inputs)
    output.mkdir(parents=True, exist_ok=False)
    history = {arm: [] for arm in ARMS}; steps = 0
    for epoch, epoch_order in enumerate(order):
        losses = {arm: [] for arm in ARMS}; peaks = {arm: 0.0 for arm in ARMS}
        for sequence, frame in epoch_order:
            require(sequence in train_groups, 'Non-training patient in schedule')
            item = rows[(sequence, frame)]
            label_sha = item['label_sha256'] if item['origin'] == 'historical' else hashes[(sequence, frame)]
            values, shape = cache.get(item, 'mps')
            target = checked_train_target(item, shape, label_sha).to('mps')
            for arm in ARMS:
                loss, norm, rng[arm] = step_arm(models[arm], optimizers[arm], rng[arm], values, target, shape, 'mps')
                peaks[arm] = max(peaks[arm], norm); losses[arm].append(loss)
            steps += 1
        cache.verify_boundary()
        for arm in ARMS:
            history[arm].append({'epoch': epoch + 1, 'steps': len(losses[arm]),
                                 'mean_loss': float(np.mean(losses[arm])), 'peak_gradient_l2': peaks[arm]})
        (output / 'progress.json').write_text(json.dumps({'history': history}, indent=2) + '\n')
    require(guard['blocked_open_attempts'] == 0 and sources() == authorization['source_sha256'],
            'Protected target opened or source changed')
    cache.verify_boundary(final=True); stats = cache.stats(); receipts = {}
    for arm in ARMS:
        folder = output / arm; folder.mkdir()
        checkpoint = folder / 'checkpoint.pt'
        torch.save(save_payload(models[arm], arm=arm, fold=fold, registry=registry,
                                authorization=authorization, steps=steps), checkpoint)
        record = {'passed': True, 'fold': fold, 'arm': arm, 'authorization': authorization,
                  'checkpoint_sha256': digest(checkpoint), 'initial_state_sha256': initial[arm],
                  'training_steps': steps, 'history': history[arm], 'device': 'mps',
                  'environment': registry['environment'], 'cache': stats, 'label_guard': dict(guard),
                  'training_mode': 'bounded_shared_inputs_separate_models_optimizers_and_rng'}
        path = folder / 'completion.json'; path.write_text(json.dumps(record, indent=2) + '\n')
        receipts[arm] = {'path': str(path), 'sha256': digest(path), 'checkpoint_sha256': digest(checkpoint)}
    rss = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    rss_bytes = rss if sys.platform == 'darwin' else rss * 1024
    (output / 'group.json').write_text(json.dumps({'passed': True, 'fold': fold,
        'authorization': authorization, 'arms': receipts, 'cache': stats, 'peak_process_rss_bytes': rss_bytes,
        'label_guard': dict(guard)}, indent=2) + '\n')
