"""Fold-specific warm Lean/refiner runtime, bound to a video certificate.

Prepared for exploratory full-video evaluation only. No target labels are
opened and no detector, new threshold or new token-pruning rule is used.
"""
from contextlib import ExitStack, contextmanager
from pathlib import Path
from types import SimpleNamespace
import json
import sys

import numpy as np
import torch

from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_live_state_audit import module_tensor_fingerprints
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.lean_multiscale_anchor import install_multiscale_anchor
from modeling.lean_token_window_storage import TokenWindowRuntime
from modeling.lean_verified_runtime_lease import _FACTORY_ISSUER, _RuntimeLease
from scripts.channel_value_fixed_read_v1 import require_environment_flags
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import EXISTING, OUT, phase_artifact, validate_plan
from scripts.memory_token_io import digest
from scripts.run_author_baseline import build_author_predictor
from scripts.run_crossfit_warm_video_v2 import FORK, warm_plan
from scripts.run_source_group_capture_v1 import state_hash
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.warm_crossfit_fold_training_v1 import load_preflight
from scripts.warm_crossfit_video_contract_v1 import (fresh_sequence_inputs, load_heldout_checkpoint,
    old_sequence_inputs, read_bound, require)


def verify_video_certificate(path, expected_sha):
    from scripts.preflight_warm_crossfit_video_v1 import sources
    path = Path(path).resolve(strict=True)
    require(path.is_relative_to(OUT.resolve(strict=True)) and path.name == 'certificate.json',
            'Video certificate is outside experiment root')
    certificate = read_bound(path, expected_sha)
    done = json.loads((path.parent / 'completion.json').read_text())
    require(done.get('passed') is True and done.get('certificate_sha256') == expected_sha and
            done.get('label_guard', {}).get('open_attempts') == 0 and
            certificate.get('passed') is True and certificate.get('source_sha256') == sources() and
            certificate.get('label_guard', {}).get('open_attempts') == 0,
            'Video certificate/source binding differs')
    plan, sha = load_preflight(certificate['preflight_path'])
    require(sha == certificate['preflight_sha256'] and
            phase_artifact(certificate['crossfit_root'], 'verify',
                Path(certificate['crossfit_root']) / 'verify/verification.json') ==
                certificate['fixed_read_verification_sha256'], 'Fixed-read certificate changed')
    aggregate = read_bound(certificate['aggregate_path'], certificate['aggregate_sha256'])
    warm_certificate, warm_sha = verify_certificate(certificate['warm_capture_certificate_path'])
    require(warm_sha == certificate['warm_capture_certificate_sha256'] == aggregate['certificate_sha256'],
            'Warm capture certificate changed')
    manifest = read_bound(EXISTING / 'manifest.json', certificate['old_manifest_sha256'])
    return certificate, plan, sha, aggregate, warm_certificate, manifest


@contextmanager
def verified_warm_crossfit_runtime(certificate_path, certificate_sha, sequence, output):
    require_environment_flags(); require(torch.backends.mps.is_available(), 'MPS unavailable')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(2)
    certificate, plan, sha, aggregate, warm_certificate, manifest = verify_video_certificate(
        certificate_path, certificate_sha)
    require(sequence in certificate['references'], 'Patient missing from video certificate')
    warm = warm_plan(); index = validate_plan()
    require(environment_identity() == warm['old_identity']['runtime_environment'], 'Runtime environment differs')
    refiner, fold_proof = load_heldout_checkpoint(certificate['crossfit_root'], certificate['preflight_path'],
                                                   plan, sha, sequence)
    expected = certificate['references'][sequence]
    require(fold_proof == certificate['checkpoints'][str(expected['fold'])], 'Fold checkpoint certificate differs')
    info = (old_sequence_inputs(sequence, manifest, certificate['old_manifest_sha256'])
            if sequence in manifest['sequences'] else
            fresh_sequence_inputs(sequence, aggregate, warm_certificate, index, warm))
    if sequence in manifest['sequences']:
        require(info['reference'].get('identity') == warm['old_identity'],
                'Historical warm source/runtime identity differs from certified captures')
    require(info['reference_path'] == expected['reference_path'] and
            info['reference_sha256'] == expected['reference_sha256'] and info['start'] == expected['prompt_frame'] and
            len(info['frames']) == expected['frames'] and info['image_sha256'] == expected['image_sha256'] and
            [list(size) for size in info['image_sizes']] == expected['image_sizes'] and
            info['roi_xyxy'] == expected['roi_xyxy'], 'Live warm reference differs')
    checkpoint = Path(manifest['checkpoint_path']).resolve(strict=True)
    require(digest(checkpoint) == manifest['checkpoint_sha256'] == plan['backbone_sha256'], 'Lean checkpoint differs')
    torch.manual_seed(20261021); torch.mps.manual_seed(20261021); np.random.seed(20261021)
    sys.path.insert(0, str(FORK)); model = build_author_predictor(
        SimpleNamespace(method='lean', checkpoint=checkpoint, device='mps'), FORK)
    import sam2
    require(Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()), 'Wrong Lean checkout')
    require(model.num_maskmem == 7 and model.mem_keep_ratio == .25 and model.insurance_max_frames == 3 and
            model.enable_insurance_bank and model.fill_hole_area == 0, 'Lean configuration differs')
    model.ema_history_replace_old_tail_in_attn = False; model.ema_history_resident_prune = False
    model.token_window_runtime = TokenWindowRuntime(16384, 3072)
    selection = []; anchor = install_multiscale_anchor(model, .5, selection_events=selection)
    require(state_hash(model) == warm['old_identity']['model_state_sha256'], 'Lean model fingerprint differs')
    refiner = refiner.to('mps').eval().requires_grad_(False)
    model_fingerprint = module_tensor_fingerprints(model); refiner_fingerprint = module_tensor_fingerprints(refiner)
    positions = []; out = Path(output).resolve(); require(out.is_relative_to(OUT.resolve(strict=True)) and not out.exists(),
        'Invalid runtime output')
    out.mkdir(parents=True, exist_ok=False)
    with ExitStack() as stack:
        stack.enter_context(indexed_spatial_rope_mps(model, positions))
        stack.enter_context(blocking_mps_pointer_position(model))
        lease = _RuntimeLease(model, refiner, model.token_window_runtime, model_fingerprint, refiner_fingerprint,
                              _issuer=_FACTORY_ISSUER, positions=positions, selections=selection)
        lease.validate_live(model, refiner, tensors=True)
        rec = {'passed': True, 'scope': 'Fold-specific warm runtime; no video efficacy claim.',
               'certificate_sha256': certificate_sha, 'sequence': sequence, 'fold_checkpoint': fold_proof,
               'reference_sha256': info['reference_sha256'], 'source_sha256': digest(Path(__file__)),
               'device': 'mps', 'environment': environment_identity()}
        (out / 'runtime_verification.json').write_text(json.dumps(rec, indent=2) + '\n')
        try:
            yield model, refiner, lease, anchor, info
            lease.validate_live(model, refiner, tensors=True)
            require(digest(Path(certificate_path)) == certificate_sha, 'Video certificate changed')
        finally:
            lease.close()
