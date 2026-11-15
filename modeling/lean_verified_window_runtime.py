"""Verifier-owned MPS token-window runtime for an eligible refiner.

This context performs a conservative development-selection audit and two
same-predictor baseline replays before issuing a scoped lease. It intentionally
does not itself score or select an evaluation cohort. Full-video users must
also enforce their sequence/prompt/crop protocol and freeze predictions before
opening labels. Python in-process hooks are not a security boundary.
"""
from contextlib import ExitStack, contextmanager
import gc
import json
import os
from pathlib import Path
import platform
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import numpy as np
from PIL import Image
import torch

from datasets.video import load_manifest, stage_frames
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_live_state_audit import module_tensor_fingerprints
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.lean_verified_runtime_lease import _FACTORY_ISSUER, _RuntimeLease
from modeling.foreground_conditioned_refiner import ForegroundConditionedRefiner
from modeling.multilevel_contrast_refiner import MultilevelContrastRefiner
from scripts.assess_refiner_candidate_promotion import verify as assess_candidate
from scripts.assess_region_conditioned_candidate import assess as assess_region_candidate
from scripts.memory_token_io import digest, validate_output
from scripts.verify_refiner_capture_static import verify as verify_capture
from scripts.verify_token_window_refiner_routing_mps import build_model, verify_archived_baseline
from scripts.run_author_baseline import build_author_predictor

ROOT = Path(__file__).resolve().parents[1]
FORK = ROOT / 'external/Lean-SAM2-token-window'


def _mps_hardware_inventory():
    """Record GPU models only; do not include display serial information."""
    raw = subprocess.check_output(['system_profiler', 'SPDisplaysDataType', '-json'], text=True)
    data = json.loads(raw)
    models = sorted(({'name': entry.get('sppci_model', entry.get('_name')),
                      'cores': entry.get('sppci_cores')}
                     for entry in data.get('SPDisplaysDataType', [])
                     if entry.get('sppci_device_type') == 'spdisplays_gpu'),
                    key=lambda entry: str(entry['name']))
    if not models or any(not entry['name'] for entry in models):
        raise ValueError('Cannot record the available GPU model')
    return models


def _same_model_baseline(model, runtime, frames, prompt, out, end):
    """Capture only baseline masks/events using the existing active MPS hooks."""
    start = prompt['first_detection_index']
    rows = []
    out.mkdir()
    with tempfile.TemporaryDirectory(prefix='live_baseline_', dir=out) as temporary:
        stage_frames(frames, Path(temporary))
        state = model.init_state(video_path=temporary, offload_video_to_cpu=True,
                                 offload_state_to_cpu=False)
        try:
            with torch.inference_mode():
                _, ids, _ = model.add_new_points_or_box(
                    state, frame_idx=start, obj_id=1, box=np.asarray(prompt['box'], np.float32))
                _, ids, logits = model.add_new_points_or_box(
                    state, frame_idx=start, obj_id=1,
                    points=np.asarray(prompt['points'], np.float32),
                    labels=np.array([1, 0], np.int32), clear_old_points=False)
                initial = logits[0, 0].cpu().numpy() > 0
                if ids != [1] or not initial.any():
                    raise ValueError('Invalid archived automatic prompt')
                model.add_new_mask(state, frame_idx=start, obj_id=1, mask=initial)
                for i, ids, logits in model.propagate_in_video(
                        state, start_frame_idx=start, max_frame_num_to_track=end-start, reverse=False):
                    if ids != [1] or i != start + len(rows):
                        raise ValueError('Unexpected baseline frame order')
                    path = out / f'{i:06d}.png'
                    Image.fromarray((logits[0, 0].cpu().numpy() > 0).astype(np.uint8) * 255).save(path)
                    rows.append({'frame': i, 'mask_sha256': digest(path)})
        finally:
            model.reset_state(state)
    if len(rows) != end-start+1:
        raise ValueError('Incomplete same-model baseline')
    return {'rows': rows, 'token_window_events': list(runtime.events)}


@contextmanager
def verified_mps_window_refiner(*, preflight, split_preflight, static_capture, seq4_capture,
                                checkpoint, detector, baseline_reference, candidate_run,
                                control_run, data_preflight, runner_source, output,
                                global_run=None, shuffled_run=None):
    """Construct, validate and temporarily own the exact model/refiner pair.

    The development screen is recomputed from frozen masks, not accepted from a
    caller-provided pass flag. No currently failed pilot is eligible. This
    context only validates seq4 and is not an independent evaluation.
    """
    if not torch.backends.mps.is_available():
        raise RuntimeError('MPS unavailable')
    if os.environ.get('PYTORCH_ENABLE_MPS_FALLBACK') != '0' or os.environ.get('PYTORCH_MPS_FAST_MATH') != '0':
        raise ValueError('Explicitly disable MPS fallback and fast math')
    torch.use_deterministic_algorithms(True)
    torch.set_num_threads(2)
    out = validate_output(Path(output))
    out.mkdir(parents=True, exist_ok=False)
    payload = torch.load(Path(candidate_run) / 'checkpoint.pt', map_location='cpu', weights_only=True)
    model_format = payload.get('format')
    if model_format == 'multilevel_contrast_refiner_v1':
        if global_run is not None or shuffled_run is not None:
            raise ValueError('Unexpected controls for this checkpoint format')
        assess_args = SimpleNamespace(data_preflight=Path(data_preflight), candidate_run=Path(candidate_run),
                                      control_run=Path(control_run), runner_source=Path(runner_source))
        selection = assess_candidate(assess_args)
        additional_controls = {}
    elif model_format == 'foreground_conditioned_refiner_v1':
        if global_run is None or shuffled_run is None:
            raise ValueError('Region-conditioned screening requires matching global and shuffled controls')
        if digest(Path(runner_source)) != payload.get('runner_source_sha256'):
            raise ValueError('Region-conditioned training runner differs')
        selection = assess_region_candidate(Path(data_preflight), Path(candidate_run), Path(global_run),
                                             Path(shuffled_run), Path(control_run), out / 'region_screen')
        additional_controls = {'global': digest(Path(global_run) / 'checkpoint.pt'),
                               'shuffled': digest(Path(shuffled_run) / 'checkpoint.pt')}
    else:
        raise ValueError('Unsupported refiner checkpoint format')
    (out / 'development_screen.json').write_text(json.dumps(selection, indent=2) + '\n')
    if not selection['decision']['eligible']:
        raise ValueError('Candidate did not pass the matched development screen')
    static = verify_capture(Path(preflight), Path(split_preflight), Path(static_capture),
                            Path(checkpoint), Path(detector))
    training = json.loads((Path(data_preflight) / 'manifest.json').read_text())
    if training['feature_pipeline_sha256'] != static['feature_pipeline_sha256']:
        raise ValueError('Training capture pipeline differs')
    seq4 = json.loads((Path(seq4_capture) / 'channel_feature_receipt.json').read_text())
    if seq4['feature_pipeline_sha256'] != static['feature_pipeline_sha256']:
        raise ValueError('Seq4 capture pipeline differs')
    pre = json.loads((Path(split_preflight) / 'manifest.json').read_text())
    source = pre['sequences']['seq4']
    prompt_dir = Path(source['source']) / 'seq4'
    prompt_path = prompt_dir / 'prompt.json'
    if digest(prompt_path) != source['prompt_sha256']:
        raise ValueError('Prompt changed')
    prompt = json.loads(prompt_path.read_text())
    if prompt['first_detection_index'] != 9 or prompt['points'] is None:
        raise ValueError('Unexpected archived prompt')
    videos = {**load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'dev'),
              **load_manifest(ROOT / 'data/polypgen_sequence.jsonl', 'test')}
    frames = videos['seq4']
    frame_metadata = json.loads((prompt_dir / 'frames.json').read_text())
    if len(frames) != len(frame_metadata) or any(
            digest(frame.image_path) != frame_metadata[i]['image_sha256'] for i, frame in enumerate(frames)):
        raise ValueError('Seq4 image source changed')
    sys.path.insert(0, str(FORK))
    cpu_model = build_author_predictor(SimpleNamespace(method='lean', checkpoint=Path(checkpoint), device='cpu'), FORK)
    expected_model = module_tensor_fingerprints(cpu_model)
    del cpu_model
    gc.collect()
    model, runtime, selections = build_model(Path(checkpoint), expected_model)
    refiner = (ForegroundConditionedRefiner(gate_mode='region') if
               model_format == 'foreground_conditioned_refiner_v1' else MultilevelContrastRefiner())
    refiner.load_trained_state(payload, backbone_sha256=training['checkpoint_sha256'],
                               feature_pipeline_sha256=training['feature_pipeline_sha256'],
                               train_manifest_sha256=digest(Path(data_preflight) / 'manifest.json'))
    refiner.to('mps').eval().requires_grad_(False)
    expected_refiner = module_tensor_fingerprints(refiner)
    positions = []
    lease = None
    with ExitStack() as stack:
        stack.enter_context(indexed_spatial_rope_mps(model, positions))
        stack.enter_context(blocking_mps_pointer_position(model))
        for attempt in (1, 2):
            torch.manual_seed(20261021)
            torch.mps.manual_seed(20261021)
            replay = _same_model_baseline(model, runtime, frames, prompt, out / f'baseline_{attempt}', 13)
            verify_archived_baseline(replay, Path(baseline_reference), checkpoint=Path(checkpoint),
                                     prompt=prompt_path, seq4_capture=Path(seq4_capture), end=13)
            if module_tensor_fingerprints(model) != expected_model:
                raise RuntimeError('Predictor tensor state changed during live verification')
            runtime.events.clear()
            positions.clear()
            selections.clear()
        lease = _RuntimeLease(model, refiner, runtime, expected_model, expected_refiner, _issuer=_FACTORY_ISSUER,
                              positions=positions, selections=selections)
        lease.validate_live(model, refiner, tensors=True)
        model_fingerprints = out / 'model_tensor_fingerprints.json'
        refiner_fingerprints = out / 'refiner_tensor_fingerprints.json'
        model_fingerprints.write_text(json.dumps(expected_model, indent=2) + '\n')
        refiner_fingerprints.write_text(json.dumps(expected_refiner, indent=2) + '\n')
        (out / 'runtime_verification.json').write_text(json.dumps({
            'passed': True, 'scope': 'Matched development and same-model seq4 engineering only; not an independent test.',
            'candidate_checkpoint_sha256': digest(Path(candidate_run) / 'checkpoint.pt'),
            'control_checkpoint_sha256': digest(Path(control_run) / 'checkpoint.pt'),
            'candidate_format': model_format, 'additional_controls': additional_controls,
            'selection_auditor_sha256': digest(ROOT / ('scripts/assess_region_conditioned_candidate.py'
                                                        if model_format == 'foreground_conditioned_refiner_v1'
                                                        else 'scripts/assess_refiner_candidate_promotion.py')),
            'underlying_screen_sha256': digest(ROOT / 'scripts/assess_refiner_candidate_promotion.py'),
            'baseline_reference_sha256': digest(Path(baseline_reference) / 'receipt.json'),
            'model_tensor_count': len(expected_model), 'refiner_tensor_count': len(expected_refiner),
            'model_tensor_fingerprints_sha256': digest(model_fingerprints),
            'refiner_tensor_fingerprints_sha256': digest(refiner_fingerprints),
            'source_sha256': {str(path.relative_to(ROOT)): digest(path) for path in (
                Path(__file__), ROOT / 'modeling/lean_verified_runtime_lease.py',
                ROOT / 'modeling/lean_contrast_refinement_integration.py',
                ROOT / 'modeling/foreground_conditioned_refiner.py',
                ROOT / 'scripts/assess_region_conditioned_candidate.py',
                ROOT / 'scripts/audit_region_conditioned_hardcase_results.py',
                ROOT / 'modeling/lean_indexed_spatial_rope_mps.py',
                ROOT / 'modeling/lean_mps_blocking_pointer_position.py',
                ROOT / 'modeling/lean_multiscale_anchor.py',
                ROOT / 'modeling/lean_live_state_audit.py',
                ROOT / 'modeling/lean_token_window_storage.py',
                ROOT / 'modeling/token_memory_window.py',
                ROOT / 'modeling/lean_token_storage_observer.py',
                ROOT / 'modeling/multilevel_contrast_refiner.py',
                ROOT / 'modeling/lean_channel_attention.py',
                ROOT / 'external/TCMN/model/cbam.py',
                ROOT / 'modeling/seq23_viewport.py',
                ROOT / 'modeling/samurai_auto_prompt.py',
                ROOT / 'scripts/run_samurai_automatic_priority.py',
                ROOT / 'scripts/run_author_baseline.py',
                ROOT / 'scripts/verify_refiner_capture_static.py',
                ROOT / 'scripts/verify_token_window_refiner_routing_mps.py',
                ROOT / 'scripts/memory_token_io.py',
                ROOT / 'datasets/video.py', ROOT / 'datasets/io.py',
                ROOT / 'data/polypgen_sequence.jsonl')},
            'device': 'mps float32', 'torch_version': str(torch.__version__),
            'python_version': sys.version, 'platform': platform.platform(), 'machine': platform.machine(),
            'mps_device_count': torch.mps.device_count(), 'gpu_inventory': _mps_hardware_inventory(),
            'gpu_inventory_note': 'System GPU inventory; PyTorch does not expose the selected MPS GPU name here.',
            'fallback': 'disabled', 'fast_math': 'disabled',
            'seed': 20261021, 'baseline_replays': 2,
            'memory_budget': 16384, 'insurance_reserve': 3072,
        }, indent=2) + '\n')
        try:
            yield model, refiner, lease
            lease.validate_live(model, refiner, tensors=True)
        finally:
            lease.close()
