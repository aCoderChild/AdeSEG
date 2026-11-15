#!/usr/bin/env python3
"""One fresh-process short seq11 MPS memory diagnostic; no labels."""
import argparse
from contextlib import ExitStack
import hashlib
import inspect
import json
from pathlib import Path
import resource
import subprocess
import sys
import tempfile
import time
from types import SimpleNamespace
import uuid

import numpy as np
import psutil
import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from datasets.video import stage_frames
from modeling.bounded_sam2_video_loader_v2 import (install_bounded_eager_loader,
    resolve_swr_init_chain, verify_predictor_image_access)
from modeling.bounded_loader_feature_observer_v1 import observe_decoder_features
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.lean_multiscale_anchor import install_multiscale_anchor
from modeling.lean_token_window_storage import TokenWindowRuntime
from modeling.prompt_feature_refinement import reference_prototypes
from modeling.warm_automatic_prompt_v2 import apply_archived_warm_prompt_v2
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.channel_value_fixed_read_v1 import require_environment_flags
from scripts.crossfit_observer_capture_shared_v1 import DATA, OUT, validate_plan
from scripts.memory_token_io import digest
from scripts.run_author_baseline import build_author_predictor
from scripts.run_crossfit_warm_video_v2 import FORK, video_inputs, warm_plan
from scripts.run_source_group_capture_v1 import state_hash
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.warm_crossfit_video_contract_v1 import require


def rss_bytes():
    value = int(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss)
    return value if sys.platform == 'darwin' else value * 1024


def memory_snapshot():
    return {'system_available_bytes': int(psutil.virtual_memory().available),
            'process_rss_bytes': int(psutil.Process().memory_info().rss),
            'mps_live_tensor_bytes': int(torch.mps.current_allocated_memory()),
            'mps_driver_bytes': int(torch.mps.driver_allocated_memory())}


def record_memory(path, frame, mode, min_ram, max_rss):
    """Check each observed sample immediately and preserve evidence on failure."""
    require(mode in ('observe','clear_idle_cache'), 'Unknown memory diagnostic mode')
    torch.mps.synchronize(); before = memory_snapshot()
    event = {'frame': int(frame), 'mode': mode, 'monotonic_seconds': time.monotonic(), 'before': before}
    def append():
        with Path(path).open('a') as stream:
            stream.write(json.dumps(event)+'\n'); stream.flush()
    if before['system_available_bytes'] < min_ram or before['process_rss_bytes'] > max_rss:
        event['stopped_at'] = 'before'; append()
        raise RuntimeError('Memory diagnostic observed resource limit before cache operation')
    if mode == 'clear_idle_cache': torch.mps.empty_cache()
    torch.mps.synchronize(); after = memory_snapshot(); event['after'] = after
    event['driver_change_bytes'] = after['mps_driver_bytes'] - before['mps_driver_bytes']
    if after['system_available_bytes'] < min_ram or after['process_rss_bytes'] > max_rss:
        event['stopped_at'] = 'after'; append()
        raise RuntimeError('Memory diagnostic observed resource limit after cache operation')
    append()
    return event


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mode', choices=('observe','clear_idle_cache'), required=True)
    parser.add_argument('--certificate', type=Path, required=True); parser.add_argument('--output', type=Path, required=True)
    parser.add_argument('--frames', type=int, default=7); parser.add_argument('--cache-frames', type=int, default=4)
    parser.add_argument('--min-available-ram-bytes', type=int, default=4*1024**3)
    parser.add_argument('--max-process-rss-bytes', type=int, default=6*1024**3)
    args = parser.parse_args(); out = args.output.resolve()
    require(args.min_available_ram_bytes >= 4*1024**3 and 0 < args.max_process_rss_bytes <= 6*1024**3,
            'Unsafe diagnostic resource settings')
    require(out.is_relative_to(OUT.resolve(strict=True)) and not out.exists(), 'Invalid diagnostic output')
    require(7 <= args.frames <= 32 and 1 <= args.cache_frames <= 4, 'Unsupported diagnostic clip/cache size')
    guard = install_mask_open_guard(DATA); require_environment_flags()
    require(torch.backends.mps.is_available(), 'MPS unavailable')
    torch.use_deterministic_algorithms(True); torch.set_num_threads(2)
    cert, cert_sha = verify_certificate(args.certificate); warm = warm_plan(); index = validate_plan()
    require(environment_identity() == warm['old_identity']['runtime_environment'], 'Runtime environment differs')
    frames, records, start, _, mask, box, points, roi = video_inputs('seq11', index, warm)
    require(start == 2 and len(frames) >= args.frames and args.frames >= start + 5, 'Seq11 clip differs')
    frames = frames[:args.frames]; records = records[:args.frames]
    for frame, row in zip(frames, records, strict=True):
        require(digest(frame.image_path) == row['image_sha256'], 'Diagnostic source frame differs')
    checkpoint = Path(cert['checkpoint_path']); require(digest(checkpoint) == cert['checkpoint_sha256'], 'Checkpoint differs')
    revision = subprocess.check_output(['git','-C',str(FORK),'rev-parse','HEAD'], text=True).strip()
    require(revision == warm['old_identity']['fork_revision'], 'Lean revision differs')
    predictor_file = FORK/'sam2/sam2_video_predictor_topk_insurance.py'
    predictor_sha = warm['old_identity']['fork_patches']['sam2/sam2_video_predictor_topk_insurance.py']
    wrapper_file = FORK/'sam2/sam2_video_predictor_swr_wrapper.py'; wrapper_sha = digest(wrapper_file)
    tracked_wrapper = subprocess.check_output(['git','-C',str(FORK),'show','HEAD:sam2/sam2_video_predictor_swr_wrapper.py'])
    require(hashlib.sha256(tracked_wrapper).hexdigest() == wrapper_sha, 'SWR wrapper differs from pinned revision')
    misc_file = FORK/'sam2/utils/misc.py'; misc_sha = digest(misc_file)
    tracked_misc = subprocess.check_output(['git','-C',str(FORK),'show','HEAD:sam2/utils/misc.py'])
    require(hashlib.sha256(tracked_misc).hexdigest() == misc_sha, 'Author frame decoder differs from pinned revision')
    access = verify_predictor_image_access(predictor_file, predictor_sha)
    torch.manual_seed(20261021); torch.mps.manual_seed(20261021); np.random.seed(20261021)
    sys.path.insert(0, str(FORK)); model = build_author_predictor(
        SimpleNamespace(method='lean', checkpoint=checkpoint, device='mps'), FORK)
    import sam2
    from sam2.utils import misc
    require(Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()) and
            Path(misc.__file__).resolve() == misc_file.resolve(), 'Wrong SAM2 worktree or decoder')
    require(model.num_maskmem == 7 and model.mem_keep_ratio == .25 and model.insurance_max_frames == 3 and
            model.enable_insurance_bank and model.fill_hole_area == 0, 'Lean configuration differs')
    model.ema_history_replace_old_tail_in_attn = False; model.ema_history_resident_prune = False
    model.token_window_runtime = TokenWindowRuntime(16384, 3072)
    selections = []; anchor = install_multiscale_anchor(model, .5, selection_events=selections)
    state_before = state_hash(model)
    require(state_before == warm['old_identity']['model_state_sha256'], 'Lean model fingerprint differs')
    resolve_swr_init_chain(type(model), wrapper_file=wrapper_file, wrapper_sha256=wrapper_sha,
                           predictor_file=predictor_file, predictor_sha256=predictor_sha)
    out.mkdir(parents=True, exist_ok=False); positions = []; rows = []; features = []; mps_peak = 0; driver_peak = 0
    with tempfile.TemporaryDirectory(prefix='staged_', dir=out) as staged:
        stage_frames(frames, Path(staged)); loader = None
        frame_bytes = 3 * model.image_size * model.image_size * 4
        with install_bounded_eager_loader(model, staged_dir=staged, expected_paths=[f.image_path for f in frames],
                    expected_sha256=[r['image_sha256'] for r in records], author_module_file=predictor_file,
                    author_module_sha256=predictor_sha, author_misc_sha256=misc_sha,
                    wrapper_module_file=wrapper_file, wrapper_module_sha256=wrapper_sha,
                    author_load_video_frames=misc.load_video_frames, author_decoder=misc._load_img_as_tensor,
                    limit_bytes=args.cache_frames * frame_bytes) as created:
            state = model.init_state(video_path=staged, offload_video_to_cpu=True, offload_state_to_cpu=False)
        loader = created[0]; require(state['images'] is loader, 'Bounded loader state was not installed')
        require(state['num_frames'] == args.frames and (state['video_height'], state['video_width']) == mask.shape,
                'Diagnostic video state geometry differs')
        try:
            with ExitStack() as stack:
                stack.enter_context(torch.inference_mode()); stack.enter_context(indexed_spatial_rope_mps(model, positions))
                stack.enter_context(blocking_mps_pointer_position(model))
                def before_mask(initial):
                    _, backbone, _, _, _ = model._get_image_feature(state, start, batch_size=1)
                    reference_prototypes(backbone['backbone_fpn'][0][0].detach().float().cpu(), initial)
                decision = apply_archived_warm_prompt_v2(model, state, start, box, points, mask, before_mask=before_mask)
                stack.enter_context(observe_decoder_features(model, range(start + 1, start + 5), out, features))
                for frame, ids, logits in model.propagate_in_video(state, start_frame_idx=start, reverse=False,
                                                                    max_frame_num_to_track=4):
                    require(frame == start + len(rows) and ids == [1] and torch.isfinite(logits).all(), 'Output differs')
                    kind = 'cond_frame_outputs' if frame == start else 'non_cond_frame_outputs'
                    entry = state['output_dict_per_obj'][0][kind][frame]
                    values = {'logits': logits.detach().cpu().clone()}
                    for key in ('pred_masks','maskmem_features','obj_ptr','object_score_logits'):
                        require(torch.isfinite(entry[key]).all(), 'Nonfinite native state')
                        values[key] = entry[key].detach().cpu().clone()
                    path = out/f'{frame:06d}.pt'; torch.save(values, path)
                    rows.append({'frame': frame, 'tensor_path': str(path), 'tensor_sha256': digest(path)})
                    record_memory(out/'memory.jsonl', frame, args.mode, args.min_available_ram_bytes, args.max_process_rss_bytes)
                    mps_peak = max(mps_peak, int(torch.mps.current_allocated_memory()))
                    driver_peak = max(driver_peak, int(torch.mps.driver_allocated_memory()))
            torch.mps.synchronize()
            require(len(rows) == 5 and state_hash(model) == state_before, 'Diagnostic coverage/model differs')
            loader.verify_sources(); cache = loader.stats()
            for frame, row in zip(frames, records, strict=True):
                require(digest(frame.image_path) == row['image_sha256'], 'Input frame changed')
        finally:
            model.reset_state(state)
    require(guard['open_attempts'] == 0 and digest(predictor_file) == predictor_sha and digest(wrapper_file) == wrapper_sha and
            digest(misc_file) == misc_sha and digest(checkpoint) == cert['checkpoint_sha256'],
            'Label guard or source changed')
    result = {'passed': True, 'mode': args.mode, 'scope': 'Short same-device memory diagnostic; no full-video parity or efficacy.',
        'run_uuid': uuid.uuid4().hex, 'process_id': __import__('os').getpid(), 'sequence': 'seq11',
        'frames_staged': args.frames, 'prompt_frame': start, 'roi_xyxy': roi,
        'min_available_ram_bytes': args.min_available_ram_bytes, 'max_process_rss_bytes': args.max_process_rss_bytes,
        'certificate_sha256': cert_sha, 'checkpoint_sha256': cert['checkpoint_sha256'],
        'predictor_sha256': predictor_sha, 'wrapper_sha256': wrapper_sha, 'misc_sha256': misc_sha, 'access_contract': access,
        'frame_sha256': [r['image_sha256'] for r in records], 'model_state_sha256': state_before,
        'conditioning_equals_archive': bool(np.array_equal(decision['conditioning_mask'], mask)),
        'generated_equals_archived': decision['generated_equals_archived'],
        'conditioning_generated_difference_pixels': decision['different_pixels'], 'rows': rows,
        'features': features,
        'memory_trace_sha256': digest(out/'memory.jsonl'),
        'positions': positions, 'selection': selections, 'anchor_events': anchor,
        'token_window_events': model.token_window_runtime.events, 'cache': cache,
        'peak_process_rss_bytes': rss_bytes(), 'sampled_peak_mps_allocated_bytes': mps_peak,
        'sampled_peak_mps_driver_bytes': driver_peak, 'label_guard': dict(guard)}
    (out/'receipt.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': True, 'mode': args.mode, 'output': str(out)}))


if __name__ == '__main__': main()
