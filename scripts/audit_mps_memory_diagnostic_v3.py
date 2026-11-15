#!/usr/bin/env python3
"""Compare two fresh-process short seq11 loader diagnostics on the same MPS device."""
import argparse
import json
import math
from pathlib import Path
import sys

import torch
ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.crossfit_observer_capture_shared_v1 import DATA, OUT, phase_artifact
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.warm_crossfit_video_contract_v1 import read_bound, require


def audit(root):
    from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
    from scripts.run_mps_memory_diagnostic_study_v3 import sources, AUTHOR_FILES
    root = Path(root).resolve(strict=True); require(root.is_relative_to(OUT.resolve(strict=True)), 'Invalid gate root')
    manifest_path = root/'run_manifest.json'; manifest_sha = digest(manifest_path)
    manifest = read_bound(manifest_path, manifest_sha)
    gib = 1024**3
    require(manifest.get('source_sha256') == sources() and manifest.get('phases') == ['clear_idle_cache','observe','audit'] and
            manifest.get('frames_staged') == 7 and type(manifest.get('max_process_rss_bytes')) is int and
            0 < manifest['max_process_rss_bytes'] <= 6*gib and
            type(manifest.get('min_disk_bytes')) is int and manifest['min_disk_bytes'] >= 26*gib and
            type(manifest.get('min_available_ram_bytes')) is int and manifest['min_available_ram_bytes'] >= 4*gib and
            type(manifest.get('cache_frames')) is int and 1 <= manifest['cache_frames'] <= 4,
            'Gate manifest, resource limits or source closure differs')
    snapshot = read_bound(root/'source_snapshot/snapshot_manifest.json', manifest['source_snapshot_manifest_sha256'])
    source_map = manifest['source_sha256']; snapshot_map = snapshot.get('source_sha256', {})
    require(set(snapshot_map) == set(source_map) | set(AUTHOR_FILES) and
            all(snapshot_map.get(name) == sha for name, sha in source_map.items()) and
            all(snapshot_map[name] == digest(ROOT/name) for name in AUTHOR_FILES), 'Diagnostic source snapshot differs')
    for name, sha in snapshot_map.items():
        path=(root/'source_snapshot'/name).resolve(strict=True)
        require(path.is_relative_to((root/'source_snapshot').resolve()) and digest(path)==sha,
                'Diagnostic source snapshot changed')
    certificate, cert_sha = verify_certificate(manifest['certificate_path'])
    require(cert_sha == manifest['certificate_sha256'], 'Gate certificate differs')
    phases = [json.loads(line) for line in (root/'phases.jsonl').read_text().splitlines() if line.strip()]
    names = [row.get('phase') for row in phases]
    require(names in (['clear_idle_cache','observe'], ['clear_idle_cache','observe','audit']), 'Gate phase order differs')
    if len(phases) == 3:
        phase_artifact(root, 'audit', root/'audit/audit.json')
    for phase in phases[:2]:
        monitor = phase.get('resource_monitor', {})
        require(type(monitor.get('minimum_sampled_disk_free_bytes')) is int and
                monitor['minimum_sampled_disk_free_bytes'] >= manifest['min_disk_bytes'] and
                type(monitor.get('minimum_sampled_system_available_ram_bytes')) is int and
                monitor['minimum_sampled_system_available_ram_bytes'] >= manifest['min_available_ram_bytes'] and
                type(monitor.get('peak_sampled_process_rss_bytes')) is int and
                0 <= monitor['peak_sampled_process_rss_bytes'] <= manifest['max_process_rss_bytes'],
                'Gate phase resource evidence differs')
    records = {}
    for mode in ('observe', 'clear_idle_cache'):
        path = root/mode/'receipt.json'; sha = phase_artifact(root, mode, path)
        rec = read_bound(path, sha)
        require(rec.get('passed') is True and rec.get('mode') == mode and rec.get('sequence') == 'seq11' and
                rec.get('certificate_sha256') == cert_sha and
                rec.get('min_available_ram_bytes') == manifest['min_available_ram_bytes'] and
                rec.get('max_process_rss_bytes') == manifest['max_process_rss_bytes'] and
                type(rec.get('peak_process_rss_bytes')) is int and
                0 < rec['peak_process_rss_bytes'] <= manifest['max_process_rss_bytes'] and
                rec.get('label_guard', {}).get('open_attempts') == 0, 'Diagnostic receipt differs')
        records[mode] = rec
    eager, bounded = records['observe'], records['clear_idle_cache']
    require(eager['process_id'] != bounded['process_id'] and eager['run_uuid'] != bounded['run_uuid'],
            'Diagnostics must run in fresh processes')
    for key in ('frames_staged','prompt_frame','roi_xyxy','certificate_sha256','checkpoint_sha256',
                'predictor_sha256','wrapper_sha256','misc_sha256','access_contract','frame_sha256','model_state_sha256',
                'conditioning_equals_archive','generated_equals_archived','conditioning_generated_difference_pixels'):
        require(eager[key] == bounded[key], f'Diagnostic input differs: {key}')
    require(eager['conditioning_equals_archive'] is True and eager['prompt_frame'] == 2 and
            eager['frames_staged'] == manifest['frames_staged'] == 7 and
            [x['frame'] for x in eager['rows']] == [x['frame'] for x in bounded['rows']] == list(range(2, 7)),
            'Diagnostic frame coverage or conditioning differs')
    cache = bounded['cache']
    require(isinstance(cache, dict) and cache.get('final_sha256_verified') is True and
            cache.get('peak_cached_tensor_bytes', 0) <= cache.get('limit_bytes', -1) and
            type(cache.get('evictions')) is int and cache['evictions'] > 0 and
            isinstance(eager['cache'], dict) and eager['cache'].get('final_sha256_verified') is True and
            eager['cache'].get('evictions', 0) > 0,
            'Diagnostic cache evidence differs')
    comparisons = []; failed = []
    for left, right in zip(eager['rows'], bounded['rows'], strict=True):
        tensors = []
        for row, mode in ((left,'observe'), (right,'clear_idle_cache')):
            path = Path(row['tensor_path']).resolve(strict=True)
            require(path.is_relative_to((root/mode).resolve()) and digest(path) == row['tensor_sha256'],
                    'Diagnostic tensor changed')
            tensors.append(torch.load(path, map_location='cpu', weights_only=True))
            require(digest(path) == row['tensor_sha256'], 'Diagnostic tensor changed during load')
        frame = {'frame': left['frame'], 'tensors': {}}
        for key in ('logits','pred_masks','maskmem_features','obj_ptr','object_score_logits'):
            a, b = tensors[0][key], tensors[1][key]
            require(a.shape == b.shape and a.dtype == b.dtype and torch.isfinite(a).all() and torch.isfinite(b).all(),
                    'Diagnostic tensor geometry or finiteness differs')
            af = a.float(); bf = b.float(); difference = (af-bf).abs()
            maximum = float(difference.max()) if difference.numel() else 0.
            mean = float(difference.mean()) if difference.numel() else 0.
            close = bool(torch.allclose(af, bf, rtol=1e-5, atol=1e-6))
            require(math.isfinite(maximum) and math.isfinite(mean), 'Diagnostic difference is nonfinite')
            entry = {'max_abs': maximum, 'mean_abs': mean, 'within_tolerance': close}
            if key == 'logits':
                flips = (a > 0) != (b > 0)
                entry['binary_mask_differing_pixels'] = int(flips.sum())
                entry['sign_flip_max_abs_logit'] = float(torch.maximum(af.abs(), bf.abs())[flips].max()) if flips.any() else 0.
                entry['sign_flip_max_abs_difference'] = float(difference[flips].max()) if flips.any() else 0.
            frame['tensors'][key] = entry
            # Same-device mask-bit differences are reported, not treated as
            # failure when logits remain within the declared tolerance.
            if not close:
                failed.append({'frame': left['frame'], 'tensor': key})
        comparisons.append(frame)
    traces = {key: eager[key] == bounded[key] for key in
              ('positions','selection','anchor_events','token_window_events')}
    failed.extend({'trace': key} for key,value in traces.items() if not value)
    expected_features = list(range(eager['prompt_frame'] + 1, eager['prompt_frame'] + 5))
    require([x['frame'] for x in eager['features']] == [x['frame'] for x in bounded['features']] ==
            expected_features, 'Decoder-feature coverage differs')
    feature_comparisons = []
    for left, right in zip(eager['features'], bounded['features'], strict=True):
        values = []
        for row, mode in ((left,'observe'), (right,'clear_idle_cache')):
            path = Path(row['path']).resolve(strict=True)
            require(path.is_relative_to((root/mode).resolve()) and digest(path) == row['sha256'],
                    'Decoder feature changed')
            values.append(torch.load(path, map_location='cpu', weights_only=True))
            require(digest(path) == row['sha256'], 'Decoder feature changed during load')
        frame = {'frame': left['frame'], 'features': {}}
        for key in ('early','deep'):
            a, b = values[0][key], values[1][key]
            require(a.shape == b.shape and a.dtype == b.dtype and torch.isfinite(a).all() and torch.isfinite(b).all(),
                    'Decoder feature geometry or finiteness differs')
            difference = (a.float()-b.float()).abs()
            maximum = float(difference.max()) if difference.numel() else 0.
            mean = float(difference.mean()) if difference.numel() else 0.
            close = bool(torch.allclose(a.float(), b.float(), rtol=1e-5, atol=1e-6))
            frame['features'][key] = {'max_abs': maximum, 'mean_abs': mean, 'within_tolerance': close}
            if not close: failed.append({'frame': left['frame'], 'feature': key})
        feature_comparisons.append(frame)
    milestones = {}
    names=['before_model_load','after_model_load','before_state_init','after_state_init','before_prompt','after_prompt']
    for mode, record in records.items():
        path=root/mode/'milestones.jsonl'; sha=record['milestone_trace_sha256']
        require(digest(path)==sha, 'Milestone trace changed')
        events=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require(digest(path)==sha and [x.get('name') for x in events]==names and
                all(isinstance(x.get('monotonic_seconds'),(int,float)) and math.isfinite(x['monotonic_seconds'])
                    for x in events) and all(a['monotonic_seconds'] <= b['monotonic_seconds']
                    for a,b in zip(events,events[1:])), 'Milestone trace coverage differs')
        for event in events:
            sample=event['sample']
            require(all(type(sample.get(k)) is int and sample[k]>=0 for k in
                ('system_available_bytes','process_rss_bytes','mps_live_tensor_bytes','mps_driver_bytes')) and
                sample['system_available_bytes']>=manifest['min_available_ram_bytes'] and
                sample['process_rss_bytes']<=manifest['max_process_rss_bytes'] and
                not event.get('stopped_on_resource_limit',False), 'Milestone sample exceeds guard')
        milestones[mode]=events
    setup_cache = {}
    for mode,record in records.items():
        path=root/mode/'setup_cache.jsonl'; sha=record['setup_cache_trace_sha256']
        require(digest(path)==sha, 'Setup cache trace changed')
        events=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require(digest(path)==sha and [x.get('name') for x in events] ==
                ['after_model_load','after_state_init','after_prompt'] and
                all(x.get('mode')==mode for x in events), 'Setup cache trace coverage differs')
        for event in events:
            for stage in ('before','after'):
                sample=event[stage]
                require(all(type(sample.get(k)) is int and sample[k]>=0 for k in
                    ('system_available_bytes','process_rss_bytes','mps_live_tensor_bytes','mps_driver_bytes')) and
                    sample['system_available_bytes']>=manifest['min_available_ram_bytes'] and
                    sample['process_rss_bytes']<=manifest['max_process_rss_bytes'], 'Setup cache sample exceeds guard')
            require(event['driver_change_bytes']==event['after']['mps_driver_bytes']-event['before']['mps_driver_bytes'],
                    'Setup cache driver delta differs')
        setup_cache[mode]=events
    memory = {}
    for mode,record in records.items():
        path=root/mode/'memory.jsonl';sha=record['memory_trace_sha256']
        require(digest(path)==sha,'Memory trace changed')
        events=[json.loads(line) for line in path.read_text().splitlines() if line.strip()]
        require(digest(path)==sha and [x['frame'] for x in events]==[r['frame'] for r in record['rows']] and
                all(x.get('mode')==mode and isinstance(x.get('monotonic_seconds'),(int,float)) and
                    math.isfinite(x['monotonic_seconds']) for x in events), 'Memory trace coverage differs')
        for event in events:
            for phase in ('before','after'):
                sample=event[phase]
                require(all(type(sample.get(k)) is int and sample[k]>=0 for k in
                    ('system_available_bytes','process_rss_bytes','mps_live_tensor_bytes','mps_driver_bytes')) and
                    sample['system_available_bytes']>=manifest['min_available_ram_bytes'] and
                    sample['process_rss_bytes']<=manifest['max_process_rss_bytes'], 'Memory sample exceeds guard')
            require(event['driver_change_bytes']==event['after']['mps_driver_bytes']-event['before']['mps_driver_bytes'],
                    'Memory driver delta differs')
        memory[mode]={'events':events,'minimum_available_bytes':min(x[p]['system_available_bytes'] for x in events
                           for p in ('before','after')),
                      'max_sampled_driver_bytes':max(x[p]['mps_driver_bytes'] for x in events for p in ('before','after')),
                      'driver_change_bytes':[x['driver_change_bytes'] for x in events]}
    require(digest(manifest_path) == manifest_sha and sources() == manifest['source_sha256'] and
            verify_certificate(manifest['certificate_path'])[1] == cert_sha,
            'Gate manifest, source or certificate changed during audit')
    return {'passed': not failed, 'scope': 'Same-device short cache-clearing memory diagnostic; not CPU/MPS parity, full-video proof or efficacy.',
        'sequence': 'seq11', 'frames_staged': eager['frames_staged'], 'comparisons': comparisons,
        'feature_comparisons': feature_comparisons, 'traces_equal': traces,
        'manifest_sha256': manifest_sha, 'certificate_sha256': cert_sha,
        'observe_peak_process_rss_bytes': eager['peak_process_rss_bytes'],
        'clear_peak_process_rss_bytes': bounded['peak_process_rss_bytes'],
        'observe_sampled_peak_mps_driver_bytes': eager['sampled_peak_mps_driver_bytes'],
        'clear_sampled_peak_mps_driver_bytes': bounded['sampled_peak_mps_driver_bytes'],
        'clear_cache': cache, 'memory': memory, 'milestones': milestones, 'setup_cache': setup_cache, 'errors': failed, 'auditor_sha256': digest(Path(__file__))}


def main():
    parser = argparse.ArgumentParser(description=__doc__); parser.add_argument('--root', type=Path, required=True)
    parser.add_argument('--output', type=Path, required=True); args = parser.parse_args(); out = args.output.resolve()
    require(out.is_relative_to(OUT.resolve(strict=True)) and not out.exists(), 'Invalid audit output')
    guard = install_mask_open_guard(DATA); result = audit(args.root)
    require(guard['open_attempts'] == 0, 'Annotation opened during gate audit')
    result['label_guard'] = dict(guard); out.mkdir(parents=True, exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result, indent=2) + '\n')
    print(json.dumps({'passed': result['passed'], 'output': str(out), 'errors': len(result['errors'])}))
    require(result['passed'], 'Loader diagnostic differs; full evidence retained')


if __name__ == '__main__': main()
