#!/usr/bin/env python3
"""Verify all eight exploratory runs before scoring any target label masks.

The matched comparator is the freshly frozen A arm, not an unrelated prior
CPU run. Previously inspected patients are not a blinded evaluation.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest, validate_output
from scripts.run_samurai_automatic_priority import MANIFEST
from scripts.score_boundary_video_factorial_v1 import (_check_events, check_conditioning,
    check_telemetry, metric, summary, paired_summary)
from scripts.run_remaining_boundary_factorial_v1 import (ARMS, BASE, CHECKPOINT, DETECTOR,
    DETECTOR_SHA256, RUNNER, SEQUENCES, input_digest, study_sources, verify_original_gate, verify_extended_gate)


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok: raise ValueError(message)


def mapped_runs(path, experiment_root):
    def no_duplicate_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj: raise ValueError('Duplicate run-map key')
            obj[key] = value
        return obj
    data = json.loads(Path(path).read_text(), object_pairs_hook=no_duplicate_keys)
    require(set(data) == set(SEQUENCES) and all(set(data[s]) == set(ARMS) for s in SEQUENCES),
            'Run map must contain both sequences and four arms')
    root = experiment_root.resolve(strict=True)
    mapping = {s:{a:Path(data[s][a]).resolve(strict=True) for a in ARMS} for s in SEQUENCES}
    paths = [mapping[s][a] for s in SEQUENCES for a in ARMS]
    require(len(set(paths)) == 8 and all(p.is_dir() and p.is_relative_to(root) for p in paths),
            'Run folders must be distinct and within experiment root')
    return mapping


def verify_launch(run_map, mapping):
    """Check launch/gates and all 8 completed phase logs before labels."""
    run = Path(run_map).parent.resolve(strict=True)
    launch_path = run/'run_manifest.json'; launch = read(launch_path)
    require(launch.get('run_map') == read(run_map) and launch.get('sequences') == list(SEQUENCES)
            and launch.get('arms') == list(ARMS) and launch.get('python_hash_seed') == '0'
            and launch.get('mps_fallback') == '0' and launch.get('mps_fast_math') == '0'
            and launch.get('checkpoint_sha256') == digest(CHECKPOINT)
            and launch.get('detector_sha256') == DETECTOR_SHA256 == digest(DETECTOR),
            'Frozen launch configuration differs')
    source_paths = study_sources()
    require(launch.get('source_sha256') == {k:digest(v) for k,v in source_paths.items()}, 'Launch sources changed')
    expected_inputs = ('preflight','gate','gate_audit','extension_gate','extension_audit')
    require(set(launch.get('inputs',{})) == set(expected_inputs), 'Launch input coverage differs')
    inputs = {k:Path(launch['inputs'][k]['path']).resolve(strict=True) for k in expected_inputs}
    require(all(input_digest(k,v) == launch['inputs'][k]['sha256'] for k,v in inputs.items()) and
            digest(inputs['preflight']/'completion.json') == launch['preflight_completion_sha256'], 'Launch input changed')
    require(all(p.is_relative_to(BASE.resolve(strict=True)) for p in inputs.values()), 'Gate outside experiment root')
    from scripts.preflight_remaining_boundary_video_v1 import load_extended_preflight
    load_extended_preflight(inputs['preflight'],CHECKPOINT,DETECTOR)
    verify_original_gate(inputs['gate'],inputs['gate_audit'])
    verify_extended_gate(inputs['preflight'],inputs['extension_gate'],inputs['extension_audit'],inputs['gate'])
    phases = [json.loads(x) for x in (run/'phases.jsonl').read_text().splitlines() if x.strip()]
    expected = [f'{seq}_{arm}' for seq in SEQUENCES for arm in ARMS]
    require([p.get('phase') for p in phases] == expected and all(
        p.get('passed') is True and p.get('log_sha256') == digest(run/(p['phase']+'.log')) for p in phases),
        'All 8 full-run phases and logs must freeze before scoring')
    require(all(mapping[s][a].is_relative_to(run) for s in SEQUENCES for a in ARMS), 'Run folder outside batch')
    return run, launch_path, inputs


def verify_runs(mapping, videos, preflight, inputs):
    """Validate inputs, frozen predictions, routing and telemetry; no labels."""
    receipts, telemetry, timing, runtime_identity = {}, {}, {}, None
    pre = read(preflight/'manifest.json')
    require(pre.get('selection') == list(SEQUENCES), 'Frozen sequence selection differs')
    original_identity=read(inputs['gate'])['runtime_identity']
    extension_identity=read(inputs['extension_gate'])['runtime_identity']
    require(original_identity == extension_identity, 'Engineering runtime identities differ')
    from scripts.run_remaining_boundary_factorial_v1 import STUDY
    candidate = digest(STUDY/'runs/region_boundary/checkpoint.pt')
    control = digest(STUDY/'runs/mask_only_boundary/checkpoint.pt')
    for seq in SEQUENCES:
        frames, meta = videos[seq], pre['sequences'][seq]
        require(meta['frame_count'] == len(frames), 'Preflight frame count differs')
        expected_images = [{'frame':i, 'source_frame_index':frame.frame_index, 'sha256':digest(frame.image_path)}
                           for i,frame in enumerate(frames)]
        require(meta['frames'] == [{'frame':i, 'source_frame_index':f.frame_index,
                    'image_path':str(f.image_path.resolve()), 'image_sha256':expected_images[i]['sha256']}
                   for i,f in enumerate(frames)], 'Frozen images or source frames differ')
        receipts[seq], telemetry[seq], timing[seq] = {}, {}, {}
        for arm in ARMS:
            folder = mapping[seq][arm]; rec = read(folder/'completion.json')
            expected = {'passed':True, 'full_video':True, 'limit':0, 'sequence':seq, 'arm':arm,
                'runner_source_sha256':digest(RUNNER), 'prompt_frame':meta['first_detection_index'],
                'prompt_sha256':meta['prompt_metadata_sha256'], 'prompt_mask_sha256':meta['prompt_mask_sha256'],
                'view':meta['view'], 'checkpoint_sha256':digest(CHECKPOINT), 'detector_sha256':DETECTOR_SHA256,
                'candidate_checkpoint_sha256':candidate, 'control_checkpoint_sha256':control,
                'engineering_gate_sha256':digest(inputs['gate']),
                'extended_preflight_sha256':input_digest('preflight',preflight),
                'extension_gate_sha256':digest(inputs['extension_gate']),
                'extension_audit_sha256':digest(inputs['extension_audit'])}
            require(all(rec.get(k) == v for k,v in expected.items()) and rec.get('label_guard',{}).get('open_attempts') == 0,
                    'Run identity, prompt, gate or label guard differs')
            require(rec.get('input_images') == expected_images and
                    [m['frame'] for m in rec.get('masks',[])] == list(range(len(frames))) and
                    [x['frame'] for x in rec.get('post_prompt_state',[])] == list(range(rec['prompt_frame'],len(frames))),
                    'Full input/output frame coverage differs')
            runtime = _check_events(folder,rec)
            require(runtime == original_identity, 'Live runtime does not match both short gates')
            telemetry[seq][arm], timing[seq][arm] = check_telemetry(folder,rec,len(frames))
            if runtime_identity is None: runtime_identity = runtime
            require(runtime == runtime_identity, 'Live runtime identity differs between arms or videos')
            for row in rec['masks']:
                i = row['frame']; mask = folder/f'{i:06d}.png'
                require(digest(mask) == row['sha256'], 'Frozen prediction changed')
                with Image.open(frames[i].image_path) as source, Image.open(mask) as image:
                    require(image.format == 'PNG' and image.size == source.size, 'Prediction geometry differs')
                    binary = np.asarray(image.convert('L'))
                    require(np.isin(binary,(0,255)).all(), 'Prediction not binary')
                    if i < rec['prompt_frame']: require(not binary.any(), 'Pre-prompt prediction not empty')
            receipts[seq][arm] = rec
        a = receipts[seq]['A']; b = receipts[seq]['B']
        for arm in 'BCD':
            cur = receipts[seq][arm]
            check_conditioning(cur['post_prompt_state'][0],a['post_prompt_state'][0])
            for key in ('pred_masks_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                require(cur['post_prompt_state'][1].get(key) is not None and
                        cur['post_prompt_state'][1][key] == a['post_prompt_state'][1][key],
                        'First post-prompt native decoder state differs')
        first = {arm:receipts[seq][arm]['post_prompt_state'][1] for arm in ARMS}
        require(first['A']['output_logits_sha256'] == first['C']['output_logits_sha256'] and
                first['B']['output_logits_sha256'] == first['D']['output_logits_sha256'] and
                first['C']['maskmem_features_sha256'] == first['D']['maskmem_features_sha256'],
                'First output/memory routes differ')
        for cur,base in zip(b['post_prompt_state'],a['post_prompt_state'],strict=True):
            for key in ('pred_masks_sha256','maskmem_features_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                require(cur.get(key) is not None and cur[key] == base.get(key), 'B changed native state')
        for name in ('memory_events.json','positions.json','selections.json'):
            require(read(mapping[seq]['A']/name) == read(mapping[seq]['B']/name), 'B changed memory routing')
    return receipts,telemetry,timing


def score(mapping, videos, preflight, receipts):
    """Called only after complete prediction/gate checks; label paths now open."""
    pre = read(preflight/'manifest.json'); metadata = {}
    for seq in SEQUENCES:
        meta = pre['sequences'][seq]; path = Path(meta['archive_frames_path'])
        require(digest(path) == meta['archive_frames_sha256'], 'Frozen archive metadata differs')
        metadata[seq] = read(path)
        require(len(metadata[seq]) == len(videos[seq]), 'Label metadata coverage differs')
    rows_all, summaries, paired, preprompt = {}, {}, {}, {}
    for seq in SEQUENCES:
        rows = []; start = receipts[seq]['A']['prompt_frame']
        for i,frame in enumerate(videos[seq]):
            require(digest(frame.mask_path) == metadata[seq][i]['label_sha256'], 'Label hash changed')
            truth = load_label_mask(frame.mask_path,[1]) == 1
            row = {'frame':i, 'source_frame_index':frame.frame_index,
                   'post_prompt':i>start, 'initialization_frame':i==start}
            for arm in ARMS:
                with Image.open(mapping[seq][arm]/f'{i:06d}.png') as image:
                    prediction = np.asarray(image.convert('L')) > 0
                row[arm] = metric(prediction,truth)
            rows.append(row)
        rows_all[seq] = rows
        summaries[seq] = {a:{'post_prompt':summary(rows,a), 'all':summary(rows,a,False)} for a in ARMS}
        paired[seq] = {name:paired_summary(rows,arm,base) for name,arm,base in
                       (('B_minus_A','B','A'),('C_minus_A','C','A'),('D_minus_B','D','B'),('D_minus_A','D','A'))}
        preprompt[seq] = {'frames':start,'present_misses':sum(r['A']['present'] for r in rows[:start])}
    return rows_all,summaries,paired,preprompt


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for n in ('experiment-root','run-map','output'): p.add_argument('--'+n,type=Path,required=True)
    a = p.parse_args(); root = a.experiment_root.resolve(strict=True)
    require(root == BASE.resolve(strict=True), 'Unexpected experiment root')
    mapping = mapped_runs(a.run_map,root)
    run,launch_path,inputs = verify_launch(a.run_map,mapping)
    videos = {**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')}
    receipts,telemetry,timing = verify_runs(mapping,videos,inputs['preflight'],inputs)
    rows,summaries,paired,preprompt = score(mapping,videos,inputs['preflight'],receipts)
    out = validate_output(a.output); out.mkdir(parents=True,exist_ok=False)
    lines = ['# Exploratory additional-patient A/B/C/D video factorial','',
      'The user reports each sequence is a distinct patient. These sequences and baseline outcomes were previously inspected; this is not a blinded or externally validated study. Post-prompt excludes initialization.','',
      '| Sequence | Present | Dice A / B / C / D | Labeled-empty alarms A / B / C / D |','|---|---:|---:|---:|']
    for seq in SEQUENCES:
        s = summaries[seq]; dice = ' / '.join(f"{s[x]['post_prompt']['mean_present_dice']:.4f}" if s[x]['post_prompt']['mean_present_dice'] is not None else 'NA' for x in ARMS)
        alarms = ' / '.join(f"{s[x]['post_prompt']['false_alarms']}/{s[x]['post_prompt']['empty_frames']}" for x in ARMS)
        lines.append(f"| {seq} | {s['A']['post_prompt']['present_frames']} | {dice} | {alarms} |")
    lines += ['', 'A: original output and memory; B: refined output only; C: refined memory input only; D: both. Conditioning and native pointer/presence are protected. C/D can change future trajectories.',
      '', 'Project Lean is a modified token-window system, not unmodified author Lean. Historical YOLO scanned to first detection; the saved prompt is replayed without redetection. Labeled-empty does not establish clinical absence; per-frame rows, pre-prompt misses, precision, recall, false-positive areas and five-native-pixel boundary scores are in results.json.',
      '', 'The model was selected after development outcomes were seen. Patient distinctness is user-reported rather than independently verified; detector/backbone exposure and near-duplicate/source relationships remain unresolved. No clinical margin or independent final-test claim is made.']
    report = out/'report.md'; report.write_text('\n'.join(lines)+'\n')
    result = {'passed':True,'scope':'Previously inspected user-reported distinct patients; exploratory.',
      'summaries':summaries,'paired_descriptive':paired,'rows':rows,'preprompt':preprompt,
      'suppression_telemetry':telemetry,'timing':timing,
      'run_receipt_sha256':{s:{a:digest(mapping[s][a]/'completion.json') for a in ARMS} for s in SEQUENCES},
      'run_map_sha256':digest(a.run_map),'launch_manifest_sha256':digest(launch_path),
      'preflight_sha256':input_digest('preflight',inputs['preflight']),'engineering_gate_sha256':digest(inputs['gate']),
      'extension_gate_sha256':digest(inputs['extension_gate']), 'scorer_sha256':digest(Path(__file__)),
      'report_sha256':digest(report)}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(out),'summaries':summaries},indent=2))


if __name__ == '__main__': main()
