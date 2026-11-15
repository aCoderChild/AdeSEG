#!/usr/bin/env python3
"""Verify all 36 training-exposed videos before scoring any target label masks.

The matched comparator is the freshly frozen A arm, not an unrelated prior
CPU run. Previously inspected patients are not a blinded evaluation.
"""
import argparse
import json
import os
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
from scripts.source_group_training_label_guard_v1 import mask_path_sets
from scripts.score_boundary_video_factorial_v1 import (_check_events, check_conditioning,
    check_telemetry, metric, summary, paired_summary)
from scripts.run_training_exposed_boundary_factorial_v1 import (ARMS, BASE, CHECKPOINT, DETECTOR,
    DETECTOR_SHA256, RUNNER, SEQUENCES, input_digest, verify_original_gate, verify_extended_gate)


def read(path):
    return json.loads(Path(path).read_text())


def require(ok, message):
    if not ok: raise ValueError(message)


def install_prelabel_guard():
    train, protected = mask_path_sets(MANIFEST)
    blocked = train | protected
    state = {'blocked_paths': len(blocked), 'open_attempts': 0, 'active': True}
    def audit(event, args):
        if (state['active'] and event == 'open' and args
                and isinstance(args[0], (str, bytes, os.PathLike))
                and os.path.realpath(os.fsdecode(args[0])) in blocked):
            state['open_attempts'] += 1
            raise PermissionError('Label mask blocked during pre-score validation')
    sys.addaudithook(audit)
    return state


def mapped_runs(path, experiment_root):
    def no_duplicate_keys(pairs):
        obj = {}
        for key, value in pairs:
            if key in obj: raise ValueError('Duplicate run-map key')
            obj[key] = value
        return obj
    data = json.loads(Path(path).read_text(), object_pairs_hook=no_duplicate_keys)
    require(set(data) == set(SEQUENCES) and all(set(data[s]) == set(ARMS) for s in SEQUENCES),
            'Run map must contain all nine sequences and four arms')
    root = experiment_root.resolve(strict=True)
    mapping = {s:{a:Path(data[s][a]).resolve(strict=True) for a in ARMS} for s in SEQUENCES}
    paths = [mapping[s][a] for s in SEQUENCES for a in ARMS]
    require(len(set(paths)) == 36 and all(p.is_dir() and p.is_relative_to(root) for p in paths),
            'Run folders must be distinct and within experiment root')
    return mapping


def check_phase_completions(phases,mapping):
    result={}
    for phase in phases:
        if phase['phase'] == 'score': continue
        sequence,arm=phase['phase'].split('_',1)
        require(phase.get('completion_sha256')==digest(mapping[sequence][arm]/'completion.json'),
                'Completion changed after its launch phase')
        result[(sequence,arm)]=phase['completion_sha256']
    return result


def verify_launch(run_map, mapping, *, score_complete=False):
    """Check launch/gates and all 36 completed phase logs before labels."""
    run = Path(run_map).parent.resolve(strict=True)
    launch_path = run/'run_manifest.json'; launch = read(launch_path)
    require(launch.get('run_map') == read(run_map) and launch.get('sequences') == list(SEQUENCES)
            and launch.get('arms') == list(ARMS) and launch.get('python_hash_seed') == '0'
            and launch.get('mps_fallback') == '0' and launch.get('mps_fast_math') == '0'
            and launch.get('checkpoint_sha256') == digest(CHECKPOINT)
            and launch.get('detector_sha256') == DETECTOR_SHA256 == digest(DETECTOR),
            'Frozen launch configuration differs')
    source_paths = {'launcher':ROOT/'scripts/run_training_exposed_boundary_factorial_v1.py',
                    'scorer':Path(__file__), 'runner':RUNNER,
                    'auditor':ROOT/'scripts/audit_training_exposed_boundary_factorial_v1.py',
                    'protocol':ROOT/'protocols/TRAINING_EXPOSED_BOUNDARY_FACTORIAL_V1.md',
                    'independent_metric':ROOT/'scripts/audit_boundary_video_factorial_v1.py',
                    'primary_metric':ROOT/'scripts/score_boundary_video_factorial_v1.py',
                    'label_loader':ROOT/'datasets/io.py',
                    'label_guard':ROOT/'scripts/source_group_training_label_guard_v1.py',
                    'preflight_source':ROOT/'scripts/preflight_training_exposed_boundary_video_v1.py',
                    'archive_loader':ROOT/'scripts/preflight_author_trainval_v1.py',
                    'preflight_remaining':ROOT/'scripts/preflight_remaining_boundary_video_v1.py',
                    'preflight_extended':ROOT/'scripts/preflight_extended_boundary_video_v1.py',
                    'prompt_helper':ROOT/'modeling/samurai_auto_prompt.py',
                    'runtime_label_guard':ROOT/'scripts/source_group_label_guard_v1.py',
                    'primary_boundary_metric':ROOT/'scripts/score_source_group_boundary_v1.py',
                    'extension_gate_source':ROOT/'scripts/verify_training_exposed_boundary_video_short_gate_v1.py',
                    'extension_auditor_source':ROOT/'scripts/audit_training_exposed_boundary_video_short_v1.py'}
    require(launch.get('source_sha256') == {k:digest(v) for k,v in source_paths.items()}, 'Launch sources changed')
    expected_inputs = ('preflight','gate','gate_audit','extension_gate','extension_audit')
    require(set(launch.get('inputs',{})) == set(expected_inputs), 'Launch input coverage differs')
    inputs = {k:Path(launch['inputs'][k]['path']).resolve(strict=True) for k in expected_inputs}
    require(all(input_digest(k,v) == launch['inputs'][k]['sha256'] for k,v in inputs.items()) and
            digest(inputs['preflight']/'completion.json') == launch['preflight_completion_sha256'], 'Launch input changed')
    require(all(p.is_relative_to(BASE.resolve(strict=True)) for p in inputs.values()), 'Gate outside experiment root')
    from scripts.preflight_training_exposed_boundary_video_v1 import load_training_exposed_preflight
    load_training_exposed_preflight(inputs['preflight'],CHECKPOINT,DETECTOR)
    verify_original_gate(inputs['gate'],inputs['gate_audit'])
    verify_extended_gate(inputs['preflight'],inputs['extension_gate'],inputs['extension_audit'],inputs['gate'])
    phases = [json.loads(x) for x in (run/'phases.jsonl').read_text().splitlines() if x.strip()]
    expected = [f'{seq}_{arm}' for seq in SEQUENCES for arm in ARMS]
    if score_complete: expected.append('score')
    require([p.get('phase') for p in phases] == expected and all(
        p.get('passed') is True and p.get('log_sha256') == digest(run/(p['phase']+'.log')) for p in phases),
        'All 36 full-run phases and logs must freeze before scoring')
    phase_receipts=check_phase_completions(phases,mapping)
    require(all(mapping[s][a].is_relative_to(run) for s in SEQUENCES for a in ARMS), 'Run folder outside batch')
    return run, launch_path, inputs, phase_receipts


def verify_runs(mapping, videos, preflight, inputs, phase_receipts):
    """Validate inputs, frozen predictions, routing and telemetry; no labels."""
    receipts, telemetry, timing, runtime_identity = {}, {}, {}, None
    pre = read(preflight/'manifest.json')
    require(set(pre.get('sequences', {})) == set(SEQUENCES), 'Frozen sequence selection differs')
    original_identity=read(inputs['gate'])['runtime_identity']
    extension_identity=read(inputs['extension_gate'])['runtime_identity']
    require(original_identity == extension_identity, 'Engineering runtime identities differ')
    from scripts.run_training_exposed_boundary_factorial_v1 import STUDY
    candidate = digest(STUDY/'runs/region_boundary/checkpoint.pt')
    control = digest(STUDY/'runs/mask_only_boundary/checkpoint.pt')
    for seq in SEQUENCES:
        frames, meta = videos[seq], pre['sequences'][seq]
        require(meta['frames'] == len(frames), 'Preflight frame count differs')
        expected_images = [{'frame':i, 'source_frame_index':frame.frame_index, 'sha256':digest(frame.image_path)}
                           for i,frame in enumerate(frames)]
        require(meta['image_sha256'] == [r['sha256'] for r in expected_images], 'Frozen images differ')
        receipts[seq], telemetry[seq], timing[seq] = {}, {}, {}
        for arm in ARMS:
            folder = mapping[seq][arm]; completion=folder/'completion.json'; before=digest(completion)
            require(before==phase_receipts[(seq,arm)], 'Completion differs from launch phase')
            rec=read(completion)
            require(digest(completion)==phase_receipts[(seq,arm)], 'Completion changed while reading')
            expected = {'passed':True, 'full_video':True, 'limit':0, 'sequence':seq, 'arm':arm,
                'runner_source_sha256':digest(RUNNER), 'prompt_frame':meta['prompt_frame'],
                'prompt_sha256':meta['archived_prompt_sha256'], 'prompt_mask_sha256':meta['archived_prompt_mask_sha256'],
                'view':{'kind':'full_source_frame'}, 'checkpoint_sha256':digest(CHECKPOINT), 'detector_sha256':DETECTOR_SHA256,
                'candidate_checkpoint_sha256':candidate, 'control_checkpoint_sha256':control,
                'engineering_gate_sha256':digest(inputs['gate']),
                'training_exposed_preflight_sha256':input_digest('preflight',preflight),
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
        meta = pre['sequences'][seq]; path = Path(meta['source'])/seq/'frames.json'
        require(digest(path) == meta['archived_frames_sha256'], 'Frozen archive metadata differs')
        metadata[seq] = read(path)
        require(digest(path) == meta['archived_frames_sha256'], 'Frozen archive metadata changed during parsing')
        require(len(metadata[seq]) == len(videos[seq]), 'Label metadata coverage differs')
    rows_all, summaries, paired, preprompt = {}, {}, {}, {}
    for seq in SEQUENCES:
        rows = []; start = receipts[seq]['A']['prompt_frame']
        for i,frame in enumerate(videos[seq]):
            image_sha=receipts[seq]['A']['input_images'][i]['sha256']
            require(digest(frame.image_path)==image_sha, 'Input image changed before scoring')
            require(digest(frame.mask_path) == metadata[seq][i]['label_sha256'], 'Label hash changed')
            truth = load_label_mask(frame.mask_path,[1]) == 1
            require(digest(frame.mask_path) == metadata[seq][i]['label_sha256'], 'Label changed during decode')
            row = {'frame':i, 'source_frame_index':frame.frame_index,
                   'post_prompt':i>start, 'initialization_frame':i==start}
            for arm in ARMS:
                file=mapping[seq][arm]/f'{i:06d}.png'; wanted=receipts[seq][arm]['masks'][i]['sha256']
                require(digest(file)==wanted, 'Prediction changed before decode')
                with Image.open(file) as image:
                    prediction = np.asarray(image.convert('L')) > 0
                require(digest(file)==wanted, 'Prediction changed during decode')
                row[arm] = metric(prediction,truth)
            require(digest(frame.image_path)==image_sha, 'Input image changed during scoring')
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
    out=validate_output(a.output); require(out.is_relative_to(root) and not out.exists(), 'Invalid or existing score output')
    guard=install_prelabel_guard()
    try:
        mapping = mapped_runs(a.run_map,root)
        run,launch_path,inputs,phase_receipts = verify_launch(a.run_map,mapping)
        videos = {**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')}
        receipts,telemetry,timing = verify_runs(mapping,videos,inputs['preflight'],inputs,phase_receipts)
        require(guard['open_attempts']==0, 'Label open attempted during validation')
    finally:
        guard['active']=False
    rows,summaries,paired,preprompt = score(mapping,videos,inputs['preflight'],receipts)
    out.mkdir(parents=True,exist_ok=False)
    lines = ['# Training/selection-exposed A/B/C/D video factorial','',
      'Seq2/3/4/5/12/13/14 contributed training reads; seq9/15 contributed checkpoint-selection validation reads. These are in-sample or selection-exposed results, not held-out validation. The user reports each sequence is a distinct patient. Post-prompt excludes initialization.','',
      '| Sequence | Present | Dice A / B / C / D | Labeled-empty alarms A / B / C / D |','|---|---:|---:|---:|']
    for seq in SEQUENCES:
        s = summaries[seq]; dice = ' / '.join(f"{s[x]['post_prompt']['mean_present_dice']:.4f}" if s[x]['post_prompt']['mean_present_dice'] is not None else 'NA' for x in ARMS)
        alarms = ' / '.join(f"{s[x]['post_prompt']['false_alarms']}/{s[x]['post_prompt']['empty_frames']}" for x in ARMS)
        lines.append(f"| {seq} | {s['A']['post_prompt']['present_frames']} | {dice} | {alarms} |")
    lines += ['', 'A: original output and memory; B: refined output only; C: refined memory input only; D: both. Conditioning and native pointer/presence are protected. C/D can change future trajectories.',
      '', 'Project Lean is a modified token-window system, not unmodified author Lean. Historical YOLO scanned to first detection; the saved prompt is replayed without redetection. Labeled-empty does not establish clinical absence; per-frame rows, pre-prompt misses, precision, recall, false-positive areas and five-native-pixel boundary scores are in results.json.',
      '', 'The model was selected after development outcomes were seen. Patient distinctness is user-reported rather than independently verified; detector/backbone exposure and near-duplicate/source relationships remain unresolved. No clinical margin or independent final-test claim is made.']
    report = out/'report.md'; report.write_text('\n'.join(lines)+'\n')
    result = {'passed':True,'scope':'Training/selection-exposed cohort; in-sample/selection-exposed descriptive results only.',
      'sequence_exposure':{s:('selection_validation' if s in ('seq9','seq15') else 'training') for s in SEQUENCES},
      'summaries':summaries,'paired_descriptive':paired,'rows':rows,'preprompt':preprompt,
      'suppression_telemetry':telemetry,'timing':timing,'prelabel_guard':dict(guard),
      'run_receipt_sha256':{s:{a:digest(mapping[s][a]/'completion.json') for a in ARMS} for s in SEQUENCES},
      'run_map_sha256':digest(a.run_map),'launch_manifest_sha256':digest(launch_path),
      'preflight_sha256':input_digest('preflight',inputs['preflight']),'engineering_gate_sha256':digest(inputs['gate']),
      'extension_gate_sha256':digest(inputs['extension_gate']), 'scorer_sha256':digest(Path(__file__)),
      'report_sha256':digest(report)}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(out),'summaries':summaries},indent=2))


if __name__ == '__main__': main()
