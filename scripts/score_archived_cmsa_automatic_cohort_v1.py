#!/usr/bin/env python3
"""Rescore frozen released-CMSA predictions on 20 automatic-prompt target sets."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from scripts.crossfit_observer_capture_shared_v1 import archived_frame_metadata as priority_metadata, sequence_inputs, validate_plan
from scripts.local_import_closure_v1 import local_import_closure
from scripts.memory_token_io import digest
from scripts.run_additional_author_comparator_v2 import DATA, SEQUENCES as ADDITIONAL, new_output
from scripts.run_additional_author_comparator_study_v2 import commands, source_hashes as comparator_sources
from scripts.score_additional_author_comparators_v2 import install_phase_mask_guard
from scripts.score_boundary_video_factorial_v1 import metric, summary
from scripts.score_cmsa_seq23_viewport_v1 import PRIOR, PRIOR_AUDIT, read_bound, require, verify_prior
from scripts.source_group_label_guard_v1 import install_mask_open_guard

EXPERIMENT = Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment')
CMSA = EXPERIMENT/'cmsa_net_polypgen_v1/full_attempt01'
CHECKPOINT = EXPERIMENT/'cmsa_net_polypgen_v1/assets/ckpt_epoch_007.pth'
PRIORITY = ('seq11', 'seq20')
SEQUENCES = tuple(f'seq{i}' for i in range(2, 23) if i != 7)
SOURCES = ('scripts/score_archived_cmsa_automatic_cohort_v1.py',
           'scripts/audit_archived_cmsa_automatic_cohort_v1.py',
           'scripts/local_import_closure_v1.py', 'scripts/score_cmsa_seq23_viewport_v1.py')


def source_hashes():
    names = set(SOURCES) | set(local_import_closure(ROOT, SOURCES))
    return {name: digest(ROOT/name) for name in sorted(names)}


def require_phase_artifacts(phase_rows, hashes):
    lookup = {row['phase']: row['artifact_sha256'] for row in phase_rows}
    require(all(lookup.get(phase) == hashes[key] for phase, key in
                (('verify', 'verification'), ('score', 'score'), ('audit', 'audit'))),
            'Comparator phase-frozen verification, score or audit differs')


def validate_inputs(batch):
    """Verify frozen benchmark and archived predictions without annotation opens."""
    batch = Path(batch).resolve(strict=True)
    require(batch.is_relative_to(EXPERIMENT.resolve(strict=True)), 'Batch outside Drive experiments')
    paths = {'launch': batch/'run_manifest.json', 'phase_log': batch/'phases.jsonl',
             'score': batch/'scores/results.json', 'audit': batch/'independent_audit/audit.json',
             'verification': batch/'verification/verification.json', 'run_map': batch/'run_map.json'}
    hashes = {key: digest(path) for key, path in paths.items()}
    launch, score, audit, verification = (read_bound(paths[k], hashes[k]) for k in ('launch','score','audit','verification'))
    phase_rows = [json.loads(s) for s in paths['phase_log'].read_text().splitlines()]
    require(digest(paths['phase_log']) == hashes['phase_log'] and
            [r['phase'] for r in phase_rows] == [name for name, _ in commands(batch)] and
            all(r.get('passed') is True and r.get('log_sha256') == digest(batch/(r['phase']+'.log')) for r in phase_rows) and
            launch.get('sources') == comparator_sources() and launch.get('sequences') == list(ADDITIONAL),
            'The entire comparator batch must be frozen and audited')
    require_phase_artifacts(phase_rows, hashes)
    require(score.get('passed') is True and audit.get('passed') is True and audit.get('error_count') == 0 and
            audit.get('score_sha256') == hashes['score'] and audit.get('run_map_sha256') == hashes['run_map'] and
            audit.get('auditor_sha256') == digest(ROOT/'scripts/audit_additional_author_comparators_v2.py') and
            score.get('scorer_sha256') == digest(ROOT/'scripts/score_additional_author_comparators_v2.py') and
            score.get('verification_sha256') == hashes['verification'] and
            score.get('run_map_sha256') == hashes['run_map'] and
            verification.get('passed') is True and verification.get('label_guard', {}).get('open_attempts') == 0,
            'Comparator scoring or independent audit differs')
    setup_path, archived_path = CMSA/'setup.json', CMSA/'frames.json'
    setup_sha, archived_sha = digest(setup_path), digest(archived_path)
    setup, archived = read_bound(setup_path, setup_sha), read_bound(archived_path, archived_sha)
    repo = ROOT/'external/CMSA-Net'
    revision = subprocess.check_output(['git','-C',str(repo),'rev-parse','HEAD'], text=True).strip()
    dirty = subprocess.check_output(['git','-C',str(repo),'status','--porcelain','--untracked-files=no'], text=True)
    require(revision == setup['author_revision'] and not dirty and
            digest(repo/'cmsa_project/scripts/my_test.py') == setup['author_test_sha256'] and
            digest(CHECKPOINT) == setup['checkpoint_sha256'] and digest(DATA) == setup['manifest_sha256'] and
            setup.get('runner_sha256') == '871b7548ad43328e8961d4aff698a0896b661be6502181a2428db63f66b20a55' and
            setup.get('probability_threshold') == 128 and setup.get('device') == 'cpu' and
            'Author normalization retained' in setup.get('note', ''), 'Archived released-CMSA provenance differs')
    grouped = {seq: [r for r in archived if r['sequence'] == seq] for seq in SEQUENCES}
    videos = {**load_manifest(DATA,'dev'), **load_manifest(DATA,'test')}
    prior_sha, prior_audit_sha = digest(PRIOR), digest(PRIOR_AUDIT)
    prior, prior_audit = read_bound(PRIOR, prior_sha), read_bound(PRIOR_AUDIT, prior_audit_sha)
    plan = validate_plan(); records = {}; data_root = (ROOT/'data').resolve()
    for seq in SEQUENCES:
        frames = videos[seq]; old = grouped[seq]
        require([r['frame'] for r in old] == list(range(len(frames))), 'Archived CMSA frame coverage differs')
        if seq in PRIORITY:
            prior_frames, source_rows, start, _, _, baseline, _, prompt = sequence_inputs(plan, seq)
            verify_prior(prior, prior_audit, prior_sha, baseline, sequence=seq)
            meta = priority_metadata(plan, seq); images = [r['image_sha256'] for r in source_rows]
            reference = prior['rows'][seq]; ref_arm = 'A'
            require([f.frame_index for f in frames] == [f.frame_index for f in prior_frames], 'Priority source frames differ')
        else:
            inputs = verification['inputs'][seq]; start = inputs['prompt_frame']; images = inputs['images']
            meta = inputs['frame_metadata']; prompt = inputs['prompt']; reference = score['rows'][seq]; ref_arm = 'native'
        require(len(frames) == len(images) == len(reference), 'Reference frame coverage differs')
        predictions = []
        for i, (frame, row, old_row) in enumerate(zip(frames, reference, old, strict=True)):
            path = CMSA/'predictions'/frame.image_path.resolve().relative_to(data_root).with_suffix('.png')
            require(row['frame'] == i and row['source_frame_index'] == frame.frame_index and
                    row['post_prompt'] == (i > start) and row['initialization_frame'] == (i == start) and
                    digest(frame.image_path) == images[i] and digest(path) == old_row['prediction_sha256'],
                    'CMSA target frame, current image or archived prediction differs')
            with Image.open(frame.image_path) as source, Image.open(path) as image:
                require(image.format == 'PNG' and image.size == source.size, 'Archived prediction geometry differs')
            require(digest(path) == old_row['prediction_sha256'], 'Archived prediction changed during decode')
            predictions.append({'path': str(path), 'sha256': old_row['prediction_sha256']})
        require(digest(Path(meta['path'])) == meta['sha256'], 'Target label metadata changed')
        records[seq] = {'prompt_frame': start, 'images': images, 'frame_metadata': meta,
                        'predictions': predictions, 'reference_arm': ref_arm}
    return {'batch': str(batch), 'batch_hashes': hashes, 'cmsa_setup_sha256': setup_sha,
            'cmsa_frames_sha256': archived_sha, 'prior_sha256': prior_sha, 'prior_audit_sha256': prior_audit_sha,
            'records': records, 'source_sha256': source_hashes(), 'scorer_sha256': digest(Path(__file__))}


def score_rows(bound):
    batch = Path(bound['batch']); scores = read_bound(batch/'scores/results.json', bound['batch_hashes']['score'])
    prior = read_bound(PRIOR, bound['prior_sha256']); old = read_bound(CMSA/'frames.json', bound['cmsa_frames_sha256'])
    videos = {**load_manifest(DATA,'dev'), **load_manifest(DATA,'test')}; rows = {}
    for seq in SEQUENCES:
        data = bound['records'][seq]; frames = videos[seq]; start = data['prompt_frame']
        meta = read_bound(data['frame_metadata']['path'], data['frame_metadata']['sha256'])
        reference = prior['rows'][seq] if seq in PRIORITY else scores['rows'][seq]
        old_rows = [r for r in old if r['sequence'] == seq]; seq_rows = []
        require(len(frames) == len(meta) == len(reference) == len(old_rows), 'Score coverage differs')
        for i, (frame, ref, archived, pred) in enumerate(zip(frames, reference, old_rows, data['predictions'], strict=True)):
            label_sha, image_sha = meta[i]['label_sha256'], data['images'][i]
            require(digest(frame.mask_path) == label_sha and digest(frame.image_path) == image_sha and
                    digest(Path(pred['path'])) == pred['sha256'], 'Score input changed')
            truth = load_label_mask(frame.mask_path,[1]) == 1
            with Image.open(pred['path']) as image: mask = np.asarray(image.convert('L')) >= 128
            require(digest(frame.mask_path) == label_sha and digest(frame.image_path) == image_sha and
                    digest(Path(pred['path'])) == pred['sha256'], 'Score input changed during decode')
            reproduced = calculate_scores(mask, truth)
            require(bool(truth.any()) == archived['present'] == ref[data['reference_arm']]['present'] and
                    reproduced['dice'] == archived['dice'] and reproduced['iou'] == archived['iou'],
                    'Historical CMSA score or current label differs')
            seq_rows.append({'frame': i, 'source_frame_index': frame.frame_index, 'post_prompt': i > start,
                             'initialization_frame': i == start, 'cmsa': metric(mask, truth)})
        rows[seq] = seq_rows
    return rows


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--mode',choices=('verify','score'),required=True)
    p.add_argument('--batch',type=Path,required=True);p.add_argument('--verification',type=Path);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=new_output(a.output)
    if a.mode=='verify':
        guard=install_mask_open_guard(DATA);result=validate_inputs(a.batch)
        require(guard['open_attempts']==0,'Annotation opened during verification')
        result.update(passed=True,label_guard=dict(guard));out.mkdir(parents=True)
        (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'sequences':len(SEQUENCES)}));return
    require(a.verification is not None,'Separate no-label verification required')
    guard=install_phase_mask_guard(DATA);proof_sha=digest(a.verification);proof=read_bound(a.verification,proof_sha)
    require(proof.get('passed') is True and proof.get('label_guard',{}).get('open_attempts')==0,'Verification differs')
    current=validate_inputs(a.batch)
    require(all(proof.get(k)==v for k,v in current.items()) and guard['open_attempts']==0,'Frozen inputs changed before labels')
    out.mkdir(parents=True);guard['active']=False;rows=score_rows(proof)
    summaries={s:summary(rows[s],'cmsa') for s in SEQUENCES};eligible=[s for s in SEQUENCES if summaries[s]['present_frames']]
    count=sum(summaries[s]['present_frames'] for s in eligible)
    overall={'patients_with_present_frames':len(eligible),'present_frames':count,
             'patient_macro_present_dice':float(np.mean([summaries[s]['mean_present_dice'] for s in eligible])) if eligible else None,
             'frame_weighted_present_dice':sum(summaries[s]['mean_present_dice']*summaries[s]['present_frames'] for s in eligible)/count if count else None,
             'empty_frames':sum(summaries[s]['empty_frames'] for s in SEQUENCES),
             'false_alarms':sum(summaries[s]['false_alarms'] for s in SEQUENCES)}
    lines=['# Archived CMSA on automatic-prompt target frames','','Released column-wise normalization, saved-output threshold 128/255. '
           'CMSA is unprompted and uses reference/clip images, unlike initialized trackers; its archived input image hashes were not recorded. '
           'The historical setup and frame-record files also lack an original cryptographic completion binding, so current mask/label reproduction cannot prove which setup produced the saved predictions. '
           'All results are exploratory and seq23 is excluded because its historical input includes the inset.','',
           '| Sequence | Present frames | Mean present Dice | Labeled-empty alarms |','|---|---:|---:|---:|']
    for s in SEQUENCES:
        x=summaries[s];v=x['mean_present_dice'];v='N/A' if v is None else f'{v:.4f}'
        lines.append(f"| {s} | {x['present_frames']} | {v} | {x['false_alarms']}/{x['empty_frames']} |")
    lines += ['',f"Patient macro: {overall['patient_macro_present_dice']} on {overall['patients_with_present_frames']} patients; "
              f"frame weighted: {overall['frame_weighted_present_dice']} on {count} present frames.",
              'Post-prompt excludes initialization; labeled-empty is not proven clinical absence.']
    (out/'report.md').write_text('\n'.join(lines)+'\n')
    result={'passed':True,'scope':'20 previously inspected sequences; descriptive matched target indices, unprompted CMSA.',
            'verification_sha256':proof_sha,'scorer_sha256':digest(Path(__file__)),'report_sha256':digest(out/'report.md'),
            'prelabel_guard':dict(guard),'rows':rows,'summaries':summaries,'overall':overall}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'sequences':len(rows)}))


if __name__=='__main__':main()
