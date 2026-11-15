#!/usr/bin/env python3
"""Build a provenance-bound mixed CPU/MPS fixed-read development manifest.

The prior 19 training and 10 validation reads were captured on CPU. The new
seq6/13 reads were captured on MPS using a separately verified runtime. This
manifest identifies both pipelines explicitly; it does not assert that device
features, temporal trajectories, patients or centers are independent.
"""
import argparse
import hashlib
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest as load_videos
from scripts.memory_token_io import digest, validate_output
from scripts.preflight_expanded_train_capture_v1 import TARGETS
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture
from scripts.train_region_conditioned_hardcase_v1 import load_manifest as load_previous


def read(path):
    return json.loads(Path(path).read_text())


def canonical_sha(value):
    return hashlib.sha256(json.dumps(value, sort_keys=True, separators=(',', ':')).encode()).hexdigest()


def require_no_cross_split_image_overlap(train_hashes, protected_hashes):
    for train_seq, train in train_hashes.items():
        for other_seq, other in protected_hashes.items():
            if train & other:
                raise ValueError(f'Exact source-image overlap: {train_seq} vs {other_seq}')


def validate_samples(samples):
    expected = {'seq2': 5, 'seq3': 5, 'seq4': 9, 'seq6': 10, 'seq13': 10}
    train = [x for x in samples if x['split'] == 'train']
    validation = [x for x in samples if x['split'] == 'validation']
    keys = [(x['sequence'], x['frame']) for x in samples]
    if len(samples) != 49 or len(set(keys)) != 49 or len(train) != 39 or len(validation) != 10:
        raise ValueError('Expected 39 train and 10 validation unique reads')
    if {x['sequence'] for x in train} != set(expected) or {x['sequence'] for x in validation} != {'seq9','seq10'}:
        raise ValueError('Unexpected sequence split')
    if any(sum(x['sequence'] == seq for x in train) != count for seq, count in expected.items()):
        raise ValueError('Unexpected training sequence coverage')
    if any(sum(x['sequence'] == seq for x in validation) != 5 for seq in ('seq9','seq10')):
        raise ValueError('Unexpected validation coverage')
    if any(x['capture_device'] != ('mps' if x['sequence'] in ('seq6','seq13') else 'cpu') for x in samples):
        raise ValueError('Incorrect per-sample capture device')
    for seq in ('seq6','seq13'):
        if sorted(x['frame'] for x in train if x['sequence'] == seq) != TARGETS[seq]:
            raise ValueError('Expanded target plan changed')


def validate_expanded(seq, preflight, pre_sha, cpu_pipeline, checkpoint_sha, samples_dir, reference, capture, verification, prompt_audit):
    sample_file = samples_dir/'samples.json'
    sample_done = read(samples_dir/'completion.json')
    report = read(sample_file)
    ref_path, capture_path = reference/'receipt.json', capture/'receipt.json'
    audit_path, prompt_path = verification/'verification.json', prompt_audit/'verification.json'
    ref, cap, audit, prompt = (read(p) for p in (ref_path,capture_path,audit_path,prompt_path))
    if (not sample_done.get('passed') or sample_done.get('samples_sha256') != digest(sample_file)
            or not report.get('passed') or report.get('sequence') != seq
            or report.get('runner_sha256') != digest(ROOT/'scripts/score_expanded_train_capture_v1.py')
            or report.get('preflight_sha256') != pre_sha
            or report.get('capture_verification_sha256') != digest(audit_path)
            or report.get('prompt_audit_sha256') != digest(prompt_path)
            or report.get('capture_receipt_sha256') != digest(capture_path)
            or report.get('cpu_training_feature_pipeline_sha256') != cpu_pipeline):
        raise ValueError(f'{seq}: sample audit identity differs')
    if (not ref.get('passed') or not cap.get('passed') or ref.get('mode') != 'reference'
            or cap.get('mode') != 'capture' or ref.get('sequence') != seq or cap.get('sequence') != seq
            or ref.get('identity') != cap.get('identity') or ref.get('targets') != TARGETS[seq]
            or cap.get('targets') != TARGETS[seq] or ref.get('rows') != cap.get('rows')):
        raise ValueError(f'{seq}: reference/capture trajectory differs')
    identity = cap['identity']
    if (identity.get('device') != 'mps' or identity.get('fallback') != '0' or identity.get('fast_math') != '0'
            or identity.get('deterministic_algorithms') is not True or identity.get('checkpoint_sha256') != checkpoint_sha
            or identity.get('cpu_feature_pipeline_sha256') != cpu_pipeline or identity.get('preflight_sha256') != pre_sha
            or identity.get('capacity') != 16384 or identity.get('insurance_reserve') != 3072
            or identity.get('anchor_strength') != 0.5 or canonical_sha(identity) != report['mps_runtime_identity_sha256']):
        raise ValueError(f'{seq}: MPS runtime or pipeline identity differs')
    sources = {'runner_sha256':'scripts/run_expanded_train_capture_v1.py',
               'observer_sha256':'modeling/lean_mps_decoder_observer.py',
               'token_policy_sha256':'modeling/token_memory_window.py',
               'token_runtime_sha256':'modeling/lean_token_window_storage.py',
               'mps_indexed_sha256':'modeling/lean_indexed_spatial_rope_mps.py',
               'mps_blocking_sha256':'modeling/lean_mps_blocking_pointer_position.py',
               'anchor_sha256':'modeling/lean_multiscale_anchor.py'}
    if any(identity[key] != digest(ROOT/path) for key,path in sources.items()):
        raise ValueError(f'{seq}: MPS capture implementation changed')
    fork = ROOT/'external/Lean-SAM2-token-window'
    if (subprocess.check_output(['git','-C',str(fork),'rev-parse','HEAD'],text=True).strip()!=identity['fork_revision']
            or any(digest(fork/path)!=sha for path,sha in identity['fork_patches'].items())):
        raise ValueError(f'{seq}: MPS capture fork changed')
    if (not audit.get('passed') or audit.get('sequence') != seq or audit.get('targets') != TARGETS[seq]
            or audit.get('auditor_sha256') != digest(ROOT/'scripts/audit_expanded_train_capture_v1.py')
            or audit.get('reference_receipt_sha256') != digest(ref_path) or audit.get('capture_receipt_sha256') != digest(capture_path)
            or not prompt.get('passed') or prompt.get('sequence') != seq
            or prompt.get('runner_sha256') != digest(ROOT/'scripts/audit_expanded_initial_prompt_v1.py')
            or prompt.get('reference_receipt_sha256') != digest(ref_path) or prompt.get('capture_receipt_sha256') != digest(capture_path)):
        raise ValueError(f'{seq}: capture or prompt verification differs')
    spec = preflight['sequences'][seq]
    if (cap['prompt_frame'] != spec['prompt_frame'] or cap['prompt_sha256'] != spec['prompt_sha256']
            or len(cap['rows']) != spec['frames'] - spec['prompt_frame']):
        raise ValueError(f'{seq}: prompt or output coverage differs')
    if any(ref[name] != cap[name] for name in ('positions','selection','anchor_events','token_window_events')):
        raise ValueError(f'{seq}: memory or position trace differs')
    for row in cap['rows']:
        if digest(capture/f"{row['frame']:06d}.png") != row['mask_sha256'] or digest(reference/f"{row['frame']:06d}.png") != row['mask_sha256']:
            raise ValueError(f'{seq}: captured video mask differs')
    observations = {x['frame']:x for x in cap['observations']}
    if sorted(observations) != TARGETS[seq] or len(observations) != len(cap['observations']):
        raise ValueError(f'{seq}: feature capture coverage differs')
    if sorted(x['frame'] for x in report['samples']) != TARGETS[seq]:
        raise ValueError(f'{seq}: sample set differs')
    videos = load_videos(ROOT/'data/polypgen_sequence.jsonl','dev')[seq]
    source_frames = Path(spec['source'])/seq/'frames.json'
    if digest(source_frames) != spec['source_frames_sha256']:
        raise ValueError(f'{seq}: source metadata changed')
    source_rows = read(source_frames)
    samples = []
    for item in report['samples']:
        frame = item['frame']; obs = observations[frame]; source = videos[frame]
        if (item['split'] != 'train' or item['sequence'] != seq or item['capture_device'] != 'mps'
                or Path(item['image']).resolve() != source.image_path.resolve()
                or Path(item['label']).resolve() != source.mask_path.resolve()
                or item['image_sha256'] != spec['image_sha256'][frame]
                or item['label_sha256'] != source_rows[frame]['label_sha256']
                or item['feature_sha256'] != obs['feature_sha256'] or item['snapshot_sha256'] != obs['snapshot_sha256']
                or Path(item['feature']).resolve() != Path(obs['feature_file']).resolve()
                or Path(item['snapshot']).resolve() != Path(obs['snapshot_file']).resolve()
                or not Path(item['feature']).resolve().is_relative_to(capture.resolve())
                or not Path(item['snapshot']).resolve().is_relative_to(capture.resolve())):
            raise ValueError(f'{seq}/{frame}: sample provenance differs')
        for key in ('image','label','feature','snapshot'):
            if digest(item[key]) != item[key+'_sha256']:
                raise ValueError(f'{seq}/{frame}: sample content changed: {key}')
        read_capture(item)
        samples.append({**item, 'capture_pipeline_sha256': report['mps_runtime_identity_sha256']})
    return samples, identity, {'samples_path':str(sample_file.resolve()), 'samples_sha256':digest(sample_file),
                              'reference_path':str(ref_path.resolve()),'reference_sha256':digest(ref_path),
                              'capture_path':str(capture_path.resolve()),'capture_sha256':digest(capture_path),
                              'verification_path':str(audit_path.resolve()),'verification_sha256':digest(audit_path),
                              'prompt_audit_path':str(prompt_path.resolve()),'prompt_audit_sha256':digest(prompt_path)}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for name in ('cpu-preflight','split-preflight','expanded-preflight','seq6-samples','seq6-reference','seq6-capture','seq6-verification','seq6-prompt-audit',
                 'seq13-samples','seq13-reference','seq13-capture','seq13-verification','seq13-prompt-audit','output'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args()
    cpu, cpu_sha=load_previous(args.cpu_preflight)
    split_path=args.split_preflight/'manifest.json'; split=read(split_path); split_done=read(args.split_preflight/'completion.json')
    if (not split_done['passed'] or split_done['manifest_sha256']!=digest(split_path)
            or cpu['split_sha256']!=digest(split_path)):
        raise ValueError('Original sequence split changed')
    exp_path=args.expanded_preflight/'manifest.json';exp=read(exp_path);exp_done=read(args.expanded_preflight/'completion.json');exp_sha=digest(exp_path)
    if (not exp_done['passed'] or exp_done['manifest_sha256']!=exp_sha or not exp['passed']
            or exp['runner_sha256']!=digest(ROOT/'scripts/preflight_expanded_train_capture_v1.py')
            or exp['protocol_sha256']!=digest(ROOT/'protocols/EXPANDED_TRAIN_CAPTURE_SEQ6_SEQ13_V1.md')
            or exp['data_manifest_sha256']!=digest(ROOT/'data/polypgen_sequence.jsonl')
            or exp['checkpoint_sha256']!=cpu['checkpoint_sha256']
            or exp['cpu_feature_pipeline_sha256']!=cpu['feature_pipeline_sha256']):
        raise ValueError('Expanded preflight or CPU pipeline differs')
    train_hashes={seq:set(exp['sequences'][seq]['image_sha256']) for seq in ('seq6','seq13')}
    protected={}
    for seq in ('seq8','seq9','seq10','seq11','seq20','seq23'):
        meta=split['sequences'][seq];path=Path(meta['source'])/seq/'frames.json'
        if digest(path)!=meta['frames_metadata_sha256']:
            raise ValueError(f'{seq}: protected sequence metadata changed')
        protected[seq]={x['image_sha256'] for x in read(path)}
    require_no_cross_split_image_overlap(train_hashes,protected)
    samples=[]
    for old in cpu['samples']:
        for key in ('image','label','feature','snapshot'):
            if digest(old[key])!=old[key+'_sha256']:
                raise ValueError('Earlier sample changed')
        read_capture(old)
        samples.append({**old,'capture_device':'cpu','capture_pipeline_sha256':cpu['feature_pipeline_sha256']})
    identities={};sources={}
    for seq in ('seq6','seq13'):
        rows,identity,receipt=validate_expanded(seq,exp,exp_sha,cpu['feature_pipeline_sha256'],cpu['checkpoint_sha256'],
            getattr(args,seq+'_samples'),getattr(args,seq+'_reference'),getattr(args,seq+'_capture'),
            getattr(args,seq+'_verification'),getattr(args,seq+'_prompt_audit'))
        samples.extend(rows);identities[seq]=identity;sources[seq]=receipt
    validate_samples(samples)
    if identities['seq6']!=identities['seq13']:
        raise ValueError('Expanded captures have different MPS runtime identities')
    mix={'schema':'mixed_cpu_mps_fixed_read_v1','cpu_feature_pipeline_sha256':cpu['feature_pipeline_sha256'],
         'mps_runtime_identity_sha256':canonical_sha(identities['seq6']),
         'warning':'CPU and MPS features are combined; seq4 parity does not prove equivalence for other videos.'}
    out=validate_output(args.output);out.mkdir(parents=True,exist_ok=False)
    result={'passed':True,'samples':samples,'checkpoint_sha256':cpu['checkpoint_sha256'],
            'feature_pipeline_sha256':canonical_sha(mix),'pipeline_identity':mix,
            'cpu_manifest_path':str((args.cpu_preflight/'manifest.json').resolve()),'cpu_manifest_sha256':cpu_sha,
            'expanded_manifest_path':str(exp_path.resolve()),'expanded_manifest_sha256':exp_sha,
            'split_manifest_path':str(split_path.resolve()),'split_manifest_sha256':digest(split_path),
            'expanded_sources':sources,'source_sha256':digest(Path(__file__)),
            'selection':'The four seq4 extra cases were label-selected during development; seq6/13 target indices were frozen before label scoring.',
            'scope':'Exploratory 39 train/10 previously inspected validation reads. Different capture devices; patient/center and training-exposure independence unknown.',
            'duplicate_audit':'Exact byte hashes across all seq6/13 images vs all seq8/9/10/11/20/23 images; not decoded near-duplicates or patient-level separation.'}
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'manifest_sha256':digest(out/'manifest.json')},indent=2)+'\n')
    print(json.dumps({'passed':True,'train':39,'validation':10,'composite_pipeline_sha256':result['feature_pipeline_sha256']}))


if __name__=='__main__':
    main()
