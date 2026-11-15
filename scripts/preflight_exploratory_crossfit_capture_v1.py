#!/usr/bin/env python3
"""Freeze a label-free five-fold index/capture feasibility plan; no GPU work.

All 23 sequences were previously inspected. This is not a held-out final test,
and this plan does not itself authorize capturing features or training models.
"""
import argparse
import hashlib
import json
import re
import shutil
import sys
from pathlib import Path

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest
from scripts.preflight_author_trainval_v1 import trainval_inputs
from scripts.preflight_remaining_boundary_video_v1 import image_metadata_only
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_refiner_shared_v1 import load_prepared,dependencies as existing_dependencies
from scripts.run_boundary_video_factorial_v1 import BASE,CHECKPOINT,DETECTOR,OUT,PRE
from modeling.seq23_viewport import VIEWPORT

DATA=ROOT/'data/polypgen_sequence.jsonl'
PROTOCOL=ROOT/'protocols/EXPLORATORY_CROSSFIT_CAPTURE_PREFLIGHT_V1.md'
EXISTING=OUT/'source_group_boundary_factorial_v1_bvqxd475/manifest'
COMBINED=OUT/'combined_12_boundary_factorial_v1_attempt01/receipt.json'
COMBINED_AUDIT=OUT/'combined_12_boundary_factorial_v1_independent_audit_attempt01/audit.json'
NOINIT_AUDIT=OUT/'unarchived_auto_prompt_targets_v1_independent_audit_attempt01/audit.json'
FOLDS=(('seq6','seq10','seq11','seq20','seq23'),('seq1','seq2','seq3','seq4','seq5'),
       ('seq7','seq8','seq9','seq15'),('seq12','seq13','seq14','seq16','seq17'),
       ('seq18','seq19','seq21','seq22'))
NOINIT=('seq1','seq7')
EXISTING_SEQUENCES=('seq2','seq3','seq4','seq5','seq9','seq12','seq13','seq14','seq15')
FRESH=('seq6','seq8','seq10','seq11','seq16','seq17','seq18','seq19','seq20','seq21','seq22','seq23')
SOURCES=('scripts/preflight_exploratory_crossfit_capture_v1.py',
         'protocols/EXPLORATORY_CROSSFIT_CAPTURE_PREFLIGHT_V1.md','datasets/video.py','datasets/io.py',
         'scripts/memory_token_io.py','scripts/source_group_label_guard_v1.py',
         'scripts/source_group_refiner_shared_v1.py','scripts/preflight_author_trainval_v1.py',
         'scripts/preflight_remaining_boundary_video_v1.py','scripts/run_boundary_video_factorial_v1.py',
         'scripts/preflight_extended_boundary_video_v1.py','modeling/samurai_auto_prompt.py',
         'modeling/seq23_viewport.py','scripts/freeze_source_group_runtime_v1.py',
         'scripts/prepare_source_group_refiner_manifest_v1.py','scripts/run_source_group_capture_v1.py',
         'scripts/source_group_training_label_guard_v1.py','scripts/train_multilevel_refiner_seq4_empty_v2.py',
         'scripts/preflight_source_group_capture_v1.py','protocols/source_group_capture_v1.json',
         'scripts/audit_combined_12_boundary_factorial_v1.py','scripts/audit_unarchived_no_detection_targets_v1.py')


def require(value,message):
    if not value:raise ValueError(message)


def project_field(path,key):
    """Deserialize only a named top-level field, never unrelated result tables."""
    raw=Path(path).read_text(); match=re.search(r'^  '+re.escape(json.dumps(key))+r'\s*:\s*',raw,re.M)
    require(match is not None, f'Missing evidence field: {key}')
    return json.JSONDecoder().raw_decode(raw,match.end())[0]


def uniform_indices(start,count,limit=12):
    require(type(start) is int and type(count) is int and 0<=start<count-1,'No usable post-prompt frames')
    n=min(limit,count-start-1)
    selected=np.rint(np.linspace(start+1,count-1,n)).astype(int).tolist()
    require(len(selected)==len(set(selected))==n and all(start<i<count for i in selected),'Invalid index grid')
    return selected


def verify_folds():
    flat=[s for fold in FOLDS for s in fold]
    require(len(flat)==len(set(flat))==23 and set(flat)=={f'seq{i}' for i in range(1,24)},'Fold coverage differs')
    owner={s:i for i,fold in enumerate(FOLDS) for s in fold}
    for group in (('seq1','seq2'),('seq3','seq4'),('seq6','seq10'),('seq7','seq8')):
        require(len({owner[s] for s in group})==1,'Suspected shared source spans folds')
    return owner


def register_image(seen,key,sequence,index,owner,kind):
    prior=seen.get(key)
    if prior and prior[0]!=sequence:
        require(owner[prior[0]]==owner[sequence],f'Cross-fold {kind} image duplicate: {prior[0]} / {sequence}')
        return {'kind':kind,'sequences':sorted((prior[0],sequence))}
    seen[key]=(sequence,index)
    return None


def validate_baseline_rows(sequence,rows,frames,supplemental_indices=None):
    require(len(rows)==len(frames),'Baseline image coverage differs')
    legacy=sequence in ('seq11','seq20','seq23')
    require(not legacy or (supplemental_indices is not None and len(supplemental_indices)==len(rows)),
            'Historical baseline requires separately bound source indices')
    for i,(row,frame) in enumerate(zip(rows,frames,strict=True)):
        require(row['frame']==i and row['sha256']==frame['image_sha256'], 'Baseline input bytes differ')
        if 'source_frame_index' in row:
            require(row['source_frame_index']==frame['source_frame_index'],'Baseline source frame index differs')
        else:
            require(legacy and supplemental_indices[i]==frame['source_frame_index'],
                    'Source index not bound by baseline or archived metadata')
    return ('archived frame metadata and current manifest; not recorded in baseline completion'
            if legacy else 'recorded in baseline completion and current manifest')


def prior_baselines():
    require(project_field(COMBINED_AUDIT,'passed') is True and
            project_field(COMBINED_AUDIT,'receipt_sha256')==digest(COMBINED) and
            project_field(COMBINED_AUDIT,'audit_source_sha256')==digest(ROOT/'scripts/audit_combined_12_boundary_factorial_v1.py'),
            'Independent combined baseline audit differs')
    entries=project_field(COMBINED,'completions'); records={}
    for row in entries:
        path=Path(row['path']).resolve(strict=True)
        if path.parent.name!='A':continue
        sequence=path.parent.parent.name
        require(sequence in FRESH and sequence not in records and path.is_relative_to(OUT.resolve()),'Baseline mapping differs')
        require(digest(path)==row['sha256'],'Baseline receipt changed')
        rec=json.loads(path.read_text());require(digest(path)==row['sha256'],'Baseline receipt changed while reading')
        require(rec.get('passed') is True and rec.get('arm')=='A' and rec.get('sequence')==sequence and
                rec.get('full_video') is True and rec.get('checkpoint_sha256')==digest(CHECKPOINT) and
                rec.get('detector_sha256')==digest(DETECTOR) and rec.get('label_guard',{}).get('open_attempts')==0,
                'Baseline identity differs')
        provenance=rec['prompt_provenance'];prompt_path=Path(provenance['prompt_file']).resolve(strict=True)
        require(prompt_path.is_relative_to(BASE.resolve()) and digest(prompt_path)==provenance['prompt_mask_sha256'],
                'Frozen prompt mask changed')
        meta_path=prompt_path.parent/'prompt.json'
        require(digest(meta_path)==provenance['prompt_metadata_sha256'],'Frozen prompt metadata changed')
        prompt_meta=json.loads(meta_path.read_text())
        require(prompt_meta.get('first_detection_index')==rec.get('prompt_frame'), 'Prompt frame metadata differs')
        records[sequence]=(path,row['sha256'],rec)
    require(set(records)==set(FRESH),'Twelve fresh-capture baselines not found')
    return records


def noinit_records():
    require(project_field(NOINIT_AUDIT,'passed') is True and
            project_field(NOINIT_AUDIT,'audit_source_sha256')==digest(ROOT/'scripts/audit_unarchived_no_detection_targets_v1.py') and
            project_field(NOINIT_AUDIT,'label_guard').get('open_attempts')==0,'No-initialization audit differs')
    results={}
    all_completions=project_field(NOINIT_AUDIT,'completion_sha256')
    for seq in NOINIT:
        seen_pid=set();seen_uuid=set();items=[]
        require(len(all_completions[seq])==2,'No-initialization replay coverage differs')
        for entry in all_completions[seq]:
            folder=Path(entry['folder']).resolve(strict=True);path=folder/'completion.json'
            require(folder.is_relative_to(OUT.resolve()) and digest(path)==entry['completion_sha256'], 'No-init completion changed')
            rec=json.loads(path.read_text());require(digest(path)==entry['completion_sha256'],'No-init completion changed during read')
            prompt=folder/'prompt.json';require(digest(prompt)==rec['prompt_sha256'],'No-init prompt changed')
            p=json.loads(prompt.read_text())
            require(rec.get('passed') is True and rec.get('sequence')==seq and rec.get('label_guard',{}).get('open_attempts')==0 and
                    rec['signature']['checkpoint_sha256']==digest(CHECKPOINT) and rec['signature']['detector_sha256']==digest(DETECTOR) and
                    p.get('first_detection_index') is None and p.get('status')=='no_detection', 'No-init outcome differs')
            seen_pid.add(rec['pid']);seen_uuid.add(rec['run_uuid']);items.append({'completion_path':str(path),'completion_sha256':digest(path),
                'prompt_sha256':digest(prompt),'pid':rec['pid'],'run_uuid':rec['run_uuid']})
        require(len(seen_pid)==len(seen_uuid)==2,'No-init replays are not independent processes')
        results[seq]=items
    return results


def build_plan(guard):
    owner=verify_folds()
    videos={**load_manifest(DATA,'dev'),**load_manifest(DATA,'test')}
    require(set(videos)==set(owner),'Dataset sequence coverage differs')
    old,old_sha=load_prepared(EXISTING)
    require(old.get('checkpoint_sha256')==digest(CHECKPOINT),'Frozen capture backbone differs')
    old_samples=old['samples'];require(len(old_samples)==214 and {x['sequence'] for x in old_samples}==set(EXISTING_SEQUENCES),
                                      'Existing capture coverage differs')
    seen_byte={};seen_rgb={};frame_hashes={};same_fold_duplicates=[]
    for seq in sorted(videos):
        frame_hashes[seq]=[]
        for i,frame in enumerate(videos[seq]):
            before=digest(frame.image_path)
            with Image.open(frame.image_path) as image:
                rgb=image.convert('RGB');decoded=hashlib.sha256(str(rgb.size).encode('ascii')+rgb.tobytes()).hexdigest()
            require(digest(frame.image_path)==before,'Image changed during duplicate scan')
            for kind,key,seen in (('byte',before,seen_byte),('decoded',decoded,seen_rgb)):
                duplicate=register_image(seen,key,seq,i,owner,kind)
                if duplicate:same_fold_duplicates.append(duplicate)
            frame_hashes[seq].append({'frame':i,'source_frame_index':frame.frame_index,'image_sha256':before})
    starts={};prompts={}
    for seq in EXISTING_SEQUENCES:
        frames,start,mask,provenance=trainval_inputs(seq,CHECKPOINT,DETECTOR)
        require(len(frames)==len(videos[seq]) and mask.any(),'Existing automatic prompt differs')
        starts[seq]=start;prompts[seq]={'initial_frame':start,'view':{'kind':'full_source_frame'},**provenance}
    baselines=prior_baselines()
    for seq,(path,sha,rec) in baselines.items():
        rows=rec['input_images'];supplemental=None;index_metadata=None
        if seq in ('seq11','seq20','seq23'):
            split_path=PRE/'manifest.json';split_done=json.loads((PRE/'completion.json').read_text())
            require(split_done.get('passed') is True and split_done.get('manifest_sha256')==digest(split_path) and
                    rec['prompt_provenance']['preflight_sha256']==digest(split_path), 'Historical split preflight differs')
            split=json.loads(split_path.read_text());entry=split['sequences'][seq]
            file=Path(entry['source'])/seq/'frames.json'
            require(digest(file)==entry['frames_metadata_sha256'],'Historical image metadata changed')
            supplemental,archived_hashes=image_metadata_only(file)
            require(archived_hashes==[x['image_sha256'] for x in frame_hashes[seq]],'Historical image metadata differs')
            index_metadata={'path':str(file),'sha256':digest(file),'split_preflight_sha256':digest(split_path)}
        source_index_evidence=validate_baseline_rows(seq,rows,frame_hashes[seq],supplemental)
        with Image.open(videos[seq][0].image_path) as image:width,height=image.size
        expected_view={'kind':'fixed_seq23_main_viewport','xyxy':list(VIEWPORT)} if seq=='seq23' else {'kind':'full_source_frame'}
        expected_roi=list(VIEWPORT) if seq=='seq23' else [0,0,width,height]
        require(rec['view']==expected_view and rec['prompt_provenance']['roi_xyxy']==expected_roi,
                'Baseline view or prompt ROI differs')
        start=rec['prompt_frame'];require(type(start) is int and 0<=start<len(rows)-1,'Baseline prompt ordinal invalid')
        starts[seq]=start
        prompts[seq]={'initial_frame':start,'view':rec['view'],'prompt_provenance':rec['prompt_provenance'],
                      'source_index_evidence':source_index_evidence,'source_index_metadata':index_metadata,
                      'baseline_completion_path':str(path),'baseline_completion_sha256':sha}
    existing=[]
    for item in old_samples:
        seq,i=item['sequence'],item['frame']
        require(seq in EXISTING_SEQUENCES and type(i) is int and starts[seq]<i<len(videos[seq]) and
                item['image_sha256']==frame_hashes[seq][i]['image_sha256'],'Existing read input differs')
        existing.append({'sequence':seq,'frame':i,'image_sha256':item['image_sha256'],
                         'feature_path':item['feature'],'feature_sha256':item['feature_sha256'],
                         'snapshot_path':item['snapshot'],'snapshot_sha256':item['snapshot_sha256']})
    require(len({(r['sequence'],r['frame']) for r in existing})==214,'Existing reads are duplicated')
    fresh=[]
    for seq in FRESH:
        for i in uniform_indices(starts[seq],len(videos[seq])):
            fresh.append({'sequence':seq,**frame_hashes[seq][i]})
    require(len(fresh)==143,'Fresh-read grid count differs')
    noinit=noinit_records()
    counts={s:sum(r['sequence']==s for r in existing+fresh) for s in owner}
    require(all(counts[s]>0 for s in owner if s not in NOINIT) and all(counts[s]==0 for s in NOINIT),
            'Initialized/no-init capture coverage differs')
    splits=[]
    for i,test in enumerate(FOLDS):
        val=FOLDS[(i+1)%len(FOLDS)];train=tuple(s for j,fold in enumerate(FOLDS) if j not in (i,(i+1)%len(FOLDS)) for s in fold)
        initialized=[s for s in train if counts[s]>0]
        weights={s:{'numerator':1,'denominator':len(initialized)*counts[s]} for s in initialized}
        require(abs(sum(counts[s]/weights[s]['denominator'] for s in initialized)-1)<1e-12,'Sequence loss weights differ')
        splits.append({'outer_fold':i+1,'test':list(test),'validation':list(val),'train':list(train),
                       'per_frame_loss_weight':weights,'initialized_train_sequences':len(initialized)})
    require(guard['open_attempts']==0,'Annotation-mask open attempted during preflight')
    free=shutil.disk_usage(OUT).free
    return {'passed':True,'scope':'Previously inspected exploratory groups; not blind or independently confirmed.',
        'patient_mapping':'User states each sequence is one distinct patient; no independent patient-ID table was available.',
        'limitation':'Detector/backbone exposure, patient/exam sources and near-duplicate frames remain unresolved.',
        'capture_authorized':False,'existing_manifest_path':str(EXISTING/'manifest.json'),'existing_manifest_sha256':old_sha,
        'feature_pipeline_sha256':old['feature_pipeline_sha256'],'checkpoint_sha256':digest(CHECKPOINT),'detector_sha256':digest(DETECTOR),
        'data_manifest_sha256':digest(DATA),'combined_receipt_sha256':digest(COMBINED),'combined_audit_sha256':digest(COMBINED_AUDIT),
        'noinit_audit_sha256':digest(NOINIT_AUDIT),'source_sha256':{p:digest(ROOT/p) for p in SOURCES},
        'existing_capture_dependency_sha256':existing_dependencies(),
        'folds':splits,'prompts':prompts,'no_initialization':noinit,'existing_reads':existing,'fresh_reads':fresh,
        'source_frames':frame_hashes,'source_frames_checked':sum(map(len,frame_hashes.values())),
        'capture_counts':counts,'fresh_payload_estimate_bytes':143*41*1024**2,
        'free_space_bytes_at_preflight':free,'minimum_reserve_bytes':22*1024**3,
        'same_fold_exact_or_decoded_matches':same_fold_duplicates,
        'near_duplicate_screen':'not performed; shared exams and resized/adjacent frames remain unresolved',
        'label_guard':dict(guard)}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    out=a.output.resolve();require(out.is_relative_to(BASE.resolve()) and not out.exists(),'Output must be a new Drive experiment folder')
    guard=install_mask_open_guard(DATA);plan=build_plan(guard)
    require(guard['open_attempts']==0,'Label opened during plan');out.mkdir(parents=True,exist_ok=False)
    file=out/'plan.json';file.write_text(json.dumps(plan,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'plan_sha256':digest(file),'label_guard':dict(guard)},indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(out),'existing':len(plan['existing_reads']),
                      'fresh':len(plan['fresh_reads']),'capture_authorized':False,'label_open_attempts':guard['open_attempts']}))


if __name__=='__main__':main()
