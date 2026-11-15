#!/usr/bin/env python3
"""Score both seeds only after all 32 exploratory video runs freeze.

Previously inspected development patients; this is not a blinded test. This
module validates all launch, gate, source, prompt, mask and route evidence
before opening any target label file. Each seed has its own matched A baseline.
"""
import argparse
import json
import os
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output
from scripts.run_samurai_automatic_priority import MANIFEST
from scripts.source_group_training_label_guard_v1 import mask_path_sets
from scripts.run_verified_refiner_priority_development import priority_inputs
from scripts.preflight_extended_boundary_video_v1 import extended_priority_inputs
from scripts.score_boundary_video_factorial_v1 import (_check_events,check_conditioning,check_telemetry,
    metric,summary,paired_summary)
from scripts.run_multiseed_boundary_factorial_v1 import (ARMS,BASE,BASELINE,CHECKPOINT,DETECTOR,
    DETECTOR_SHA256,PRE,REF,RUNNER,SEQ8_PRE,SEQ23,SEQ23_BACKUP,SEEDS,SEQUENCES,STUDY,
    checked_gates,inputs,sources)


def read(path):return json.loads(Path(path).read_text())
def require(ok,message):
    if not ok:raise ValueError(message)


def install_prelabel_guard():
    """Block manifest-mask opens during validation; disarm only after it passes.

    CPython audit events do not cover arbitrary native-code system calls.
    """
    train,protected=mask_path_sets(MANIFEST);blocked=train|protected
    state={'blocked_paths':len(blocked),'open_attempts':0,'active':True}
    def audit(event,args):
        if (not state['active'] or event!='open' or not args or
                not isinstance(args[0],(str,bytes,os.PathLike))):return
        if os.path.realpath(os.fsdecode(args[0])) in blocked:
            state['open_attempts']+=1
            raise PermissionError('Label mask blocked during pre-score validation')
    sys.addaudithook(audit)
    return state


def checked_output(path):
    out=validate_output(path)
    require(out.is_relative_to(BASE.resolve()) and not out.exists(),
            'Score output must be a new folder within the experiment root')
    return out


def mapped_runs(run):
    def unique(pairs):
        out={}
        for key,value in pairs:
            if key in out:raise ValueError('Duplicate map key')
            out[key]=value
        return out
    raw=json.loads((run/'run_map.json').read_text(),object_pairs_hook=unique)
    require(set(raw)=={str(s) for s in SEEDS},'Seed coverage differs')
    mapping={};paths=[]
    for seed in SEEDS:
        v=raw[str(seed)];require(set(v)==set(SEQUENCES),'Sequence coverage differs')
        mapping[str(seed)]={}
        for seq in SEQUENCES:
            require(set(v[seq])==set(ARMS),'Arm coverage differs')
            mapping[str(seed)][seq]={}
            for arm in ARMS:
                path=Path(v[seq][arm]).resolve(strict=True)
                require(path.is_relative_to(run),'Prediction outside run root')
                paths.append(path);mapping[str(seed)][seq][arm]=path
    require(len(paths)==32 and len(set(paths))==32,'Output directories must be distinct')
    return raw,mapping


def verify_launch(run,raw,*,score_complete=False):
    manifest=read(run/'run_manifest.json')
    require(manifest.get('run_map')==raw and manifest.get('seeds')==list(SEEDS) and
        manifest.get('sequences')==list(SEQUENCES) and manifest.get('arms')==list(ARMS) and
        manifest.get('python_hash_seed')=='0' and manifest.get('mps_fallback')=='0' and
        manifest.get('mps_fast_math')=='0','Frozen launch differs')
    src=sources();require(manifest.get('sources')=={k:{'path':str(v),'sha256':digest(v)} for k,v in src.items()},
        'Launch source differs')
    short_root=Path(manifest['short_root']).resolve(strict=True);gates=checked_gates(short_root)
    inp=inputs(short_root,gates)
    require(manifest.get('inputs')=={k:{'path':str(v),'sha256':digest(v)} for k,v in inp.items()},
        'Launch input differs')
    phases=[json.loads(x) for x in (run/'phases.jsonl').read_text().splitlines() if x.strip()]
    expected=[(f'{seed}_{seq}_{arm}',seed) for seed in SEEDS for seq in SEQUENCES for arm in ARMS]
    if score_complete: expected.append(('score',SEEDS[0]))
    require([(p.get('phase'),p.get('seed')) for p in phases]==expected and all(
        p.get('passed') is True and p.get('log_sha256')==digest(run/(p['phase']+'.log')) for p in phases),
        'All 32 full-run phases must finish before labels')
    return gates


def expected_inputs(seq):
    if seq=='seq8':
        return extended_priority_inputs(seq,SEQ8_PRE,CHECKPOINT,DETECTOR)
    return priority_inputs(seq,PRE,SEQ23,SEQ23_BACKUP,CHECKPOINT,DETECTOR)


def validate_all(mapping,gates):
    """Check all 32 runs; deliberately never open a target label."""
    expected={seq:expected_inputs(seq) for seq in SEQUENCES}
    receipts={}; telemetry={}; timing={}
    for seed in SEEDS:
        key=str(seed);receipts[key]={};telemetry[key]={};timing[key]={}
        candidate=digest(STUDY/key/'runs/region_boundary/checkpoint.pt')
        control=digest(STUDY/key/'runs/mask_only_boundary/checkpoint.pt')
        gate,audit=gates[seed];runtime_identity=read(gate)['runtime_identity']
        for seq in SEQUENCES:
            frames,prompt,prompt_sha,view,prompt_mask,provenance=expected[seq]
            start=prompt['first_detection_index'];receipts[key][seq]={};telemetry[key][seq]={};timing[key][seq]={}
            expected_images=[{'frame':i,'sha256':digest(frame.image_path)} for i,frame in enumerate(frames)]
            for arm in ARMS:
                folder=mapping[key][seq][arm];rec=read(folder/'completion.json')
                wanted={'passed':True,'full_video':True,'limit':0,'seed':seed,'sequence':seq,'arm':arm,
                    'runner_source_sha256':digest(RUNNER),'prompt_frame':start,'prompt_sha256':prompt_sha,
                    'prompt_mask_sha256':provenance['prompt_mask_sha256'],'view':view,
                    'checkpoint_sha256':digest(CHECKPOINT),'detector_sha256':DETECTOR_SHA256,
                    'candidate_checkpoint_sha256':candidate,'control_checkpoint_sha256':control,
                    'engineering_gate_sha256':digest(gate),'engineering_audit_sha256':digest(audit),
                    'seq8_preflight_sha256':digest(SEQ8_PRE/'manifest.json'),
                    'seq8_preflight_completion_sha256':digest(SEQ8_PRE/'completion.json')}
                require(all(rec.get(k)==v for k,v in wanted.items()) and
                    rec.get('label_guard',{}).get('open_attempts')==0 and rec.get('input_images')==expected_images,
                    'Run identity or guarded inputs differ')
                require([r['frame'] for r in rec['masks']]==list(range(len(frames))) and
                    [r['frame'] for r in rec['post_prompt_state']]==list(range(start,len(frames))),
                    'Frame coverage differs')
                runtime=_check_events(folder,rec)
                require(runtime==runtime_identity and runtime.get('seed')==seed,'Live model/seed differs')
                telemetry[key][seq][arm],timing[key][seq][arm]=check_telemetry(folder,rec,len(frames))
                for row in rec['masks']:
                    i=row['frame'];path=folder/f'{i:06d}.png'
                    require(digest(path)==row['sha256'],'Saved prediction changed')
                    with Image.open(frames[i].image_path) as image,Image.open(path) as mask:
                        values=np.asarray(mask.convert('L'))
                        require(mask.format=='PNG' and mask.size==image.size and np.isin(values,(0,255)).all(),
                            'Prediction format or geometry differs')
                        if i<start:require(not values.any(),'Pre-prompt prediction not empty')
                receipts[key][seq][arm]=rec
            a,b=receipts[key][seq]['A'],receipts[key][seq]['B']
            for arm in 'BCD':
                cur=receipts[key][seq][arm];check_conditioning(cur['post_prompt_state'][0],a['post_prompt_state'][0])
                for name in ('pred_masks_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                    require(cur['post_prompt_state'][1].get(name) is not None and
                        cur['post_prompt_state'][1][name]==a['post_prompt_state'][1][name],
                        'First native decoder state differs')
            first={arm:receipts[key][seq][arm]['post_prompt_state'][1] for arm in ARMS}
            require(first['A']['output_logits_sha256']==first['C']['output_logits_sha256'] and
                first['B']['output_logits_sha256']==first['D']['output_logits_sha256'] and
                first['C']['maskmem_features_sha256']==first['D']['maskmem_features_sha256'],
                'First-step routes differ')
            for cur,base in zip(b['post_prompt_state'],a['post_prompt_state'],strict=True):
                for name in ('pred_masks_sha256','maskmem_features_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                    require(cur.get(name) is not None and cur[name]==base.get(name),'B changed native state')
            for name in ('memory_events.json','positions.json','selections.json'):
                require(read(mapping[key][seq]['A']/name)==read(mapping[key][seq]['B']/name),'B changed memory trace')
    return expected,receipts,telemetry,timing


def label_metadata(videos):
    """Bind archived label hash metadata only after all predictions validate."""
    pre=read(PRE/'manifest.json'); ext=read(SEQ8_PRE/'manifest.json'); metadata={}
    for seq in SEQUENCES:
        if seq=='seq23':
            path=SEQ23/'frames.json';done=read(SEQ23/'completion.json');wanted=done['frames_sha256']
        elif seq=='seq8':
            item=ext['sequences'][seq];path=Path(item['archive_frames_path']);wanted=item['archive_frames_sha256']
        else:
            item=pre['sequences'][seq];path=Path(item['source'])/seq/'frames.json';wanted=item['frames_metadata_sha256']
        require(digest(path)==wanted,'Archived label metadata changed')
        metadata[seq]=read(path);require(len(metadata[seq])==len(videos[seq]),'Label metadata coverage differs')
    return metadata


def score_all(mapping,videos,receipts,metadata):
    rows={};summaries={};paired={};preprompt={}
    for seed in SEEDS:
        key=str(seed);rows[key]={};summaries[key]={};paired[key]={};preprompt[key]={}
        for seq in SEQUENCES:
            entries=[];start=receipts[key][seq]['A']['prompt_frame']
            for i,frame in enumerate(videos[seq]):
                label_sha=metadata[seq][i]['label_sha256']
                require(digest(frame.mask_path)==label_sha,'Label hash changed')
                gt=load_label_mask(frame.mask_path,[1])==1
                require(digest(frame.mask_path)==label_sha,'Label changed while decoding')
                row={'frame':i,'source_frame_index':frame.frame_index,'post_prompt':i>start,
                     'initialization_frame':i==start}
                for arm in ARMS:
                    path=mapping[key][seq][arm]/f'{i:06d}.png'
                    wanted=receipts[key][seq][arm]['masks'][i]['sha256']
                    require(digest(path)==wanted,'Prediction hash changed before decoding')
                    with Image.open(path) as image:
                        pred=np.asarray(image.convert('L'))>0
                    require(digest(path)==wanted,'Prediction changed while decoding')
                    row[arm]=metric(pred,gt)
                entries.append(row)
            rows[key][seq]=entries
            summaries[key][seq]={arm:{'post_prompt':summary(entries,arm),'all':summary(entries,arm,False)} for arm in ARMS}
            paired[key][seq]={name:paired_summary(entries,arm,base) for name,arm,base in
                (('B_minus_A','B','A'),('C_minus_A','C','A'),('D_minus_B','D','B'),('D_minus_A','D','A'))}
            preprompt[key][seq]={'frames':start,'present_misses':sum(r['A']['present'] for r in entries[:start])}
    return rows,summaries,paired,preprompt


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--run',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    a=p.parse_args();out=checked_output(a.output)
    run=a.run.resolve(strict=True);require(run.is_relative_to(BASE.resolve(strict=True)),'Run outside experiment root')
    guard=install_prelabel_guard()
    try:
        raw,mapping=mapped_runs(run);gates=verify_launch(run,raw)
        expected,receipts,telemetry,timing=validate_all(mapping,gates)
        videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')}
        # All predictions and label-independent evidence are checked before this boundary.
        metadata=label_metadata(videos)
        require(guard['open_attempts']==0,'Label open attempted during pre-score validation')
    finally:
        guard['active']=False
    rows,summaries,paired,preprompt=score_all(mapping,videos,receipts,metadata)
    out.mkdir(parents=True,exist_ok=False)
    lines=['# Exploratory two-seed A/B/C/D video factorial','',
        'Same selected, previously inspected patients and archived automatic prompts. Additional seeds are post-hoc; not a blinded or independent test. Post-prompt excludes initialization.','',
        '| Seed | Sequence | Present | Dice A / B / C / D | Labeled-empty alarms A / B / C / D |',
        '|---:|---|---:|---:|---:|']
    for seed in SEEDS:
        for seq in SEQUENCES:
            s=summaries[str(seed)][seq]
            dice=' / '.join(f"{s[x]['post_prompt']['mean_present_dice']:.4f}" if s[x]['post_prompt']['mean_present_dice'] is not None else 'NA' for x in ARMS)
            alarms=' / '.join(f"{s[x]['post_prompt']['false_alarms']}/{s[x]['post_prompt']['empty_frames']}" for x in ARMS)
            lines.append(f"| {seed} | {seq} | {s['A']['post_prompt']['present_frames']} | {dice} | {alarms} |")
    lines+=['','A: original output/memory; B: refined output; C: refined memory input; D: both. Initial conditioning and native object score/pointer are protected; C/D change future trajectories.','',
        'Project Lean is a modified token-window system. Original YOLO scanned until its first detection; archived prompts are replayed without redetection. Seq23 uses a retrospective inset crop. Labeled-empty is not confirmed clinical absence.','',
        'Each seed has a separately matched A baseline. Both seeds and all arms completed before labels were opened. Frames from one sequence are correlated; per-patient identity is user-reported, and detector/backbone exposure remains unknown.']
    report=out/'report.md';report.write_text('\n'.join(lines)+'\n')
    result={'passed':True,'scope':'Exploratory, previously inspected patients; no independent validation.',
        'summaries':summaries,'paired_descriptive':paired,'rows':rows,'preprompt':preprompt,
        'suppression_telemetry':telemetry,'timing':timing,
        'run_receipt_sha256':{str(seed):{s:{a:digest(mapping[str(seed)][s][a]/'completion.json') for a in ARMS} for s in SEQUENCES} for seed in SEEDS},
        'run_map_sha256':digest(run/'run_map.json'),'launch_manifest_sha256':digest(run/'run_manifest.json'),
        'prelabel_guard':dict(guard),'scorer_sha256':digest(Path(__file__)),'report_sha256':digest(report)}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(out),'summaries':summaries},indent=2))


if __name__=='__main__':main()
