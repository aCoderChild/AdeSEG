#!/usr/bin/env python3
"""Independently audit and report four frozen automatic-prompt comparators.

SAM2Long selects its final trajectory offline. The three cases and earlier
comparators were inspected during development; this is not a locked test.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output
from scripts.run_samurai_automatic_priority import MANIFEST,prepare_archived_prompt
from scripts.score_sam2long_automatic_priority import check_frozen as check_long
from scripts.score_lean_window_mps_vs_samurai import check_frozen as check_samurai


def read(path):return json.loads(path.read_text())


def metric(pred,truth):
    if pred.shape!=truth.shape:raise ValueError('Prediction/label geometry differs')
    area=int(pred.sum());den=area+int(truth.sum())
    return {'present':bool(truth.any()),'dice':2*int((pred&truth).sum())/den if den else 1.,
      'false_alarm':bool(area) if not truth.any() else None,'predicted_pixels':area}


def summarize(rows,arm):
    rows=[row for row in rows if row['post_prompt']]
    present=[row[arm]['dice'] for row in rows if row[arm]['present']]
    empty=[row[arm]['predicted_pixels'] for row in rows if not row[arm]['present']]
    return {'present_frames':len(present),'mean_present_dice':float(np.mean(present)) if present else None,
      'empty_frames':len(empty),'false_alarms':sum(v>0 for v in empty),
      'mean_empty_predicted_pixels':float(np.mean(empty)) if empty else None}


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--experiment-root',required=True,type=Path);p.add_argument('--output',required=True,type=Path)
    a=p.parse_args();base=a.experiment_root;native=base/'native_sam2_automatic_priority_v1';leanbase=base/'contrast_refinement_v1';sam=base/'samurai_automatic_priority_v1';long=base/'sam2long_automatic_priority_v1'
    prior_path=native/'matched_three_system_report_attempt01/results.json';prior=read(prior_path)
    prior_audit=read(native/'alternate_score_audit_attempt01/audit.json')
    if (not prior_audit.get('passed') or prior_audit['result_sha256']!=digest(prior_path)
        or prior_audit['auditor_sha256']!=digest(ROOT/'scripts/audit_native_report_alternate.py')
        or prior['scorer_sha256']!=digest(ROOT/'scripts/report_native_lean_samurai_automatic.py')):
        raise ValueError('Prior three-system scores or independent audit differ')
    pre=base/'lean_channel_attention_training_pilot_v1/split_preflight_attempt03';pre_data=read(pre/'manifest.json')
    crop=base/'lean_token_window_candidate_v1/seq23_attempt01';backup=base/'lean_seq23_no_inset_v1/full_attempt02'
    videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')};results={};rows_all={};sources={};preprompt={}
    for seq in ('seq11','seq20','seq23'):
        frames=videos[seq];lp=leanbase/f'lean_mps_baseline_{seq}_full_attempt01';sp=sam/f'{seq}_full_mps_attempt02';npth=native/f'{seq}_full_mps_attempt01';xp=long/f'{seq}_full_mps_attempt01'
        lg=leanbase/f'lean_mps_baseline_{seq}_gate_attempt02/gate.json';sg=sam/f'{seq}_gate_verified_v3/gate.json';xg=long/f'{seq}_gate_verified_attempt03/gate.json'
        _,_,_,prompt=prepare_archived_prompt(seq,frames,pre,crop,backup)
        start,_,_=check_samurai(seq,frames,lp,sp,lg,sg,prompt)
        if check_long(seq,frames,lp,xp,lg,xg,prompt)!=start:raise ValueError('Prompt schedule differs')
        nr=read(npth/'receipt.json')
        if (not nr.get('passed') or not nr.get('full_video') or nr.get('sequence')!=seq or nr.get('prompt_frame')!=start
            or nr.get('prompt')!=prompt or nr['identity']['device']!='mps' or nr['identity']['fallback']!='0' or nr['identity']['fast_math']!='0'
            or nr['identity']['checkpoint_sha256']!=read(lp/'receipt.json')['identity']['checkpoint_sha256']
            or prior_audit['sequences'][seq]['native_receipt_sha256']!=digest(npth/'receipt.json')):
            raise ValueError('Native same-prompt receipt differs')
        if len(nr['records'])!=len(frames):raise ValueError('Native coverage differs')
        for i,frame in enumerate(frames):
            row=nr['records'][i]
            if (row['frame']!=i or row['source_frame_index']!=frame.frame_index or row['image_sha256']!=digest(frame.image_path)
                or row['mask_sha256']!=digest(npth/f'{i:06d}.png')):
                raise ValueError('Native frozen mask or input changed')
        long_score_path=long/f'{seq}_score_attempt01/results.json';long_score=read(long_score_path)
        if (long_score['scorer_sha256']!=digest(ROOT/'scripts/score_sam2long_automatic_priority.py')
            or long_score['lean_receipt_sha256']!=digest(lp/'receipt.json')
            or long_score['sam2long_receipt_sha256']!=digest(xp/'receipt.json') or len(long_score['rows'])!=len(frames)):
            raise ValueError('Offline SAM2Long score provenance differs')
        oldrows=prior['rows'][seq]
        if len(oldrows)!=len(frames):raise ValueError('Three-system frame coverage differs')
        meta=read(Path(pre_data['sequences'][seq]['source'])/seq/'frames.json');rows=[]
        for i,frame in enumerate(frames):
            if digest(frame.mask_path)!=meta[i]['label_sha256']:raise ValueError('Label bytes changed')
            with Image.open(frame.mask_path) as img:
                if img.format!='JPEG':raise ValueError('Unexpected label format')
                gray=np.asarray(img.convert('L'))
            if ((gray>64)&(gray<192)).mean()>0.05:raise ValueError('Label is not binary JPEG')
            truth=gray>127;out={'frame':i,'source_frame_index':frame.frame_index,'post_prompt':i>start,'initialization_frame':i==start}
            for arm,folder in (('native',npth),('lean',lp),('samurai',sp),('sam2long',xp)):
                if arm=='samurai' and i<start:pred=np.zeros_like(truth)
                else:
                    with Image.open(folder/f'{i:06d}.png') as img:pred=np.asarray(img.convert('L'))>0
                out[arm]=metric(pred,truth)
                expected=long_score['rows'][i][arm] if arm=='sam2long' else oldrows[i][arm]
                if out[arm]!=expected:raise ValueError(f'Independent {arm} metric differs at {seq}/{i}')
            if out['lean']!=long_score['rows'][i]['lean'] or any(out[k]!=oldrows[i][k] for k in ('frame','source_frame_index','post_prompt','initialization_frame')):
                raise ValueError('Matched frames or Lean scores differ')
            rows.append(out)
        results[seq]={arm:summarize(rows,arm) for arm in ('native','lean','samurai','sam2long')};rows_all[seq]=rows
        preprompt[seq]={'initialization_frame':start,'frames':start,
          'labeled_present_frames':sum(row['native']['present'] for row in rows if row['frame']<start),
          'labeled_empty_frames':sum(not row['native']['present'] for row in rows if row['frame']<start)}
        if any(row[arm]['predicted_pixels'] for row in rows if row['frame']<start for arm in ('native','lean','samurai','sam2long')):
            raise ValueError('Pre-prompt predictions must be empty')
        if results[seq]['sam2long']['present_frames']!=long_score['summaries']['sam2long']['post_prompt']['present_frames']:
            raise ValueError('Present denominator differs')
        sources[seq]={'native_receipt_sha256':digest(npth/'receipt.json'),'lean_receipt_sha256':digest(lp/'receipt.json'),
          'samurai_receipt_sha256':digest(sp/'receipt.json'),'sam2long_receipt_sha256':digest(xp/'receipt.json'),
          'sam2long_score_sha256':digest(long_score_path)}
    lines=['# Four automatic-mask video systems: same-MPS development comparison','','Each method uses the same archived automatic mask and initialization frame, same image sequence and the same fixed sequence-23 viewport. The table excludes the initialization frame and includes every later labeled frame. SAM2Long is **offline/noncausal**: it selects its final pathway after processing the complete sequence.','',
      '| Sequence | Present frames | Dice: native / project Lean / SAMURAI / SAM2Long | Labeled-empty false alarms: native / project Lean / SAMURAI / SAM2Long |',
      '|---|---:|---:|---:|']
    for seq in ('seq11','seq20','seq23'):
        d=results[seq];arms=('native','lean','samurai','sam2long')
        lines.append('| '+seq+' | '+str(d['lean']['present_frames'])+' | '+' / '.join(f"{d[x]['mean_present_dice']:.4f}" for x in arms)+' | '+' / '.join(f"{d[x]['false_alarms']}/{d[x]['empty_frames']}" for x in arms)+' |')
    lines += ['', 'Before initialization, all four methods output empty masks. Pre-prompt labeled-present / total frame counts are '+', '.join(f"{seq}: {preprompt[seq]['labeled_present_frames']}/{preprompt[seq]['frames']}" for seq in ('seq11','seq20','seq23'))+'. The post-prompt table therefore excludes these frames and also excludes the initialization frame; it does not measure detection delay or end-to-end detection recall.',
      '', 'All source images and frozen prediction masks were checked against receipts; JPEG labels were independently decoded and per-frame metrics recomputed for each system. The runs use the same SAM2.1 checkpoint bytes, but project Lean also loads a trained bypass and uses a 16,384-token window; SAMURAI and SAM2Long use their author memory-selection behavior and thresholds. No new detector or project confidence gate was added.',
      '', 'The same MPS device was used, with fast math and fallback disabled. SAMURAI uses a scoped float32 Kalman-IoU compatibility cast; native and project Lean use a scoped pointer-transfer correction. SAM2Long uses the pinned author implementation and reports no per-frame predictions until offline path selection completes. The native runner did not bind every helper hash before inference; supplementary hashes measured during the run cannot prove its exact pre-run identity.',
      '', 'The original YOLO procedure scanned chronological frames until its first detection; no detector calls occur in these replay runs. Seq23 crop and all three difficult videos were previously inspected. Patient independence, detector training overlap, annotation completeness, and native acquisition timing remain unknown. These results are exploratory and do not establish clinical or independent generalization. Full per-frame values and evidence hashes are in `results.json`.']
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    report=out/'report.md';report.write_text('\n'.join(lines)+'\n')
    result={'passed':True,'scope':'Independently rescored same-MPS exploratory development videos; SAM2Long noncausal.',
      'summaries':results,'preprompt':preprompt,'rows':rows_all,'source_hashes':sources,'prior_three_system_sha256':digest(prior_path),
      'prior_native_audit_sha256':digest(native/'alternate_score_audit_attempt01/audit.json'),
      'report_sha256':digest(report),'reporter_sha256':digest(Path(__file__))}
    (out/'results.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'output':str(out),'summaries':results},indent=2))


if __name__=='__main__':main()
