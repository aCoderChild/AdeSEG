#!/usr/bin/env python3
"""Independently audit and report frozen same-MPS Lean/SAMURAI results.

Three previously inspected development sequences only. All source and output
mask hashes are checked before JPEG labels are read; no inference is performed.
"""
import argparse
import json
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.audit_lean_window_baseline_mps_full import audit as audit_cpu_mps
from scripts.memory_token_io import digest,validate_output
from scripts.run_samurai_automatic_priority import MANIFEST,prepare_archived_prompt
from scripts.score_lean_window_mps_vs_samurai import check_frozen


def read(path):return json.loads(path.read_text())


def independent_metric(pred,truth):
    if pred.shape!=truth.shape:raise ValueError('Mask geometry differs')
    n=int(pred.sum());den=n+int(truth.sum())
    return {'present':bool(truth.any()),'dice':2*int((pred&truth).sum())/den if den else 1.,
            'false_alarm':bool(n) if not truth.any() else None,'predicted_pixels':n}


def independent_summary(rows,arm,post):
    rows=[x for x in rows if not post or x['post_prompt']]
    present=[x[arm]['dice'] for x in rows if x[arm]['present']]
    empty=[x[arm] for x in rows if not x[arm]['present']]
    return {'frames':len(rows),'present_frames':len(present),'mean_present_frame_dice':float(np.mean(present)) if present else None,
      'empty_frames':len(empty),'false_alarm_frames':sum(x['false_alarm'] for x in empty),
      'mean_empty_predicted_pixels':float(np.mean([x['predicted_pixels'] for x in empty])) if empty else None}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--experiment-root',type=Path,required=True);parser.add_argument('--output',type=Path,required=True)
    a=parser.parse_args();base=a.experiment_root;v=base/'contrast_refinement_v1';s=base/'samurai_automatic_priority_v1'
    preflight=base/'lean_channel_attention_training_pilot_v1/split_preflight_attempt03';cr=base/'lean_token_window_candidate_v1/seq23_attempt01';cb=base/'lean_seq23_no_inset_v1/full_attempt02'
    videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')};pre=read(preflight/'manifest.json')
    summaries={};evidence={};lines=[]
    for seq in ('seq11','seq20','seq23'):
        frames=videos[seq];lean=v/f'lean_mps_baseline_{seq}_full_attempt01';sam=s/f'{seq}_full_mps_attempt02'
        lg=v/f'lean_mps_baseline_{seq}_gate_attempt02/gate.json';sg=s/f'{seq}_gate_verified_v3/gate.json';cpu=base/f'lean_token_window_candidate_v1/{seq}_attempt01'
        _,_,_,prompt=prepare_archived_prompt(seq,frames,preflight,cr,cb)
        start,_,_=check_frozen(seq,frames,lean,sam,lg,sg,prompt)
        cpu_audit=audit_cpu_mps(lean,cpu,sam,lg)
        previous=read(v/f'lean_mps_vs_samurai_{seq}_score_attempt01/results.json')
        if previous['scorer_sha256']!=digest(ROOT/'scripts/score_lean_window_mps_vs_samurai.py') or previous['prompt_frame']!=start or len(previous['rows'])!=len(frames):
            raise ValueError('Earlier score provenance differs')
        meta=read(Path(pre['sequences'][seq]['source'])/seq/'frames.json');rows=[]
        if len(meta)!=len(frames):raise ValueError('Label metadata coverage differs')
        for i,frame in enumerate(frames):
            if digest(frame.mask_path)!=meta[i]['label_sha256']:raise ValueError('Label bytes changed')
            with Image.open(frame.mask_path) as im:
                if im.format!='JPEG':raise ValueError('Unexpected label format')
                gray=np.asarray(im.convert('L'))
            if ((gray>64)&(gray<192)).mean()>0.05:raise ValueError('Label is not a binary JPEG')
            truth=gray>127
            with Image.open(lean/f'{i:06d}.png') as im:lm=np.asarray(im.convert('L'))>0
            if i>=start:
                with Image.open(sam/f'{i:06d}.png') as im:sm=np.asarray(im.convert('L'))>0
            else:sm=np.zeros_like(truth)
            row={'frame':i,'source_frame_index':frame.frame_index,'post_prompt':i>start,'initialization_frame':i==start,
                 'lean':independent_metric(lm,truth),'samurai':independent_metric(sm,truth)}
            if row!=previous['rows'][i]:raise ValueError(f'Independent score differs on {seq} frame {i}')
            rows.append(row)
        result={arm:{'all':independent_summary(rows,arm,False),'post_prompt':independent_summary(rows,arm,True)} for arm in ('lean','samurai')}
        if result!=previous['summaries']:raise ValueError('Independent score denominator or summary differs')
        summaries[seq]=result;evidence[seq]={'frames':len(rows),'prompt_frame':start,
          'pre_prompt_present_misses':sum(x['lean']['present'] for x in rows if x['frame']<start),
          'score_sha256':digest(v/f'lean_mps_vs_samurai_{seq}_score_attempt01/results.json'),
          'lean_receipt_sha256':digest(lean/'receipt.json'),'samurai_receipt_sha256':digest(sam/'receipt.json'),
          'cpu_mps_differing_pixels':cpu_audit['cpu_mps_differing_pixels'],'cpu_mps_differing_frames':cpu_audit['differing_frames'],
          'cpu_mps_min_frame_iou':cpu_audit['minimum_frame_iou'],'max_total_tokens':cpu_audit['max_total_tokens'],
          'memory_writes':cpu_audit['memory_writes'],'evictions':cpu_audit['evictions']}
    lines=['# Same-MPS matched automatic-mask comparison','','Both systems use the same archived automatic mask and initialization frame, without a new detector call. Sequence 23 uses the same fixed inset-excluded viewport. The table excludes the initialization frame and includes every later labeled frame.','',
      '| Sequence | Present frames | Present Dice: project Lean / SAMURAI | Labeled-empty false alarms: project Lean / SAMURAI | Mean predicted pixels on labeled-empty frames: Lean / SAMURAI |',
      '|---|---:|---:|---:|---:|']
    for seq in ('seq11','seq20','seq23'):
        l=summaries[seq]['lean']['post_prompt'];s1=summaries[seq]['samurai']['post_prompt']
        lines.append(f"| {seq} | {l['present_frames']} | {l['mean_present_frame_dice']:.4f} / {s1['mean_present_frame_dice']:.4f} | {l['false_alarm_frames']}/{l['empty_frames']} / {s1['false_alarm_frames']}/{s1['empty_frames']} | {l['mean_empty_predicted_pixels']:.0f} / {s1['mean_empty_predicted_pixels']:.0f} |")
    lines += ['', 'SAMURAI scores better for visible polyps on seq23, with five more labeled-empty false alarms. Both systems remain poor on seq11 and seq20. These three videos were selected and inspected before the comparison; results do not establish generalization.',
      '', 'Project Lean is a modified 16,384-token window with indexed RoPE and a multiscale anchor, not original author Lean. SAMURAI uses author memory-selection thresholds and an explicit float32 MPS compatibility conversion of Kalman IoUs. The Lean MPS runtime uses the tested indexed-position and blocking-transfer adapters; its token counts exclude pointer tokens, positional tensors and transient memory.',
      '', 'Frozen predictions and all source-image hashes were checked before independently decoding all labels and recomputing every frame metric. The current SAMURAI and Lean runs use MPS, the same SAM2.1 checkpoint bytes, prompt mask, crop and input frames; Lean additionally uses its trained bypass, and their architectures and memory policies differ. The historical detector scanned frames until its first detection; no additional detector or redetection was used in these runs. Sequence 23 crop was chosen retrospectively; patient independence, detector overlap, label completeness and native acquisition timing are not established.',
      '', 'CPU/MPS Lean checks compared all frames to the archived CPU project baseline: ' + '; '.join(
          f"{seq} had {evidence[seq]['cpu_mps_differing_pixels']:,} different mask pixels across {evidence[seq]['cpu_mps_differing_frames']}/{evidence[seq]['frames']} frames (minimum IoU {evidence[seq]['cpu_mps_min_frame_iou']:.6f})" for seq in ('seq11','seq20','seq23')) + '. These measurements do not certify GPU equivalence on other videos. Each MPS baseline respected its 16,384-token budget. See `audit.json` for source hashes, exact denominators and evidence paths.']
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    report=out/'report.md';report.write_text('\n'.join(lines)+'\n')
    result={'passed':True,'scope':'Independent frozen-mask rescore; exploratory development videos, not a video refiner experiment.',
      'summaries':summaries,'evidence':evidence,'report_sha256':digest(report),'runner_sha256':digest(Path(__file__))}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(out),'evidence':evidence},indent=2))


if __name__=='__main__':main()
