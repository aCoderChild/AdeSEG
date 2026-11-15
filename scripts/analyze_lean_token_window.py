#!/usr/bin/env python3
"""Precisely summarize an independently audited token-window comparison."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('audit','historical','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();done=read(a.audit/'completion.json');data=read(a.audit/'results.json')
 if not done['passed'] or digest(a.audit/'results.json')!=done['results_sha256'] or done['source_sha256']!=digest(ROOT/'scripts/audit_lean_token_window.py'):raise ValueError('Independent audit invalid')
 output={};lines=['# Token-window comparison: precise framewise analysis','','Four retrospectively inspected sequences; the native and Lean reference scores use ground-truth initialization, unlike the automatic systems. Sequence 23 uses the no-inset crop in both automatic arms. The target test uses the exact recorded historical reference score; it does not make the initial prompts comparable.','',
  '| Sequence | Better historical GT-prompt Dice | Automatic baseline Dice | Token-window Dice | Change | Window meets historical target? | Present-frame wins / losses / ties |',
  '| --- | ---: | ---: | ---: | ---: | --- | ---: |']
 for name,x in data['sequences'].items():
  targets=[json.loads(line) for line in (a.historical/'full'/name/'frames.jsonl').read_text().splitlines()];ids={z['frame'] for z in targets if z['present']}
  rows=[z for z in x['per_frame'] if z['frame'] in ids]
  if len(rows)!=len(ids) or any(not z['present'] for z in rows):raise ValueError('Historical frames differ')
  delta=np.array([z['scores']['candidate']['dice']-z['scores']['baseline']['dice'] for z in rows])
  target=max(x['native_gt_dice'],x['lean_gt_dice']);old=x['baseline']['matched_present_dice'];new=x['candidate']['matched_present_dice']
  if not np.isclose(delta.mean(),new-old,atol=1e-12,rtol=0):raise ValueError('Paired difference differs')
  wins=int((delta>1e-12).sum());loss=int((delta< -1e-12).sum());ties=int((np.abs(delta)<=1e-12).sum())
  output[name]={'better_gt_prompt_dice':target,'baseline_dice':old,'window_dice':new,'difference':new-old,'meets_gt_prompt_reference':new>=target,
   'matched_present_frames':len(rows),'wins':wins,'losses':loss,'ties':ties,'zero_dice_baseline':sum(z['scores']['baseline']['dice']==0 for z in rows),
   'zero_dice_window':sum(z['scores']['candidate']['dice']==0 for z in rows),
   'baseline_empty_false_alarms':x['baseline']['all_empty_false_alarms'],'window_empty_false_alarms':x['candidate']['all_empty_false_alarms'],'empty_frames':x['baseline']['all_empty_count'],
   'max_tokens':x['telemetry']['max_represented_spatial_plus_insurance'],'evictions':x['telemetry']['evictions']}
  lines.append(f"| {name} | {target:.6f} | {old:.6f} | {new:.6f} | {new-old:+.6f} | {'yes' if new>=target else 'no'} | {wins} / {loss} / {ties} |")
 lines+=['','Framewise wins are descriptive; neighboring video frames are correlated, and the selected sequences have been inspected repeatedly. No uncertainty or causal attribution can be inferred from these counts. The window also changes fused-history handling and temporal-age assignment, so this experiment does not isolate the token budget alone. Empty-frame false alarms, all-frame and per-frame values are in the independent audit.','', 'The budget counts represented spatial feature and insurance entries only; it excludes pointers, positional tensors, masks, other state and temporary allocations. No full-video result proves generalization or total-RAM reduction.']
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'sequences':output,'audit_results_sha256':done['results_sha256'],'source_sha256':digest(Path(__file__))};(out/'results.json').write_text(json.dumps(result,indent=2)+'\n');(out/'report.md').write_text('\n'.join(lines)+'\n')
 (out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json')},indent=2)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
