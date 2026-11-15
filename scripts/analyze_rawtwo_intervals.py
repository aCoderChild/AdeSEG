#!/usr/bin/env python3
"""Summarize post-hoc seq23 intervals from the independent raw-two audit."""
import argparse,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--audit',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
 done=read(a.audit/'completion.json');data=read(a.audit/'results.json')
 if not done['passed'] or digest(a.audit/'results.json')!=done['results_sha256'] or done['source_sha256']!=digest(ROOT/'scripts/audit_seq23_raw_two_control.py'):raise ValueError('Raw-two audit invalid')
 rows=data['frames'];intervals={'frames_0_46':(0,46),'frames_47_55':(47,55)};result={}
 for name,(lo,hi) in intervals.items():
  selected=[x for x in rows if lo<=x['frame']<=hi and x['present']]
  if not selected:raise ValueError('No labeled-present frames')
  result[name]={'present_frames':len(selected),'dice':{arm:float(np.mean([x['scores'][arm]['dice'] for x in selected])) for arm in ('baseline','fused_window','raw_two','raw_window')}}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 payload={'scope':'Intervals selected after observing the prior results; descriptive only.','intervals':result,'audit_sha256':done['results_sha256'],'source_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(payload,indent=2)+'\n')
 lines=['# Sequence 23: four-arm interval comparison','',payload['scope'],'','| Interval | Labeled-present frames | Baseline | Fused window | Two-entry raw | Full raw window |','| --- | ---: | ---: | ---: | ---: | ---: |']
 for name,x in result.items():
  d=x['dice'];lines.append(f"| {name} | {x['present_frames']} | {d['baseline']:.3f} | {d['fused_window']:.3f} | {d['raw_two']:.3f} | {d['raw_window']:.3f} |")
 lines+=['','The frames are temporally correlated and the intervals were chosen after inspection. These are full-video trajectories, so later masks and memory contents can differ. The table cannot establish a same-state causal effect or an independently validated improvement.']
 (out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
