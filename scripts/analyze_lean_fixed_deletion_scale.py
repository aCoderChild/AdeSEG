#!/usr/bin/env python3
"""Summarize already audited one-read Lean deletion effects; no new inference."""
import argparse,hashlib,json,sys
from pathlib import Path
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import validate_output
p=argparse.ArgumentParser();p.add_argument('--source',type=Path,required=True);p.add_argument('--output',type=Path,required=True);a=p.parse_args();h=lambda p:hashlib.sha256(p.read_bytes()).hexdigest();rows=[];sources={}
for seq in ('seq8','seq11','seq20','seq23'):
 folder=a.source/f'{seq}_audit_attempt01';verify=a.source/f'{seq}_audit_verification_attempt01';data=json.loads((folder/'results.json').read_text());done=json.loads((folder/'completion.json').read_text());v=json.loads((verify/'verification.json').read_text())
 if not done['passed'] or not v['passed'] or done['results_sha256']!=h(folder/'results.json') or v['audit_sha256']!=done['results_sha256'] or data['sequence']!=seq or len(data['frames'])!=4:raise ValueError('Audit or verification differs')
 sources[seq]=done['results_sha256'];rows.extend((seq,r) for r in data['frames'])
values=[];empty=[]
for seq,r in rows:
 scores=r['scores'];base=scores['noop']
 for name,z in scores.items():
  if name=='noop':continue
  if r['present']:values.append(abs(z['dice']-base['dice']))
  else:empty.append(z['false_alarm']!=base['false_alarm'])
a0=np.asarray(values,dtype=np.float64)
if not np.isfinite(a0).all() or not len(a0):raise ValueError('Invalid score effects')
s={'present_reads':sum(r['present'] for _,r in rows),'empty_reads':sum(not r['present'] for _,r in rows),'present_arm_comparisons':len(values),'median_absolute_dice_change':float(np.median(a0)),'p95_absolute_dice_change':float(np.quantile(a0,.95)),'max_absolute_dice_change':float(a0.max()),'empty_arm_comparisons':len(empty),'empty_alarm_status_changes':sum(empty)}
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);data={'summary':s,'source_sha256':sources,'runner_sha256':h(Path(__file__)),'scope':'Selected previously inspected automatic Lean reads; each intervention removes 1024 keys. Different from native tile deletion.'};(out/'results.json').write_text(json.dumps(data,indent=2)+'\n')
report=f'''# Automatic Lean fixed-read deletion effect scale\n\nOn {s['present_reads']} labeled-present and {s['empty_reads']} labeled-empty selected reads, {s['present_arm_comparisons']} method/read deletion comparisons had median absolute Dice change {s['median_absolute_dice_change']:.8f}, 95th percentile {s['p95_absolute_dice_change']:.6f}, and maximum {s['max_absolute_dice_change']:.6f}. Across {s['empty_arm_comparisons']} deletion comparisons on labeled-empty reads, false-alarm status changed {s['empty_alarm_status_changes']} times. Scores share frames and token sets and are not independent observations.\n\nEach arm removes 1,024 selected historical spatial keys from the same fixed read; no-op retains those keys, so it is descriptive rather than an equal-key control. This differs from archived native SAM2 tile removal in architecture, prompts, region size and states. The sampled frames were previously inspected and patient independence is unknown. A few selected one-step effects do not establish utility-training reliability, performance on empty videos, or full-video efficacy.\n''';(out/'report.md').write_text(report);(out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':h(out/'results.json'),'report_sha256':h(out/'report.md')},indent=2)+'\n');print(report)
