#!/usr/bin/env python3
"""Independently recompute usage percentiles and selected-mask outcomes."""
import argparse,json,sys,math
from pathlib import Path
import torch
from scipy.stats import rankdata
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for n in ('usage','captures','deletions','report','output'):p.add_argument('--'+n,type=Path,required=True)
a=p.parse_args();read=lambda f:json.loads(f.read_text());done=read(a.report/'completion.json');report=read(a.report/'results.json')
if not done['passed'] or done['results_sha256']!=digest(a.report/'results.json') or report['runner_sha256']!=digest(ROOT/'scripts/report_lean_historical_usage_groups.py'):raise ValueError('Report provenance changed')
checked=[]
for row in report['rows']:
 seq=row['sequence'];cap=a.usage/('gate_attempt01' if seq=='seq3' else f'{seq}_attempt01');source=a.captures/('source_gate_attempt02' if seq=='seq3' else f'{seq}_source_capture_attempt01');rep=a.deletions/('gate_attempt01' if seq=='seq3' else f'{seq}_attempt01');receipt=read(cap/'historical_usage_receipt.json');records={e['source_frame']:e for e in receipt['events']};score=read(a.usage/f'{seq}_scores_attempt03/scores.json');events={e['frame']:e for e in read(rep/'completion.json')['events'] if 'groups' in e};audit={z['frame']:z for z in read(a.deletions/f'{seq}_audit_attempt01/results.json')['frames']};delta=[];fa=0
 for e in score['events']:
  frame=e['frame'];snapshot=torch.load(source/'source_snapshots'/f'{frame:06d}.pt',weights_only=True);values={}
  for s in snapshot['sources']:
   i=s['frame']
   if i>=frame-1:continue
   rec=records[i];path=cap/'historical_usage'/rec['file']
   if digest(path)!=rec['sha256']:raise ValueError('Usage source changed')
   u=torch.load(path,weights_only=True);ranks=(rankdata(u['usage'].numpy(),method='average')-1)/4095
   for n,token in enumerate(s['token_ids'].tolist()):values[s['start']+n]=float(ranks[token])
  expected=[sum(values[j] for j in group)/len(group) for group in events[frame]['groups']]
  if any(abs(v-w)>1e-12 for v,w in zip(expected,e['scores']['historical_usage'])):raise ValueError('Usage rank differs')
  choice=min(range(5),key=lambda i:(expected[i],i));f=audit[frame];m=[f['scores'][f'group{i}'] for i in range(5)]
  if f['present']:delta.append(m[choice]['dice']-sum(z['dice'] for z in m)/5)
  else:fa+=int(m[choice]['false_alarm'])
 if not delta or not math.isclose(sum(delta)/len(delta),row['dice_minus_source_average']['historical_usage'],abs_tol=1e-12) or fa!=row['empty_false_alarms']['historical_usage']:raise ValueError('Outcome differs')
 checked.append({'sequence':seq,'present_reads':len(delta),'empty_false_alarms':fa})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);result={'passed':True,'checked':checked,'report_sha256':digest(a.report/'results.json'),'verifier_sha256':digest(Path(__file__)),'method':'Independent scipy tie-aware ranks and direct selected-mask metric recomputation.'};(out/'verification.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result))
