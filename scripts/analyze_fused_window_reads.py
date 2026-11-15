#!/usr/bin/env python3
"""Summarize attention-key counts for the independently audited control."""
import argparse,collections,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('audit','baseline','raw','fused','output'):p.add_argument('--'+k,type=Path,required=True)
 p.add_argument('--sequence',choices=('seq11','seq23'),required=True);a=p.parse_args()
 done=read(a.audit/'completion.json');data=read(a.audit/'results.json')
 if not done['passed'] or digest(a.audit/'results.json')!=done['results_sha256'] or done['source_sha256']!=digest(ROOT/'scripts/audit_lean_fused_window_control.py') or a.sequence not in data['sequences']:raise ValueError('Control audit invalid')
 results={}
 for name,path in [('baseline',a.baseline),('raw_window',a.raw),('fused_window',a.fused)]:
  if a.sequence=='seq23':
   d=read(path/'completion.json');expected=d['positions_sha256']
  else:
   d=read(path/'completion.json');r=read(path/'records.json')
   if digest(path/'records.json')!=d['records_sha256']:raise ValueError('Record changed')
   expected=r['positions_sha256']
  if not d['passed'] or digest(path/'positions.json')!=expected:raise ValueError('Position trace changed')
  rows=read(path/'positions.json');counts=collections.Counter(x['spatial_keys'] for x in rows)
  results[name]={'reads':len(rows),'spatial_key_distribution':dict(sorted(counts.items())),
                 'mean_spatial_keys':sum(k*v for k,v in counts.items())/len(rows),'max_spatial_keys':max(counts),'pointer_keys_max':max(x['pointer_keys'] for x in rows),'positions_sha256':expected}
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 result={'sequence':a.sequence,'scope':'Actual attention key counts include conditioning, nonconditioning and any insurance spatial keys; pointer keys are separate. Storage and attention key counts differ.',
         'conditions':results,'control_audit_sha256':done['results_sha256'],'source_sha256':digest(Path(__file__))}
 (out/'results.json').write_text(json.dumps(result,indent=2)+'\n')
 lines=['# Attention read sizes: fused and raw token windows','',result['scope'],'','| Arm | Mean spatial keys per read | Maximum spatial keys | Maximum pointer keys |','| --- | ---: | ---: | ---: |']
 for name,x in results.items():lines.append(f"| {name} | {x['mean_spatial_keys']:.1f} | {x['max_spatial_keys']} | {x['pointer_keys_max']} |")
 lines+=['','This is key-count accounting, not a controlled measurement of runtime, peak memory or causal contribution. Fused and raw methods can see different histories because their predictions change future memory.']
 (out/'report.md').write_text('\n'.join(lines)+'\n');print('\n'.join(lines))
if __name__=='__main__':main()
