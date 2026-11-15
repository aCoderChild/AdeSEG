#!/usr/bin/env python3
"""Run the verified indexed-position Lean candidate in isolated videos."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def main():
 p=argparse.ArgumentParser()
 for name in ('checkpoint','merged','cpu-verification','observed-reference','gate-a','gate-b','verification','cpu-root','output'):
  p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--sequence',action='append',required=True); a=p.parse_args()
 cohort=json.loads((ROOT/'protocols/lean_indexed_spatial_rope_v1.json').read_text())['cohort']
 if len(set(a.sequence))!=len(a.sequence) or not set(a.sequence)<=set(cohort): raise ValueError('Invalid cohort subset')
 v=json.loads(a.verification.read_text()); runner=ROOT/'scripts/run_lean_indexed_spatial_rope.py'
 if (not v['passed'] or v['source_sha256']!=digest(ROOT/'scripts/verify_lean_indexed_rope_gates.py')
     or v['identity']['runner_sha256']!=digest(runner) or v['gate_records_sha256']!=digest(a.gate_a/'records.json')
     or v['observation_sha256']!=digest(a.observed_reference/'position_observation.json')): raise ValueError('Engineering verification invalid')
 out=validate_output(a.output); out.mkdir(parents=True,exist_ok=False); (out/'logs').mkdir()
 identity={'runner_sha256':digest(runner),'verification_sha256':digest(a.verification),'sequences':a.sequence}
 (out/'setup.json').write_text(json.dumps(identity,indent=2)+'\n'); completed=[]
 for name in a.sequence:
  if digest(runner)!=identity['runner_sha256'] or digest(a.verification)!=identity['verification_sha256']: raise ValueError('Frozen runner changed')
  refs=[x/name for x in (a.cpu_root/'part_a_attempt01',a.cpu_root/'part_b_attempt01') if (x/name/'completion.json').exists()]
  if len(refs)!=1: raise ValueError(f'{name}: corrected reference missing or duplicated')
  print(f'Starting {name}: {len(completed)}/{len(a.sequence)}',flush=True)
  cmd=[sys.executable,str(runner),'--mode','candidate','--sequence',name,'--reference',str(refs[0])]
  for key in ('checkpoint','merged','cpu_verification','observed_reference','gate_a','gate_b'):
   cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
  cmd.extend(['--output',str(out/name)])
  with (out/'logs'/f'{name}.log').open('w') as stream: result=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
  if result.returncode:
   (out/'progress.json').write_text(json.dumps({'completed':completed,'failed':name},indent=2)+'\n')
   raise RuntimeError(f'{name} failed; inspect log')
  done=json.loads((out/name/'completion.json').read_text())
  if not done['passed'] or digest(out/name/'records.json')!=done['records_sha256']: raise ValueError('Run incomplete')
  completed.append(name); (out/'progress.json').write_text(json.dumps({'completed':completed},indent=2)+'\n')
  print(f'Completed {name}',flush=True)
 (out/'completion.json').write_text(json.dumps({'passed':True,'sequences':completed},indent=2)+'\n')
if __name__=='__main__':main()
