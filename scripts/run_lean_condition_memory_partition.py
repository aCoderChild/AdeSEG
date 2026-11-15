#!/usr/bin/env python3
"""Run verified fixed-budget conditioning selections on the four videos."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def main():
 p=argparse.ArgumentParser()
 for x in ('checkpoint','merged','cpu-root','indexed-root','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','full-a','full-b','usage-a','usage-b','uniform-a','uniform-b','verification','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--sequence',action='append',required=True);p.add_argument('--strategy',choices=('usage','uniform'),required=True);a=p.parse_args();cohort=json.loads((ROOT/'protocols/lean_condition_memory_pruning_v1.json').read_text())['cohort']
 if len(set(a.sequence))!=len(a.sequence) or not set(a.sequence)<=set(cohort):raise ValueError('Invalid cohort')
 v=json.loads(a.verification.read_text());run=ROOT/'scripts/run_lean_condition_memory_pruning.py'
 if (not v['passed'] or v['source_sha256']!=digest(ROOT/'scripts/verify_lean_condition_memory_pruning.py') or v['identity']['runner_sha256']!=digest(run)
     or any(v[strategy+'_intervention_sha256']!=digest(path/'condition_intervention.json') for strategy,path in [('full',a.full_a),('usage',a.usage_a),('uniform',a.uniform_a)])):
  raise ValueError('Invalid engineering verification')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();identity={'runner_sha256':digest(run),'verification_sha256':digest(a.verification),'strategy':a.strategy,'sequences':a.sequence}
 (out/'setup.json').write_text(json.dumps(identity,indent=2)+'\n');complete=[]
 for name in a.sequence:
  if digest(run)!=identity['runner_sha256'] or digest(a.verification)!=identity['verification_sha256']:raise ValueError('Frozen source changed')
  def ref(root):
   paths=[d/name for d in (root/'part_a_attempt01',root/'part_b_attempt01') if (d/name/'completion.json').exists()]
   if len(paths)!=1:raise ValueError('Reference missing or duplicated')
   return paths[0]
  print(f'Starting {name}: {len(complete)}/{len(a.sequence)}',flush=True)
  cmd=[sys.executable,str(run),'--mode','study','--strategy',a.strategy,'--sequence',name,'--reference',str(ref(a.cpu_root)),'--indexed-reference',str(ref(a.indexed_root))]
  for key in ('checkpoint','merged','cpu_verification','observed_reference','indexed_gate_a','indexed_gate_b','full_a','full_b','usage_a','usage_b','uniform_a','uniform_b'):cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
  cmd.extend(['--output',str(out/name)])
  with (out/'logs'/f'{name}.log').open('w') as stream:r=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
  if r.returncode:
   (out/'progress.json').write_text(json.dumps({'completed':complete,'failed':name},indent=2)+'\n');raise RuntimeError(f'{name} failed')
  z=json.loads((out/name/'condition_intervention.json').read_text());done=json.loads((out/name/'completion.json').read_text())
  if not z['passed'] or not done['passed'] or z['model_records_sha256']!=digest(out/name/'records.json'):raise ValueError('Incomplete video')
  complete.append(name);(out/'progress.json').write_text(json.dumps({'completed':complete},indent=2)+'\n');print(f'Completed {name}',flush=True)
 (out/'completion.json').write_text(json.dumps({'passed':True,'sequences':complete},indent=2)+'\n')
if __name__=='__main__':main()
