#!/usr/bin/env python3
"""Run verified fixed early-feature contrast on selected Lean videos."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def main():
 p=argparse.ArgumentParser()
 for x in ('checkpoint','merged','cpu-root','indexed-root','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','no-op-a','no-op-b','active-a','active-b','verification','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--sequence',action='append',required=True);a=p.parse_args();cohort=json.loads((ROOT/'protocols/lean_early_contrast_anchor_v1.json').read_text())['cohort']
 if len(set(a.sequence))!=len(a.sequence) or not set(a.sequence)<=set(cohort):raise ValueError('Invalid cohort')
 v=json.loads(a.verification.read_text());run=ROOT/'scripts/run_lean_early_contrast_anchor.py'
 if (not v['passed'] or v['source_sha256']!=digest(ROOT/'scripts/verify_lean_early_contrast_anchor.py') or v['identity']['runner_sha256']!=digest(run)
     or v['no_op_intervention_sha256']!=digest(a.no_op_a/'contrast_intervention.json') or v['active_intervention_sha256']!=digest(a.active_a/'contrast_intervention.json')):raise ValueError('Invalid engineering verification')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();identity={'runner_sha256':digest(run),'verification_sha256':digest(a.verification),'sequences':a.sequence}
 (out/'setup.json').write_text(json.dumps(identity,indent=2)+'\n');complete=[]
 for name in a.sequence:
  if digest(run)!=identity['runner_sha256'] or digest(a.verification)!=identity['verification_sha256']:raise ValueError('Frozen source changed')
  def ref(root):
   paths=[d/name for d in (root/'part_a_attempt01',root/'part_b_attempt01') if (d/name/'completion.json').exists()]
   if len(paths)!=1:raise ValueError('Reference missing or duplicated')
   return paths[0]
  print(f'Starting {name}: {len(complete)}/{len(a.sequence)}',flush=True)
  cmd=[sys.executable,str(run),'--mode','study','--sequence',name,'--reference',str(ref(a.cpu_root)),'--indexed-reference',str(ref(a.indexed_root))]
  for key in ('checkpoint','merged','cpu_verification','observed_reference','indexed_gate_a','indexed_gate_b','no_op_a','no_op_b','active_a','active_b'):cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
  cmd.extend(['--output',str(out/name)])
  with (out/'logs'/f'{name}.log').open('w') as stream:r=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
  if r.returncode:
   (out/'progress.json').write_text(json.dumps({'completed':complete,'failed':name},indent=2)+'\n');raise RuntimeError(f'{name} failed')
  z=json.loads((out/name/'contrast_intervention.json').read_text());done=json.loads((out/name/'completion.json').read_text())
  if not z['passed'] or not done['passed'] or z['model_records_sha256']!=digest(out/name/'records.json'):raise ValueError('Incomplete video')
  complete.append(name);(out/'progress.json').write_text(json.dumps({'completed':complete},indent=2)+'\n');print(f'Completed {name}',flush=True)
 (out/'completion.json').write_text(json.dumps({'passed':True,'sequences':complete},indent=2)+'\n')
if __name__=='__main__':main()
