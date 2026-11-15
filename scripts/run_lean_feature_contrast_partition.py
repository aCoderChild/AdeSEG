#!/usr/bin/env python3
"""Collect verified read-only FPN scores during indexed Lean video replays."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def main():
 p=argparse.ArgumentParser()
 for x in ('checkpoint','merged','cpu-root','indexed-root','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','verification','gate','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--sequence',action='append',required=True);a=p.parse_args();cohort=json.loads((ROOT/'protocols/lean_early_feature_contrast_diagnostic_v1.json').read_text())['cohort']
 if len(set(a.sequence))!=len(a.sequence) or not set(a.sequence)<=set(cohort):raise ValueError('Invalid cohort')
 v=json.loads(a.verification.read_text());runner=ROOT/'scripts/observe_lean_feature_contrast.py'
 if (not v['passed'] or v['source_sha256']!=digest(ROOT/'scripts/verify_lean_feature_contrast.py') or v['observer_sha256']!=digest(runner)
     or v['observation_sha256']!=digest(a.gate/'contrast_observation.json') or v['protocol_sha256']!=digest(ROOT/'protocols/lean_early_feature_contrast_diagnostic_v1.json')):raise ValueError('Invalid engineering verification')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();identity={'runner_sha256':digest(runner),'verification_sha256':digest(a.verification),'sequences':a.sequence}
 (out/'setup.json').write_text(json.dumps(identity,indent=2)+'\n');complete=[]
 for name in a.sequence:
  if digest(runner)!=identity['runner_sha256'] or digest(a.verification)!=identity['verification_sha256']:raise ValueError('Frozen source changed')
  def ref(root):
   matches=[d/name for d in (root/'part_a_attempt01',root/'part_b_attempt01') if (d/name/'completion.json').exists()]
   if len(matches)!=1:raise ValueError('Reference missing or duplicated')
   return matches[0]
  print(f'Starting {name}: {len(complete)}/{len(a.sequence)}',flush=True)
  cmd=[sys.executable,str(runner),'--sequence',name,'--reference',str(ref(a.cpu_root)),'--indexed-reference',str(ref(a.indexed_root))]
  for key in ('checkpoint','merged','cpu_verification','observed_reference','indexed_gate_a','indexed_gate_b'):cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
  cmd.extend(['--output',str(out/name)])
  with (out/'logs'/f'{name}.log').open('w') as stream:r=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
  if r.returncode:
   (out/'progress.json').write_text(json.dumps({'completed':complete,'failed':name},indent=2)+'\n');raise RuntimeError(f'{name} failed')
  obs=json.loads((out/name/'contrast_observation.json').read_text());done=json.loads((out/name/'completion.json').read_text())
  if not obs['passed'] or not done['passed'] or digest(out/name/'records.json')!=done['records_sha256']:raise ValueError('Incomplete observation')
  complete.append(name);(out/'progress.json').write_text(json.dumps({'completed':complete},indent=2)+'\n');print(f'Completed {name}',flush=True)
 (out/'completion.json').write_text(json.dumps({'passed':True,'sequences':complete},indent=2)+'\n')
if __name__=='__main__':main()
