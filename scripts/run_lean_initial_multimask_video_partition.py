#!/usr/bin/env python3
"""Run verified native-multimask Lean videos in isolated processes."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def main():
 p=argparse.ArgumentParser()
 for x in ('checkpoint','merged','indexed-root','indexed-verification','prompt-root','prompt-verification','prompt-audit','gate-a','gate-b','verification','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--sequence',action='append',required=True);a=p.parse_args()
 cohort=json.loads((ROOT/'protocols/lean_initial_multimask_video_v1.json').read_text())['cohort']
 if len(set(a.sequence))!=len(a.sequence) or not set(a.sequence)<=set(cohort):raise ValueError('Invalid subset')
 v=json.loads(a.verification.read_text());runner=ROOT/'scripts/run_lean_initial_multimask_video.py'
 if (not v['passed'] or v['source_sha256']!=digest(ROOT/'scripts/verify_lean_initial_multimask_video.py') or v['identity']['runner_sha256']!=digest(runner)
     or v['records_sha256']!=digest(a.gate_a/'records.json')):raise ValueError('Invalid video verification')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);(out/'logs').mkdir();identity={'runner_sha256':digest(runner),'verification_sha256':digest(a.verification),'sequences':a.sequence}
 (out/'setup.json').write_text(json.dumps(identity,indent=2)+'\n');complete=[]
 for name in a.sequence:
  if digest(runner)!=identity['runner_sha256'] or digest(a.verification)!=identity['verification_sha256']:raise ValueError('Frozen source changed')
  print(f'Starting {name}: {len(complete)}/{len(a.sequence)}',flush=True)
  cmd=[sys.executable,str(runner),'--mode','candidate','--sequence',name]
  for key in ('checkpoint','merged','indexed_root','indexed_verification','prompt_root','prompt_verification','prompt_audit','gate_a','gate_b'):cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
  cmd.extend(['--output',str(out/name)])
  with (out/'logs'/f'{name}.log').open('w') as stream:done=subprocess.run(cmd,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
  if done.returncode:
   (out/'progress.json').write_text(json.dumps({'completed':complete,'failed':name},indent=2)+'\n');raise RuntimeError(f'{name} failed; check log')
  receipt=json.loads((out/name/'completion.json').read_text())
  if not receipt['passed'] or digest(out/name/'records.json')!=receipt['records_sha256']:raise ValueError('Incomplete video')
  complete.append(name);(out/'progress.json').write_text(json.dumps({'completed':complete},indent=2)+'\n');print(f'Completed {name}',flush=True)
 (out/'completion.json').write_text(json.dumps({'passed':True,'sequences':complete},indent=2)+'\n')
if __name__=='__main__':main()
