#!/usr/bin/env python3
"""Track four token-window jobs and independently audit once all succeed."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def alive(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
def main():
 p=argparse.ArgumentParser();p.add_argument('--scheduler-pid',type=int,required=True)
 for x in ('root','indexed-root','crop-baseline','merged','historical','verification','output'):p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args();status=a.root/'progress.json';names=('seq8','seq11','seq20','seq23')
 while True:
  completed=[n for n in names if (a.root/f'{n}_attempt01/token_window_observation.json').exists()]
  state={'scheduler_pid':a.scheduler_pid,'completed':completed,'pending':[n for n in names if n not in completed],'audit':'waiting'}
  status.write_text(json.dumps(state,indent=2)+'\n')
  if len(completed)==len(names):break
  if not alive(a.scheduler_pid):
   state['audit']='scheduler_exited_before_all_receipts';status.write_text(json.dumps(state,indent=2)+'\n');raise RuntimeError('Missing candidate receipts')
  time.sleep(20)
 cmd=[sys.executable,str(ROOT/'scripts/audit_lean_token_window.py')]
 for key in ('root','indexed_root','crop_baseline','merged','historical','verification','output'):cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
 with (a.root/'audit_attempt01.log').open('w') as log:
  result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 state['audit']='passed' if result.returncode==0 else 'failed';state['audit_exit_code']=result.returncode
 status.write_text(json.dumps(state,indent=2)+'\n')
 if result.returncode:raise RuntimeError('Independent audit failed; inspect log')
 print('All candidate videos and independent audit completed')
if __name__=='__main__':main()
