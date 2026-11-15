#!/usr/bin/env python3
"""Wait for the selected fused-window runs, then run the independent audit."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def alive(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
def main():
 p=argparse.ArgumentParser();p.add_argument('--scheduler-pid',type=int,required=True)
 for k in ('root','raw-root','raw-audit','indexed-root','crop-baseline','merged','historical','verification','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();status=a.root/'progress.json';names=('seq11','seq23')
 while True:
  complete=[n for n in names if (a.root/f'{n}_attempt01/fused_control_receipt.json').exists()]
  state={'scheduler_pid':a.scheduler_pid,'completed':complete,'pending':[x for x in names if x not in complete],'audit':'waiting'};status.write_text(json.dumps(state,indent=2)+'\n')
  if len(complete)==len(names):break
  if not alive(a.scheduler_pid):
   state['audit']='scheduler_exited_before_receipts';status.write_text(json.dumps(state,indent=2)+'\n');raise RuntimeError('Control run failed')
  time.sleep(20)
 cmd=[sys.executable,str(ROOT/'scripts/audit_lean_fused_window_control.py')]
 for k in ('root','raw_root','raw_audit','indexed_root','crop_baseline','merged','historical','verification','output'):cmd.extend(['--'+k.replace('_','-'),str(getattr(a,k))])
 with (a.root/'audit_attempt01.log').open('w') as log:result=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 state['audit']='passed' if result.returncode==0 else 'failed';state['audit_exit_code']=result.returncode;status.write_text(json.dumps(state,indent=2)+'\n')
 if result.returncode:raise RuntimeError('Control audit failed; inspect log')
 print('Fused-window control audit completed')
if __name__=='__main__':main()
