#!/usr/bin/env python3
"""Wait for seq11 raw-two control and independently rescore it."""
import argparse,json,os,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def alive(pid):
 try:os.kill(pid,0);return True
 except ProcessLookupError:return False
def main():
 p=argparse.ArgumentParser();p.add_argument('--pid',type=int,required=True)
 for k in ('baseline','raw-window','fused-window','raw-two','fused-audit','merged','historical','verification','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();status=a.raw_two.parent/'seq11_progress.json'
 while not (a.raw_two/'raw_two_control_receipt.json').exists():
  state={'pid':a.pid,'run':'running' if alive(a.pid) else 'exited_without_receipt','audit':'waiting'};status.write_text(json.dumps(state,indent=2)+'\n')
  if state['run']!='running':raise RuntimeError('Control exited without receipt')
  time.sleep(20)
 cmd=[sys.executable,str(ROOT/'scripts/audit_seq11_raw_two_control.py')]
 for k in ('baseline','raw_window','fused_window','raw_two','fused_audit','merged','historical','verification','output'):cmd.extend(['--'+k.replace('_','-'),str(getattr(a,k))])
 with (a.raw_two.parent/'seq11_audit_attempt01.log').open('w') as log:r=subprocess.run(cmd,cwd=ROOT,stdout=log,stderr=subprocess.STDOUT)
 state={'pid':a.pid,'run':'complete','audit':'passed' if r.returncode==0 else 'failed','audit_exit_code':r.returncode};status.write_text(json.dumps(state,indent=2)+'\n')
 if r.returncode:raise RuntimeError('Audit failed; inspect log')
 print('Sequence 11 control audit completed')
if __name__=='__main__':main()
