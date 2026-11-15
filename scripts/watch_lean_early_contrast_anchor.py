#!/usr/bin/env python3
"""Audit fixed early-contrast selected videos after both partitions finish."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser();p.add_argument('--part',type=Path,action='append',required=True)
 for x in ('merged','historical','cpu-root','verification','gate','output'):p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args()
 while True:
  for part in a.part:
   f=part/'progress.json'
   if f.exists():
    try:r=json.loads(f.read_text())
    except json.JSONDecodeError:continue
    if r.get('failed'):raise RuntimeError(f'{part}: {r["failed"]} failed')
  if all((x/'completion.json').exists() for x in a.part):break
  time.sleep(20)
 cmd=[sys.executable,str(ROOT/'scripts/audit_lean_early_contrast_anchor.py')]
 for x in a.part:cmd.extend(['--part',str(x)])
 for key in ('merged','historical','cpu_root','verification','gate','output'):cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
 subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__':main()
