#!/usr/bin/env python3
"""Audit both fixed-budget conditioning selection runs when complete."""
import argparse,json,subprocess,sys,time
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1]
def main():
 p=argparse.ArgumentParser()
 for x in ('usage','uniform','merged','historical','indexed-root','verification','output'):p.add_argument('--'+x,type=Path,required=True)
 a=p.parse_args()
 while True:
  for part in (a.usage,a.uniform):
   f=part/'progress.json'
   if f.exists():
    try:r=json.loads(f.read_text())
    except json.JSONDecodeError:continue
    if r.get('failed'):raise RuntimeError(f'{part}: {r["failed"]} failed')
  if all((x/'completion.json').exists() for x in (a.usage,a.uniform)):break
  time.sleep(20)
 cmd=[sys.executable,str(ROOT/'scripts/audit_lean_condition_memory_pruning.py')]
 for key in ('usage','uniform','merged','historical','indexed_root','verification','output'):cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
 subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__':main()
