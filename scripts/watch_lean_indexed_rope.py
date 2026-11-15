#!/usr/bin/env python3
"""Run the independent indexed RoPE audit when both partitions finish."""
import argparse
import json
from pathlib import Path
import subprocess
import sys
import time
ROOT=Path(__file__).resolve().parents[1]

def main():
 p=argparse.ArgumentParser(); p.add_argument('--part',type=Path,action='append',required=True)
 for name in ('merged','historical','cpu-root','verification','gate','output'): p.add_argument('--'+name,type=Path,required=True)
 a=p.parse_args()
 while True:
  for part in a.part:
   f=part/'progress.json'
   if f.exists():
    try: progress=json.loads(f.read_text())
    except json.JSONDecodeError: continue
    if progress.get('failed'): raise RuntimeError(f'{part}: {progress["failed"]} failed')
  if all((x/'completion.json').exists() for x in a.part): break
  time.sleep(20)
 cmd=[sys.executable,str(ROOT/'scripts/audit_lean_indexed_rope.py')]
 for part in a.part: cmd.extend(['--part',str(part)])
 for key in ('merged','historical','cpu_root','verification','gate','output'): cmd.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
 subprocess.run(cmd,cwd=ROOT,check=True)
if __name__=='__main__':main()
