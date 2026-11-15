#!/usr/bin/env python3
"""Run fixed-crop diagnostic videos in isolated processes."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def main():
    p = argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','merged','historical','crop-diagnostic','prompt-gate-a','prompt-gate-b','gate-a','gate-b','output'):
        p.add_argument('--'+name, type=Path, required=True)
    p.add_argument('--sequence', action='append', required=True)
    a=p.parse_args()
    if len(set(a.sequence)) != len(a.sequence): raise ValueError('Duplicate sequence')
    out=validate_output(a.output); out.mkdir(parents=True, exist_ok=False)
    logs=out/'logs'; logs.mkdir()
    runner=ROOT/'scripts/run_lean_viewport_crop_diagnostic.py'
    identity={'runner_sha256':digest(runner),'sequences':a.sequence}
    (out/'setup.json').write_text(json.dumps(identity,indent=2)+'\n')
    complete=[]
    for name in a.sequence:
        if digest(runner) != identity['runner_sha256']: raise ValueError('Runner changed')
        print(f'Starting {name}: {len(complete)}/{len(a.sequence)} complete',flush=True)
        args=[sys.executable,str(runner),'--mode','candidate','--sequence',name,
              '--checkpoint',str(a.checkpoint),'--merged',str(a.merged),'--historical',str(a.historical),
              '--crop-diagnostic',str(a.crop_diagnostic),'--prompt-gate-a',str(a.prompt_gate_a),
              '--prompt-gate-b',str(a.prompt_gate_b),'--gate-a',str(a.gate_a),'--gate-b',str(a.gate_b),'--output',str(out/name)]
        with (logs/f'{name}.log').open('w') as stream:
            result=subprocess.run(args,cwd=ROOT,stdout=stream,stderr=subprocess.STDOUT)
        if result.returncode:
            (out/'progress.json').write_text(json.dumps({'completed':complete,'failed':name},indent=2)+'\n')
            raise RuntimeError(f'{name} failed; inspect log')
        receipt=json.loads((out/name/'completion.json').read_text())
        if not receipt['passed'] or digest(out/name/'records.json') != receipt['records_sha256']:
            raise ValueError(f'{name} incomplete')
        complete.append(name)
        (out/'progress.json').write_text(json.dumps({'completed':complete},indent=2)+'\n')
        print(f'Completed {name}',flush=True)
    (out/'completion.json').write_text(json.dumps({'passed':True,'sequences':complete},indent=2)+'\n')


if __name__=='__main__': main()
