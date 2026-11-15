#!/usr/bin/env python3
"""Dispatch pinned token-window machinery to isolated fused-read control."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
import scripts.run_lean_token_window as token_runner

FORK=ROOT/'external/Lean-SAM2-fused-window-control'
PROTOCOL=ROOT/'protocols/lean_fused_token_window_control_v1.json'
def read(path):return json.loads(path.read_text())
def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--entry',required=True,choices=('token','verify','crop'))
 opts,remaining=p.parse_known_args();token_runner.FORK=FORK;token_runner.PROTOCOL=PROTOCOL
 sys.argv=[sys.argv[0]]+remaining
 if opts.entry=='token':
  token_runner.main()
 elif opts.entry=='verify':
  import scripts.verify_lean_token_window_gates as verifier
  verifier.FORK=FORK;verifier.PROTOCOL=PROTOCOL;verifier.main()
 else:
  import scripts.run_seq23_token_window as cropped
  cropped.FORK=FORK;cropped.PROTOCOL=PROTOCOL;cropped.main()
 # The delegate has its own receipt; bind this dispatcher's identity too.
 try:out=Path(remaining[remaining.index('--output')+1])
 except (ValueError,IndexError):raise ValueError('Output path missing')
 if opts.entry=='verify':
  receipt=out/'verification.json';data=read(receipt)
  if not data['passed']:raise ValueError('Gate verification failed')
 else:
  receipt=out/'token_window_observation.json';data=read(receipt)
  if not data['passed']:raise ValueError('Control failed')
  if data.get('variant')=='window' or (opts.entry=='crop'):
   events=read(out/'token_window_events.json');reads=[x for x in events if x['stage']=='after_read']
   if not reads or any(x.get('history_read_mode')!='author_fused_old_tail' for x in reads):raise ValueError('Fused-read path not active')
   # The cropped three-frame gate has only two reads; fusion requires two
   # prior nonconditioning entries and is exercised by seq3 and full runs.
   if not (opts.entry=='crop' and data.get('mode')=='gate') and max(x['fusion_update_count'] for x in reads)<1:
    raise ValueError('Fused history never updated')
 result={'passed':True,'entry':opts.entry,'delegate_receipt_sha256':digest(receipt),'dispatcher_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL)}
 (out/'fused_control_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
