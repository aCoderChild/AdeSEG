#!/usr/bin/env python3
"""Dispatch existing token-window machinery to an isolated raw-two read control."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
import scripts.run_lean_token_window as token_runner
FORK=ROOT/'external/Lean-SAM2-raw-two-control';PROTOCOL=ROOT/'protocols/lean_raw_two_window_control_v1.json'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser(add_help=False);p.add_argument('--entry',required=True,choices=('token','verify','crop'))
 opts,args=p.parse_known_args();token_runner.FORK=FORK;token_runner.PROTOCOL=PROTOCOL;sys.argv=[sys.argv[0]]+args
 if opts.entry=='token':token_runner.main()
 elif opts.entry=='verify':
  import scripts.verify_lean_token_window_gates as verifier
  verifier.FORK=FORK;verifier.PROTOCOL=PROTOCOL;verifier.main()
 else:
  import scripts.run_seq23_token_window as cropped
  cropped.FORK=FORK;cropped.PROTOCOL=PROTOCOL;cropped.main()
 try:out=Path(args[args.index('--output')+1])
 except (ValueError,IndexError):raise ValueError('Output path missing')
 receipt=out/('verification.json' if opts.entry=='verify' else 'token_window_observation.json');data=read(receipt)
 if not data['passed']:raise ValueError('Delegate failed')
 if data.get('variant')=='window' or opts.entry=='crop':
  events=read(out/'token_window_events.json');reads=[x for x in events if x['stage']=='after_read']
  if not reads or any(x.get('history_read_mode')!='raw_two_latest' for x in reads):raise ValueError('Two-entry raw read is not active')
 result={'passed':True,'entry':opts.entry,'delegate_receipt_sha256':digest(receipt),'dispatcher_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL)}
 (out/'raw_two_control_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
