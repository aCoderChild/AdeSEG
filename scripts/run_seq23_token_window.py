#!/usr/bin/env python3
"""Run token-window Lean on seq23's existing no-inset preprocessing."""
import argparse,json,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_token_window_storage import TokenWindowRuntime
from scripts.memory_token_io import digest
from scripts.run_lean_token_window import FORK,PATCHED,PROTOCOL
import scripts.run_seq23_viewport as crop_runner

def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 p.add_argument('--mode',choices=('gate','candidate'),required=True)
 for key in ('checkpoint','detector','crop-gate-a','crop-gate-b','baseline','verification','output'):p.add_argument('--'+key,type=Path,required=True)
 p.add_argument('--window-crop-gate-a',type=Path);p.add_argument('--window-crop-gate-b',type=Path)
 a=p.parse_args();v=read(a.verification/'verification.json');ident=v['identity']
 if not v['passed'] or v['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or ident['runner_sha256']!=digest(ROOT/'scripts/run_lean_token_window.py') or ident['protocol_sha256']!=digest(PROTOCOL) or ident['checkpoint_sha256']!=digest(a.checkpoint) or ident['runtime_sha256']!=digest(ROOT/'modeling/lean_token_window_storage.py') or ident['policy_sha256']!=digest(ROOT/'modeling/token_memory_window.py') or any(ident['patch_sha256'][k]!=digest(FORK/k) for k in PATCHED):raise ValueError('Window gate verification invalid')
 base=read(a.baseline/'completion.json')
 if not base['passed'] or digest(a.baseline/'frames.json')!=base['frames_sha256'] or digest(a.baseline/'prompt.json')!=base['prompt_sha256'] or digest(a.baseline/'setup.json')!=base['setup_sha256']:raise ValueError('Cropped baseline invalid')
 if a.mode=='candidate':
  gate_receipts=[]
  for path in (a.window_crop_gate_a,a.window_crop_gate_b):
   if path is None:raise ValueError('Two cropped engineering gates required')
   r=read(path/'token_window_observation.json')
   if not r['passed'] or r['mode']!='gate' or r['source_sha256']!=digest(Path(__file__)) or r['verification_sha256']!=digest(a.verification/'verification.json') or r['baseline_completion_sha256']!=digest(a.baseline/'completion.json') or digest(path/'gate.json')!=r['records_sha256'] or digest(path/'token_window_events.json')!=r['events_sha256']:
    raise ValueError('Cropped gate invalid')
   gate_receipts.append((read(path/'gate.json'),read(path/'token_window_events.json')))
  if gate_receipts[0]!=gate_receipts[1]:raise ValueError('Cropped gates differ')
 original=crop_runner.build_author_predictor;runtime=None
 def build_fork(args,_repo):
  nonlocal runtime
  sys.path.insert(0,str(FORK));model=original(args,FORK)
  import sam2
  if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Wrong SAM2 package')
  if model.num_maskmem!=7 or model.mem_keep_ratio!=.25 or model.insurance_max_frames!=3 or not model.enable_insurance_bank:raise ValueError('Author memory config differs')
  model.ema_history_replace_old_tail_in_attn=False;model.ema_history_resident_prune=False
  runtime=TokenWindowRuntime(16384,3072);model.token_window_runtime=runtime
  return model
 crop_runner.build_author_predictor=build_fork
 sys.argv=['run_seq23_viewport.py','--checkpoint',str(a.checkpoint),'--detector',str(a.detector),'--gate-a',str(a.crop_gate_a),'--gate-b',str(a.crop_gate_b),'--output',str(a.output)]
 if a.mode=='gate':sys.argv.append('--engineering-gate')
 crop_runner.main()
 if runtime is None:raise ValueError('Window runtime did not activate')
 if digest(a.output/'prompt.json')!=base['prompt_sha256'] or digest(a.output/'prompt.png')!=digest(a.baseline/'prompt.png'):
  raise ValueError('Cropped initial prompt differs')
 for i in range(3):
  if digest(a.output/f'{i:06d}.png')!=digest(a.baseline/f'{i:06d}.png'):raise ValueError('Causal prefix differs')
 events=runtime.events;plans=[e for e in events if e['stage']=='admission_plan'];writes=[e for e in events if e['stage']=='after_write']
 if not plans or len(plans)!=len(writes) or any(e['total_tokens']>16384 or e['insurance_tokens']>3072 for e in writes) or any(p['condition_tokens']!=4096 or p['new_tokens']!=4096 for p in plans):raise ValueError('Token window accounting differs')
 (a.output/'token_window_events.json').write_text(json.dumps(events,indent=2)+'\n')
 result={'passed':True,'mode':a.mode,'sequence':'seq23','verification_sha256':digest(a.verification/'verification.json'),
  'baseline_completion_sha256':digest(a.baseline/'completion.json'),'records_sha256':digest(a.output/('gate.json' if a.mode=='gate' else 'frames.json')),
  'events_sha256':digest(a.output/'token_window_events.json'),'source_sha256':digest(Path(__file__)),
  'crop_runner_sha256':digest(ROOT/'scripts/run_seq23_viewport.py')}
 (a.output/'token_window_observation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
