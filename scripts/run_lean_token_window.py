#!/usr/bin/env python3
"""Run isolated Lean worktree as exact no-op or experimental token window."""
import argparse,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_token_window_storage import TokenWindowRuntime
from scripts.memory_token_io import digest
import scripts.run_lean_indexed_spatial_rope as runner
FORK=ROOT/'external/Lean-SAM2-token-window'
ORIGINAL=ROOT/'external/Lean-SAM2'
PROTOCOL=ROOT/'protocols/lean_token_window_candidate_v1.json'
PATCHED=('sam2/modeling/sam2_base.py','sam2/sam2_video_predictor_topk_insurance.py')
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser();p.add_argument('--variant',choices=('noop','window'),required=True);p.add_argument('--mode',choices=('gate','candidate'),required=True)
 for key in ('checkpoint','merged','reference','indexed-reference','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','output'):
  p.add_argument('--'+key,type=Path,required=True)
 for key in ('noop-gate-a','noop-gate-b','window-gate-a','window-gate-b'):p.add_argument('--'+key,type=Path)
 p.add_argument('--sequence',required=True);a=p.parse_args()
 if a.mode=='gate' and a.sequence!='seq3':raise ValueError('Engineering gates require seq3')
 if a.mode=='candidate' and a.sequence not in ('seq8','seq11','seq20'):
  raise ValueError('Sequence not supported by this runner; seq23 needs matched no-inset preprocessing')
 rev=subprocess.check_output(['git','-C',str(FORK),'rev-parse','HEAD'],text=True).strip()
 original_rev=subprocess.check_output(['git','-C',str(ORIGINAL),'rev-parse','HEAD'],text=True).strip()
 changed=subprocess.check_output(['git','-C',str(FORK),'diff','--name-only','HEAD'],text=True).splitlines()
 if rev!=original_rev or sorted(changed)!=sorted(PATCHED):raise ValueError('Isolated Lean patch identity changed')
 ref_done=read(a.indexed_reference/'completion.json');ref=read(a.indexed_reference/'records.json')
 if not ref_done['passed'] or digest(a.indexed_reference/'records.json')!=ref_done['records_sha256']:raise ValueError('Indexed reference invalid')
 identity={'runner_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),'revision':rev,
  'patch_sha256':{k:digest(FORK/k) for k in PATCHED},'policy_sha256':digest(ROOT/'modeling/token_memory_window.py'),
  'runtime_sha256':digest(ROOT/'modeling/lean_token_window_storage.py'),'observer_sha256':digest(ROOT/'modeling/lean_token_storage_observer.py'),
  'checkpoint_sha256':digest(a.checkpoint),'cpu_verification_sha256':digest(a.cpu_verification),
  'indexed_gate_a_sha256':digest(a.indexed_gate_a/'records.json'),'indexed_gate_b_sha256':digest(a.indexed_gate_b/'records.json')}
 def gates(paths,variant):
  result=[]
  for path in paths:
   if path is None:raise ValueError('Two independent prerequisite gates required')
   g=read(path/'token_window_observation.json')
   if not g['passed'] or g['variant']!=variant or g['mode']!='gate' or g['identity']!=identity or digest(path/'records.json')!=g['records_sha256'] or digest(path/'token_window_events.json')!=g['events_sha256']:
    raise ValueError('Prerequisite gate changed')
   result.append((read(path/'records.json'),read(path/'token_window_events.json')))
  if result[0]!=result[1]:raise ValueError('Prerequisite gates differ')
  return result[0]
 if a.variant=='window':
  noop,_=gates((a.noop_gate_a,a.noop_gate_b),'noop')
  # seq3 reference is available from the indexed engineering gate in all modes.
  if noop!=read(a.indexed_gate_a/'records.json'):raise ValueError('No-op gate differs from indexed reference')
  if a.mode=='candidate':gates((a.window_gate_a,a.window_gate_b),'window')
 elif a.mode!='gate':raise ValueError('No-op is an engineering gate only')
 build=runner.build_author_predictor;runtime=None
 def build_fork(args,_repo):
  nonlocal runtime
  sys.path.insert(0,str(FORK));model=build(args,FORK)
  import sam2
  if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Wrong SAM2 package loaded')
  if a.variant=='window':
   if model.num_maskmem!=7 or model.mem_keep_ratio!=.25 or model.insurance_max_frames!=3 or not model.enable_insurance_bank:
    raise ValueError('Unexpected author memory configuration')
   model.ema_history_replace_old_tail_in_attn=False;model.ema_history_resident_prune=False
   runtime=TokenWindowRuntime(16384,3072);model.token_window_runtime=runtime
  return model
 runner.build_author_predictor=build_fork
 args=['run_lean_indexed_spatial_rope.py','--mode','gate' if a.mode=='gate' else 'candidate','--sequence',a.sequence]
 for key in ('checkpoint','merged','reference','cpu_verification','observed_reference','output'):args.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
 args.extend(['--gate-a',str(a.indexed_gate_a),'--gate-b',str(a.indexed_gate_b)]);sys.argv=args;runner.main()
 records=read(a.output/'records.json');events=[] if runtime is None else runtime.events
 if a.variant=='noop':
  for key in ('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256'):
   if records[key]!=ref[key]:raise ValueError('Isolated no-op differs from reference: '+key)
 else:
  writes=[e for e in events if e['stage']=='after_write'];plans=[e for e in events if e['stage']=='admission_plan']
  if not writes or len(writes)!=len(plans) or any(e['total_tokens']>16384 or e['insurance_tokens']>3072 for e in writes):raise ValueError('Admission budget or coverage invalid')
  if any(p['planned_reserved_tokens']>16384 or not p['selected_frames'] or p['selected_frames'][-1]!=p['frame'] for p in plans):raise ValueError('Admission plan invalid')
  if a.mode=='gate' and not any(p['evict_frames'] for p in plans):raise ValueError('Gate did not exercise eviction')
 (a.output/'token_window_events.json').write_text(json.dumps(events,indent=2)+'\n')
 result={'passed':True,'variant':a.variant,'mode':a.mode,'sequence':a.sequence,'identity':identity,
  'reference_records_sha256':ref_done['records_sha256'],'records_sha256':digest(a.output/'records.json'),
  'events_sha256':digest(a.output/'token_window_events.json'),'events':len(events)}
 (a.output/'token_window_observation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
