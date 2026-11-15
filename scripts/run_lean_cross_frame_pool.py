#!/usr/bin/env python3
"""Engineer a fixed-budget cross-frame pool on the existing Lean worktree."""
import argparse,json,subprocess,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_cross_frame_pool import CrossFramePoolRuntime,capture_pool_scores
from modeling.lean_token_window_storage import TokenWindowRuntime
from scripts.memory_token_io import digest
import scripts.run_lean_indexed_spatial_rope as runner
from scripts.run_lean_token_window import FORK,PATCHED
PROTOCOL=ROOT/'protocols/lean_cross_frame_token_pool_v1.json'
SOURCES=('modeling/lean_cross_frame_pool.py','modeling/cross_frame_token_pool.py','modeling/lean_early_contrast_anchor.py',
         'modeling/prompt_feature_refinement.py','modeling/lean_token_window_storage.py','modeling/lean_token_storage_observer.py')
def read(p):return json.loads(p.read_text())

def main():
 p=argparse.ArgumentParser();p.add_argument('--strategy',choices=('score_only','usage','contrast','distributed'),required=True);p.add_argument('--mode',choices=('gate','candidate'),required=True)
 for k in ('checkpoint','merged','reference','indexed-reference','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','raw-gate','raw-verification','output'):p.add_argument('--'+k,type=Path,required=True)
 for k in ('score-gate-a','score-gate-b','strategy-gate-a','strategy-gate-b'):p.add_argument('--'+k,type=Path)
 p.add_argument('--sequence',required=True);a=p.parse_args()
 if a.mode=='gate' and a.sequence!='seq3':raise ValueError('Gate must use seq3')
 if a.mode=='candidate':raise ValueError('Candidate video execution disabled until independent pool-gate verification exists')
 rv=read(a.raw_verification/'verification.json');raw=read(a.raw_gate/'token_window_observation.json');raw_record=read(a.raw_gate/'records.json')
 if not rv['passed'] or rv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or not raw['passed'] or raw['variant']!='window' or raw['mode']!='gate' or raw['identity']!=rv['identity'] or digest(a.raw_gate/'records.json')!=raw['records_sha256'] or digest(a.raw_gate/'token_window_events.json')!=raw['events_sha256']:
  raise ValueError('Raw-window engineering gate invalid')
 ident0=rv['identity']
 if any(ident0['patch_sha256'][x]!=digest(FORK/x) for x in PATCHED) or ident0['runtime_sha256']!=digest(ROOT/'modeling/lean_token_window_storage.py') or ident0['policy_sha256']!=digest(ROOT/'modeling/token_memory_window.py') or ident0['checkpoint_sha256']!=digest(a.checkpoint):raise ValueError('Raw-window implementation changed')
 identity={'runner_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),'sources':{x:digest(ROOT/x) for x in SOURCES},
           'raw_verification_sha256':digest(a.raw_verification/'verification.json'),'raw_gate_records_sha256':raw['records_sha256'],
           'patch_sha256':ident0['patch_sha256'],'checkpoint_sha256':digest(a.checkpoint),'revision':ident0['revision']}
 def check_score_gates():
  out=[]
  for path in (a.score_gate_a,a.score_gate_b):
   if path is None:raise ValueError('Two score-only gates required')
   r=read(path/'pool_observation.json')
   if not r['passed'] or r['strategy']!='score_only' or r['identity']!=identity or digest(path/'records.json')!=r['records_sha256'] or digest(path/'pool_events.json')!=r['events_sha256'] or digest(path/'source_scores.json')!=r['source_scores_sha256']:
    raise ValueError('Score-only gate invalid')
   out.append((read(path/'records.json'),read(path/'pool_events.json'),read(path/'source_scores.json')))
  if out[0]!=out[1]:raise ValueError('Score-only gates differ')
 if a.strategy!='score_only':check_score_gates()
 changed=subprocess.check_output(['git','-C',str(FORK),'diff','--name-only','HEAD'],text=True).splitlines()
 rev=subprocess.check_output(['git','-C',str(FORK),'rev-parse','HEAD'],text=True).strip()
 if sorted(changed)!=sorted(PATCHED) or rev!=identity['revision']:raise ValueError('Worktree changed')
 build=runner.build_author_predictor;indexed=runner.indexed_spatial_rope;runtime=None;scores=[]
 def build_fork(args,_repo):
  nonlocal runtime
  sys.path.insert(0,str(FORK));model=build(args,FORK)
  import sam2
  if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Wrong author package')
  if model.num_maskmem!=7 or model.mem_keep_ratio!=.25 or model.insurance_max_frames!=3 or not model.enable_insurance_bank:raise ValueError('Memory configuration changed')
  model.ema_history_replace_old_tail_in_attn=False;model.ema_history_resident_prune=False
  runtime=TokenWindowRuntime(16384,3072) if a.strategy=='score_only' else CrossFramePoolRuntime(16384,3072,a.strategy)
  model.token_window_runtime=runtime
  return model
 @contextmanager
 def indexed_with_scores(model,positions):
  with indexed(model,positions):
   with capture_pool_scores(model,scores):yield
 runner.build_author_predictor=build_fork;runner.indexed_spatial_rope=indexed_with_scores
 args=['run_lean_indexed_spatial_rope.py','--mode','gate','--sequence',a.sequence]
 for k in ('checkpoint','merged','reference','cpu_verification','observed_reference','output'):args.extend(['--'+k.replace('_','-'),str(getattr(a,k))])
 args.extend(['--gate-a',str(a.indexed_gate_a),'--gate-b',str(a.indexed_gate_b)]);sys.argv=args;runner.main()
 records=read(a.output/'records.json')
 if runtime is None or not scores:raise ValueError('Pool instrumentation did not execute')
 if a.strategy=='score_only':
  if records!=raw_record or runtime.events!=read(a.raw_gate/'token_window_events.json'):raise ValueError('Score capture changed raw-window trajectory')
 else:
  plans=[e for e in runtime.events if e['stage']=='admission_plan'];writes=[e for e in runtime.events if e['stage']=='after_write']
  if not plans or len(plans)!=len(writes):raise ValueError('Missing admissions')
  for plan,write in zip(plans,writes):
   if plan['frame']!=write['frame'] or plan['planned_reserved_tokens']!=write['spatial_tokens']+3072 or plan['condition_tokens']!=4096 or plan['new_tokens']!=4096 or plan['pool_budget']!=5120 or plan['pool_kept']!=sum(plan['pool_counts'].values()) or plan['pool_kept']!=min(5120,plan['pool_candidates']) or write['total_tokens']>16384:
    raise ValueError('Pool budget or protected memory differs')
  if a.strategy=='distributed' and max(len(x['pool_counts']) for x in plans)<=5:raise ValueError('Distributed gate did not span more sources')
 pos=read(a.output/'positions.json');rawpos=read(a.raw_gate/'positions.json')
 if len(pos)!=len(rawpos) or [x['pointer_keys'] for x in pos]!=[x['pointer_keys'] for x in rawpos]:raise ValueError('Pointer counts changed')
 (a.output/'pool_events.json').write_text(json.dumps(runtime.events,indent=2)+'\n');(a.output/'source_scores.json').write_text(json.dumps(scores,indent=2)+'\n')
 result={'passed':True,'strategy':a.strategy,'mode':a.mode,'sequence':a.sequence,'identity':identity,'records_sha256':digest(a.output/'records.json'),
  'events_sha256':digest(a.output/'pool_events.json'),'source_scores_sha256':digest(a.output/'source_scores.json'),'source_score_events':len(scores)}
 (a.output/'pool_observation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
