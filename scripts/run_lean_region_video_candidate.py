#!/usr/bin/env python3
"""Full-video three-region candidate, using existing automatic Lean drivers.

This runner is prepared for use only after independently verified fixed-read
results warrant a propagation test. It preserves the author's initial prompt
and sequence-23 viewport; it performs no new detector pass during propagation.
"""
import argparse,json,sys
from contextlib import contextmanager
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_region_token_pool import RegionPoolRuntime,capture_region_scores
from scripts.memory_token_io import digest
import scripts.run_lean_token_window as raw_runner
import scripts.run_seq23_token_window as crop_window
import scripts.run_lean_indexed_spatial_rope as indexed_runner
import scripts.run_seq23_viewport as crop_runner


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser(add_help=False)
    p.add_argument('--candidate-sequence',choices=('seq11','seq23'),required=True)
    p.add_argument('--pool-verification',type=Path,required=True);p.add_argument('--diagnostic-verification',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True)
    a,rest=p.parse_known_args();pool=read(a.pool_verification/'verification.json');diag=read(a.diagnostic_verification/'verification.json')
    if not pool['passed'] or pool['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_region_token_pool_gates.py') or pool['identity']['sources']['modeling/lean_region_token_pool.py']!=digest(ROOT/'modeling/lean_region_token_pool.py') or pool['identity']['protocol_sha256']!=digest(ROOT/'protocols/lean_region_token_pool_v1.json'):
        raise ValueError('Region pool engineering gates changed')
    if any(digest(ROOT/name)!=sha for name,sha in pool['identity']['sources'].items()) or pool['identity']['runner_sha256']!=digest(ROOT/'scripts/run_lean_region_token_pool.py'):
        raise ValueError('Verified region sources changed')
    if not diag['passed'] or diag['sequence']!=a.candidate_sequence or diag['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_region_priority_repeat.py'):
        raise ValueError('Fixed-read diagnostic verification changed')
    if '--mode' not in rest or rest[rest.index('--mode')+1]!='candidate':raise ValueError('Expected full candidate mode')
    if a.candidate_sequence=='seq11':
        if '--sequence' not in rest or rest[rest.index('--sequence')+1]!='seq11' or '--variant' not in rest or rest[rest.index('--variant')+1]!='window':raise ValueError('Expected seq11 window driver')
        driver=raw_runner;engine=indexed_runner
    else:driver=crop_window;engine=crop_runner
    original_runtime=driver.TokenWindowRuntime;original_rope=engine.indexed_spatial_rope;runtime=[];scores=[]
    def region_runtime(budget,reserve):
        obj=RegionPoolRuntime(budget,reserve,'three_region');runtime.append(obj);return obj
    @contextmanager
    def region_rope(model,positions):
        with original_rope(model,positions):
            with capture_region_scores(model,scores):yield
    driver.TokenWindowRuntime=region_runtime;engine.indexed_spatial_rope=region_rope
    try:
        sys.argv=[driver.__file__,*rest,'--output',str(a.output)];driver.main()
    finally:
        driver.TokenWindowRuntime=original_runtime;engine.indexed_spatial_rope=original_rope
    if len(runtime)!=1 or not scores:raise ValueError('Region runtime did not execute')
    events=runtime[0].events;plans=[e for e in events if e['stage']=='admission_plan'];writes=[e for e in events if e['stage']=='after_write']
    if len(plans)!=len(writes) or not plans:raise ValueError('Missing region admissions')
    for plan,write in zip(plans,writes):
        if plan['frame']!=write['frame'] or plan['condition_tokens']!=4096 or plan['new_tokens']!=4096 or plan['pool_kept']!=min(plan['pool_budget'],plan['pool_candidates']) or plan['planned_reserved_tokens']!=write['spatial_tokens']+3072 or write['total_tokens']>16384:
            raise ValueError('Region budget or protected memory changed')
    (a.output/'region_source_scores.json').write_text(json.dumps(scores,indent=2)+'\n')
    receipt={'passed':True,'sequence':a.candidate_sequence,'strategy':'three_region','scope':'automatic, forward-only, fixed token budget',
      'driver_receipt_sha256':digest(a.output/'token_window_observation.json'),'events_sha256':digest(a.output/'token_window_events.json'),
      'source_scores_sha256':digest(a.output/'region_source_scores.json'),'pool_gate_sha256':digest(a.pool_verification/'verification.json'),
      'diagnostic_sha256':digest(a.diagnostic_verification/'verification.json'),'region_module_sha256':digest(ROOT/'modeling/lean_region_token_pool.py'),
      'runner_sha256':digest(Path(__file__))}
    (a.output/'region_video_receipt.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps(receipt,indent=2))
if __name__=='__main__':main()
