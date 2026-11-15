#!/usr/bin/env python3
"""Capture frozen 256x256 early features in the audited Lean raw-window pipeline."""
import argparse,hashlib,json,subprocess,sys
from contextlib import contextmanager
from pathlib import Path
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_channel_feature_capture import capture_channel_features
from modeling.lean_token_window_storage import TokenWindowRuntime
from scripts.memory_token_io import digest
from scripts.run_lean_token_window import FORK,PATCHED
import scripts.run_lean_indexed_spatial_rope as runner


def read(p):return json.loads(p.read_text())
def main():
    p=argparse.ArgumentParser();p.add_argument('--mode',choices=('gate','candidate'),required=True);p.add_argument('--sequence',required=True)
    for k in ('checkpoint','merged','reference','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','raw-gate','raw-verification','feature-gate','feature-reference','preflight','output'):
        p.add_argument('--'+k,type=Path,required=True)
    p.add_argument('--pipeline-gate',type=Path)
    a=p.parse_args();manifest=read(a.preflight/'manifest.json');pre=read(a.preflight/'completion.json');fg=read(a.feature_gate/'verification.json');rv=read(a.raw_verification/'verification.json')
    if not pre['passed'] or pre['manifest_sha256']!=digest(a.preflight/'manifest.json') or manifest['protocol_sha256']!=digest(ROOT/'protocols/lean_channel_attention_training_pilot_v1.json') or manifest['source_sha256']!=digest(ROOT/'scripts/preflight_lean_channel_training.py'):
        raise ValueError('Split preflight changed')
    if not fg['passed'] or fg['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_channel_feature_gates.py') or fg['capture_sha256']!=digest(ROOT/'modeling/lean_channel_feature_capture.py') or fg['preflight_sha256']!=digest(a.preflight/'manifest.json'):
        raise ValueError('Feature observer gate changed')
    if not rv['passed'] or rv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_token_window_gates.py') or rv['identity']['checkpoint_sha256']!=digest(a.checkpoint) or rv['identity']['runtime_sha256']!=digest(ROOT/'modeling/lean_token_window_storage.py') or rv['identity']['policy_sha256']!=digest(ROOT/'modeling/token_memory_window.py') or any(rv['identity']['patch_sha256'][x]!=digest(FORK/x) for x in PATCHED):raise ValueError('Raw-window author gate changed')
    raw=read(a.raw_gate/'token_window_observation.json')
    if not raw['passed'] or raw['identity']!=rv['identity'] or raw['records_sha256']!=digest(a.raw_gate/'records.json') or raw['events_sha256']!=digest(a.raw_gate/'token_window_events.json'):
        raise ValueError('Raw-window reference changed')
    allowed=set(manifest['groups']['train']+manifest['groups']['validation'])
    if a.sequence not in allowed or (a.mode=='gate' and a.sequence!='seq3'):raise ValueError('Sequence outside training/validation or gate scope')
    if a.mode=='candidate' and a.sequence=='seq3':raise ValueError('Use verified seq3 gate data')
    if a.mode=='candidate':
        if a.pipeline_gate is None:raise ValueError('Independent pipeline gates required')
        pv=read(a.pipeline_gate/'verification.json')
        if not pv['passed'] or pv['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_channel_pipeline_gates.py') or pv['runner_sha256']!=digest(Path(__file__)) or pv['preflight_sha256']!=digest(a.preflight/'manifest.json'):
            raise ValueError('Pipeline verification changed')
    changed=subprocess.check_output(['git','-C',str(FORK),'diff','--name-only','HEAD'],text=True).splitlines();rev=subprocess.check_output(['git','-C',str(FORK),'rev-parse','HEAD'],text=True).strip()
    if sorted(changed)!=sorted(PATCHED) or rev!=rv['identity']['revision']:raise ValueError('Raw-window worktree changed')
    source=manifest['sequences'][a.sequence];targets=set(source['targets']);observations=[];build=runner.build_author_predictor;rope=runner.indexed_spatial_rope;runtime=None
    def build_fork(args,_repo):
        nonlocal runtime
        sys.path.insert(0,str(FORK));model=build(args,FORK)
        import sam2
        if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()) or model.num_maskmem!=7 or model.mem_keep_ratio!=.25 or model.insurance_max_frames!=3 or not model.enable_insurance_bank:raise ValueError('Model configuration changed')
        model.ema_history_replace_old_tail_in_attn=False;model.ema_history_resident_prune=False
        runtime=TokenWindowRuntime(16384,3072);model.token_window_runtime=runtime;return model
    @contextmanager
    def capture(model,positions):
        with rope(model,positions):
            with capture_channel_features(model,targets,a.output,observations):yield
    runner.build_author_predictor=build_fork;runner.indexed_spatial_rope=capture
    try:
        sys.argv=['run_lean_indexed_spatial_rope.py','--mode',a.mode,'--sequence',a.sequence]
        for k in ('checkpoint','merged','reference','cpu_verification','observed_reference','output'):sys.argv.extend(['--'+k.replace('_','-'),str(getattr(a,k))])
        sys.argv.extend(['--gate-a',str(a.indexed_gate_a),'--gate-b',str(a.indexed_gate_b)]);runner.main()
    finally:runner.build_author_predictor=build;runner.indexed_spatial_rope=rope
    if runtime is None or {x['frame'] for x in observations}!=targets:raise ValueError('Missing feature captures')
    records=read(a.output/'records.json');writes=[e for e in runtime.events if e['stage']=='after_write']
    if not writes or any(e['total_tokens']>16384 or e['insurance_tokens']>3072 for e in writes):raise ValueError('Raw memory budget changed')
    if a.mode=='gate':
        for name in ('records.json','positions.json','selections.json','memory_events.json'):
            if digest(a.output/name)!=digest(a.raw_gate/name):raise ValueError('Capture changed raw gate: '+name)
        if runtime.events!=read(a.raw_gate/'token_window_events.json'):raise ValueError('Raw memory event trace changed')
        reference_receipt=read(a.feature_reference/'feature_capture_receipt.json')
        for obs in observations:
            old=next(x for x in reference_receipt['features'] if x['frame']==obs['frame'])
            if any(obs[k]!=old[k] for k in ('feature_sha256','foreground_sha256','background_sha256')):raise ValueError('Observed feature differs from verified reference')
    for e in observations:e['file_sha256']=digest(a.output/'features'/f"{e['frame']:06d}.pt")
    (a.output/'token_window_events.json').write_text(json.dumps(runtime.events,indent=2)+'\n')
    cpu=read(a.cpu_verification);pipeline={'feature_interface':'sam2.1_hiera_base_plus.conv_s0.fpn0','raw_identity':rv['identity'],
      'cpu_identity':cpu['identity'],'feature_observer_sha256':digest(ROOT/'modeling/lean_channel_feature_capture.py'),
      'image_loader_sha256':digest(FORK/'sam2/utils/misc.py'),'torch':torch.__version__,'device':'cpu','dtype':'float32','resolution':1024}
    canonical=json.dumps(pipeline,sort_keys=True,separators=(',',':')).encode();pipeline_sha=hashlib.sha256(canonical).hexdigest()
    if a.mode=='candidate' and pipeline_sha!=pv['feature_pipeline_sha256']:raise ValueError('Runtime feature pipeline differs from engineering gates')
    result={'passed':True,'mode':a.mode,'sequence':a.sequence,'features':observations,'targets':sorted(targets),'pipeline':pipeline,'feature_pipeline_sha256':pipeline_sha,
      'checkpoint_sha256':digest(a.checkpoint),'preflight_sha256':digest(a.preflight/'manifest.json'),'records_sha256':digest(a.output/'records.json'),
      'events_sha256':digest(a.output/'token_window_events.json'),'feature_gate_sha256':digest(a.feature_gate/'verification.json'),'runner_sha256':digest(Path(__file__))}
    (a.output/'channel_feature_receipt.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps({'passed':True,'sequence':a.sequence,'pipeline_sha256':pipeline_sha}))
if __name__=='__main__':main()
