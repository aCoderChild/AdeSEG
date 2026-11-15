#!/usr/bin/env python3
"""Freeze generalized capture source/runtime identities on CPU; no inference."""
import argparse,hashlib,json,subprocess,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

FORK=ROOT/'external/Lean-SAM2-token-window'
SOURCES=('scripts/freeze_source_group_runtime_v1.py','scripts/preflight_source_group_capture_v1.py',
         'scripts/run_source_group_capture_v1.py','scripts/audit_source_group_capture_v1.py',
         'scripts/run_author_baseline.py','scripts/run_lean_auto_multiscale.py','scripts/memory_token_io.py','scripts/source_group_label_guard_v1.py',
         'scripts/train_region_mixed_loss_v1.py',
         'modeling/lean_mps_decoder_observer.py','modeling/lean_memory_capture.py',
         'modeling/lean_token_storage_observer.py','modeling/lean_indexed_spatial_rope_mps.py',
         'modeling/lean_mps_blocking_pointer_position.py','modeling/lean_multiscale_anchor.py',
         'modeling/token_memory_window.py','modeling/lean_token_window_storage.py',
         'modeling/prompt_feature_refinement.py','modeling/early_feature_prompt.py','datasets/video.py','datasets/io.py',
         'training/tracking.py','evaluation/segmentation.py','evaluation/davis2017/metrics.py',
         'protocols/source_group_capture_v1.json','protocols/lean_multiscale_anchor_v1.json',
         'protocols/lean_auto_multiscale_early_prompt_v1.json')

def verify_fork_state(revision,status,expected_revision,patches):
    expected_status=sorted(' M '+name for name in patches)
    if revision!=expected_revision or sorted(status)!=expected_status:raise ValueError('Fork revision/dirty paths differ')


def validate_source_freeze(freeze,done,actual_sha,current):
    if not done.get('passed') or done.get('source_freeze_sha256')!=actual_sha or not freeze.get('passed') or freeze.get('identity')!=current:
        raise ValueError('Frozen live source identity differs')


def current_identity(preflight,reference,checkpoint):
    pre=json.loads((preflight/'manifest.json').read_text());done=json.loads((preflight/'completion.json').read_text())
    refpath=reference/'channel_feature_receipt.json';ref=json.loads(refpath.read_text());raw=ref['pipeline']['raw_identity'];sig=ref['pipeline']['cpu_identity']['source_signature']
    if not done.get('passed') or done.get('manifest_sha256')!=digest(preflight/'manifest.json') or pre.get('preflight_sha256')!=digest(ROOT/'scripts/preflight_source_group_capture_v1.py') or pre.get('reference_capture_sha256')!=digest(refpath) or pre.get('checkpoint_sha256')!=digest(checkpoint) or pre.get('config_sha256')!=digest(ROOT/'protocols/source_group_capture_v1.json') or pre.get('manifest_sha256')!=digest(ROOT/'data/polypgen_sequence.jsonl') or any(digest(ROOT/name)!=sha for name,sha in pre['source_sha256'].items()):raise ValueError('Preflight/reference/checkpoint identity differs')
    canonical=hashlib.sha256(json.dumps(ref['pipeline'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if ref.get('feature_pipeline_sha256')!=canonical or sig.get('checkpoint_sha256')!=digest(checkpoint) or raw.get('checkpoint_sha256')!=digest(checkpoint):raise ValueError('Historical feature/checkpoint identity differs')
    for key,path in (('lean_runner_sha256','scripts/run_author_baseline.py'),('runner_sha256','scripts/run_lean_auto_multiscale.py'),('prompt_module_sha256','modeling/prompt_feature_refinement.py'),('early_module_sha256','modeling/early_feature_prompt.py'),('memory_module_sha256','modeling/lean_multiscale_anchor.py'),('memory_protocol_sha256','protocols/lean_multiscale_anchor_v1.json'),('protocol_sha256','protocols/lean_auto_multiscale_early_prompt_v1.json')):
        if digest(ROOT/path)!=sig[key]:raise ValueError('Historical implementation differs: '+key)
    for key,path in (('policy_sha256','modeling/token_memory_window.py'),('runtime_sha256','modeling/lean_token_window_storage.py')):
        if digest(ROOT/path)!=raw[key]:raise ValueError('Historical token policy differs')
    if any(digest(FORK/k)!=v for k,v in raw['patch_sha256'].items()):raise ValueError('Fork patch differs')
    revision=subprocess.check_output(['git','-C',str(FORK),'rev-parse','HEAD'],text=True).strip()
    status=subprocess.check_output(['git','-C',str(FORK),'status','--porcelain','--untracked-files=all'],text=True).splitlines()
    verify_fork_state(revision,status,raw['revision'],raw['patch_sha256'])
    for key,path in (('bypass_sha256',FORK/'checkpoints/bypass_bottleneck_base.pth'),('lean_config_sha256',FORK/'configs/lean_sam2/lean_sam2.yaml')):
        if digest(path)!=sig[key]:raise ValueError('Historical model asset differs: '+key)
    return {'preflight_sha256':digest(preflight/'manifest.json'),'reference_capture_sha256':digest(refpath),'checkpoint_sha256':digest(checkpoint),
            'fork_revision':revision,'fork_status':status,'fork_patches':raw['patch_sha256'],
            'source_sha256':{name:digest(ROOT/name) for name in SOURCES},
            'bypass_sha256':sig['bypass_sha256'],'lean_config_sha256':sig['lean_config_sha256']}

def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('preflight','reference-capture','checkpoint','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();identity=current_identity(a.preflight,a.reference_capture,a.checkpoint)
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);path=out/'source_freeze.json'
    path.write_text(json.dumps({'passed':True,'scope':'CPU source freeze only; not a capture gate or patient-independence claim.','identity':identity},indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'source_freeze_sha256':digest(path)},indent=2)+'\n')
    print(json.dumps({'passed':True,'source_count':len(identity['source_sha256']),'fork_status':identity['fork_status']},indent=2))
if __name__=='__main__':main()
