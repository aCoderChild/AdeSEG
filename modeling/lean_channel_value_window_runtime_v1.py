"""Source-bound exploratory MPS runtime for the trained channel-value arm.

This is a new development runtime, not a promotion or independent validation.
It builds the existing Lean token-window model and issues the same scoped
runtime lease after checking the fixed-read artifacts and model fingerprints.
No validation or target label is opened here and no CPU/MPS parity is run.
"""
from contextlib import ExitStack, contextmanager
import json
import os
from pathlib import Path
import sys

import torch

from modeling.lean_boundary_window_runtime_v1 import verify_live_identity
from modeling.lean_indexed_spatial_rope_mps import indexed_spatial_rope_mps
from modeling.lean_live_state_audit import module_tensor_fingerprints
from modeling.lean_mps_blocking_pointer_position import blocking_mps_pointer_position
from modeling.lean_verified_runtime_lease import _FACTORY_ISSUER, _RuntimeLease
from scripts.channel_value_fixed_read_v1 import (ARMS, deps, load_model, load_prepared,
    phase_inputs, require_environment_flags, require_parity, require_plan, validated_runs)
from scripts.memory_token_io import digest
from scripts.resume_source_group_capture_v1 import BASE, OUT
from scripts.verify_token_window_refiner_routing_mps import build_model

ROOT=Path(__file__).resolve().parents[1]
STUDY=OUT/'channel_value_fixed_read_v1_8rvuon9v'
LEGACY=OUT/'boundary_video_factorial_v1_7c3zaram'
LEGACY_AUDIT=OUT/'boundary_video_factorial_v1_7c3zaram_independent_audit_attempt01/audit.json'
LEGACY_AUDIT_SHA='021f44175c8bddbee9968c61389e9db3edc7b7eb356e3447ed64b7dcdc00c445'
LEGACY_MODEL_SHA='1187507e769b839586a3aa2f2c5bb9afa61a51a620898734cfe58dcc4c4e29b5'
HASHES={
 'plan':'1798ed7fe8b51798a09e165bfb97be7016be141836f4094582359601774c3d3d',
 'parity':'82bb137ec42fd547f71d67d6f30d901e1a9c9a961c5aaf31e71b1f7087cb127c',
 'verification':'582624d21fd516e0b1a85d9614eb59d588a7dd819706e7341caaac7103b114d3',
 'scores':'f70c7dad16fe9ad57da58d0ae8a2d2831009ad25caf55e04f3c9305ae9b72b08',
 'audit':'7c39f69a6c0699161c3512db5d2af3d656d8a6a21d04546e5eff672f6e9182f5',
 'checkpoint':'43548f1ec016acb6b7c76345740cefade80cc71092e323320db15af92f289548',
 'manifest':'827f6be88046f89761313a67e6d0ed84d35b809ec553539f32104b53cb111c24',
}


def require(value,message):
    if not value:raise ValueError(message)


def bound_json(path,expected):
    require(digest(path)==expected,f'Frozen artifact differs: {path.name}')
    data=json.loads(Path(path).read_text())
    require(digest(path)==expected,f'Frozen artifact changed during parsing: {path.name}')
    return data


def verify_study(study=STUDY):
    """Verify selected checkpoint and all original fixed-read evidence, no labels."""
    study=Path(study).resolve(strict=True)
    require(study==STUDY.resolve(strict=True),'Unexpected fixed-read study')
    manifest_path=OUT/'source_group_boundary_factorial_v1_bvqxd475/manifest'
    manifest,sha=load_prepared(manifest_path)
    require(sha==HASHES['manifest']==digest(manifest_path/'manifest.json'),'Training manifest differs')
    plan_path=study/'plan/plan.json';parity_path=study/'parity/parity.json'
    bound_json(plan_path,HASHES['plan']);bound_json(parity_path,HASHES['parity'])
    require(phase_inputs(study,plan_path,parity_path)==[HASHES['plan'],HASHES['parity']],
            'Frozen phase identities differ')
    plan=require_plan(plan_path,manifest,sha);require_parity(parity_path,sha,plan_path,plan)
    require(digest(plan_path)==HASHES['plan'] and digest(parity_path)==HASHES['parity'],
            'Plan or parity changed')
    runs=validated_runs(study,manifest,sha,plan_path,parity_path,plan)
    verification=bound_json(study/'verification/verification.json',HASHES['verification'])
    score=bound_json(study/'scores/scores.json',HASHES['scores'])
    audit=bound_json(study/'independent_audit/audit.json',HASHES['audit'])
    require(verification.get('passed') is True and verification.get('manifest_sha256')==sha and
            verification.get('dependency_sha256')==deps() and verification.get('label_guard',{}).get('open_attempts')==0 and
            score.get('passed') is True and score.get('manifest_sha256')==sha and
            score.get('verification_sha256')==HASHES['verification'] and score.get('prelabel_guard',{}).get('open_attempts')==0 and
            audit.get('passed') is True and audit.get('error_count')==0 and audit.get('score_sha256')==HASHES['scores'] and
            audit.get('label_files_checked')==50 and audit.get('prediction_masks_checked')==500 and
            audit.get('prelabel_guard',{}).get('open_attempts')==0,'Frozen replay, score or independent audit differs')
    for arm,record in runs.items():
        require(verification['runs'][arm]=={k:record[k] for k in ('completion_sha256','checkpoint_sha256','frozen_sha256')},
                f'{arm}: fixed-read replay binding differs')
    candidate=runs['foreground_background_asinh']
    require(candidate['checkpoint_sha256']==HASHES['checkpoint'],'Selected checkpoint differs')
    return manifest,sha,plan,candidate['payload']


def expected_lean_fingerprints():
    """Reuse fingerprints from the previously independently audited MPS A run."""
    audit=bound_json(LEGACY_AUDIT,LEGACY_AUDIT_SHA);scores=LEGACY/'scores/results.json'
    require(audit.get('passed') is True and audit.get('error_count')==0 and
            audit.get('score_sha256')==digest(scores) and audit.get('run')==str(LEGACY),
            'Previous baseline audit differs')
    score=bound_json(scores,audit['score_sha256']);completion=LEGACY/'seq20/A/completion.json'
    require(score.get('passed') is True and score['run_receipt_sha256']['seq20']['A']==digest(completion),
            'Previous A completion differs')
    rec=bound_json(completion,score['run_receipt_sha256']['seq20']['A']);runtime=completion.parent/'verification/runtime_verification.json'
    require(rec.get('passed') is True and rec.get('arm')=='A' and rec.get('full_video') is True and
            rec.get('runtime_verification_sha256')==digest(runtime),'Previous A runtime differs')
    state=bound_json(runtime,rec['runtime_verification_sha256']);fingerprints=runtime.parent/'model_tensor_fingerprints.json'
    require(state.get('passed') is True and state.get('device')=='mps float32' and
            state.get('model_tensor_fingerprints_sha256')==LEGACY_MODEL_SHA==digest(fingerprints),
            'Previous A model fingerprint differs')
    return bound_json(fingerprints,LEGACY_MODEL_SHA),LEGACY_MODEL_SHA


def runtime_sources():
    paths=('modeling/lean_channel_value_window_runtime_v1.py','modeling/lean_channel_value_routing_v1.py',
           'scripts/verify_channel_value_video_study_v1.py','scripts/run_channel_value_video_v1.py',
           'scripts/verify_channel_value_video_smoke_v1.py',
           'scripts/run_channel_value_video_study_v1.py','scripts/score_channel_value_video_v1.py',
           'scripts/crossfit_observer_capture_shared_v1.py','scripts/preflight_exploratory_crossfit_capture_v2.py',
           'protocols/EXPLORATORY_CROSSFIT_CAPTURE_PREFLIGHT_V2.md','protocols/CHANNEL_VALUE_ASINH_VIDEO_V1.md',
           'modeling/lean_verified_runtime_lease.py','modeling/lean_boundary_window_runtime_v1.py',
           'scripts/verify_token_window_refiner_routing_mps.py','modeling/lean_indexed_spatial_rope_mps.py',
           'modeling/lean_mps_blocking_pointer_position.py','modeling/lean_live_state_audit.py',
           'modeling/lean_multiscale_anchor.py','modeling/lean_token_window_storage.py',
           'modeling/token_memory_window.py','modeling/lean_token_storage_observer.py',
           'scripts/run_author_baseline.py','modeling/seq23_viewport.py','modeling/samurai_auto_prompt.py',
           'scripts/memory_token_io.py','datasets/video.py','datasets/io.py','data/polypgen_sequence.jsonl')
    return {p:digest(ROOT/p) for p in paths}


def verify_certificate(path,expected_sha):
    """Fast per-process check; the certificate records a one-time full snapshot."""
    proof=bound_json(path,expected_sha)
    require(proof.get('passed') is True and proof.get('study')==str(STUDY) and
            proof.get('study_hashes')==HASHES and proof.get('runtime_sources')==runtime_sources() and
            proof.get('training_dependencies')==deps() and proof.get('label_guard',{}).get('open_attempts')==0 and
            proof.get('legacy_model_fingerprint_sha256')==LEGACY_MODEL_SHA and
            proof.get('verifier_sha256')==digest(ROOT/'scripts/verify_channel_value_video_study_v1.py'),
            'Batch study certificate or source identity differs')
    manifest_path=OUT/'source_group_boundary_factorial_v1_bvqxd475/manifest/manifest.json'
    manifest=bound_json(manifest_path,HASHES['manifest'])
    plan=bound_json(STUDY/'plan/plan.json',HASHES['plan'])
    require(plan.get('dependency_sha256')==deps() and plan.get('manifest_sha256')==HASHES['manifest'] and
            proof.get('training_environment')==plan['environment'] and
            proof.get('feature_pipeline_sha256')==manifest['feature_pipeline_sha256'],
            'Frozen training plan differs')
    ck=STUDY/'runs/foreground_background_asinh/checkpoint.pt'
    require(digest(ck)==HASHES['checkpoint'],'Selected checkpoint changed')
    payload=torch.load(ck,map_location='cpu',weights_only=True)
    require(digest(ck)==HASHES['checkpoint'] and payload.get('study_arm')=='foreground_background_asinh' and
            payload.get('plan_sha256')==HASHES['plan'] and payload.get('parity_sha256')==HASHES['parity'] and
            payload.get('dependency_sha256')==deps(),'Selected checkpoint provenance differs')
    fingerprints=bound_json(LEGACY/'seq20/A/verification/model_tensor_fingerprints.json',LEGACY_MODEL_SHA)
    return manifest,plan,payload,fingerprints


@contextmanager
def verified_channel_value_window(*,checkpoint,certificate,certificate_sha256,output):
    require_environment_flags()
    require(torch.backends.mps.is_available(),'MPS unavailable')
    torch.use_deterministic_algorithms(True);torch.set_num_threads(2)
    out=Path(output).resolve();require(out.is_relative_to(BASE.resolve(strict=True)) and not out.exists(),
                                       'Runtime output must be new and under Drive experiment root')
    manifest,plan,payload,expected_model=verify_certificate(certificate,certificate_sha256)
    sha=HASHES['manifest']
    require(digest(checkpoint)==manifest['checkpoint_sha256'],'Backbone checkpoint differs')
    environment,identity=verify_live_identity(manifest,plan)
    fingerprint_sha=LEGACY_MODEL_SHA
    from scripts.verify_token_window_refiner_routing_mps import FORK
    sys.path.insert(0,str(FORK))
    model,runtime,selections=build_model(Path(checkpoint),expected_model)
    refiner=load_model(payload,'foreground_background_asinh',manifest,sha).to('mps').eval().requires_grad_(False)
    expected_refiner=module_tensor_fingerprints(refiner);positions=[]
    out.mkdir(parents=True,exist_ok=False)
    with ExitStack() as stack:
        stack.enter_context(indexed_spatial_rope_mps(model,positions))
        stack.enter_context(blocking_mps_pointer_position(model))
        lease=_RuntimeLease(model,refiner,runtime,expected_model,expected_refiner,
                            _issuer=_FACTORY_ISSUER,positions=positions,selections=selections)
        lease.validate_live(model,refiner,tensors=True)
        model_path=out/'model_tensor_fingerprints.json';ref_path=out/'refiner_tensor_fingerprints.json'
        model_path.write_text(json.dumps(expected_model,indent=2)+'\n')
        ref_path.write_text(json.dumps(expected_refiner,indent=2)+'\n')
        record={'passed':True,'scope':'Exploratory asinh video runtime; no independent validation.',
            'study':str(STUDY),'study_hashes':HASHES,'certificate_sha256':certificate_sha256,
            'runtime_sources':runtime_sources(),'source_sha256':digest(Path(__file__)),
            'model_tensor_fingerprints_sha256':digest(model_path),'refiner_tensor_fingerprints_sha256':digest(ref_path),
            'legacy_model_fingerprint_sha256':fingerprint_sha,'environment':environment,'live_source_identity':identity,
            'dtype':'float32','device':'mps','memory_budget':16384,'insurance_reserve':3072}
        (out/'runtime_verification.json').write_text(json.dumps(record,indent=2)+'\n')
        try:
            yield model,refiner,lease
            lease.validate_live(model,refiner,tensors=True)
        finally:lease.close()
