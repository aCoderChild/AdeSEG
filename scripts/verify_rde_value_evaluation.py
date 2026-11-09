"""Fixed real TRAIN evaluation-glue gate; no candidate validation/test scoring."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from datasets.video import load_manifest
from modeling.rde_value_repair import MemoryValueRepair
from scripts.evaluate_rde_value_repair import evaluation_sources
from scripts.mavos_value_utils import activate_author_repo,atomic_json
from scripts.memory_token_io import digest,validate_output,prompt_index
from scripts.rde_value_evaluation import evaluate_condition,load_adapter
from scripts.run_author_baseline import build_author_predictor
from scripts.run_rde_value_training import verify_receipt
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT,MANIFEST,_guard_sources,_model_digest


def run(out,runner_gate):
    prior,prior_sha=verify_receipt(runner_gate)
    if not prior.get('runner_safety_passed'):raise RuntimeError('Actual training executor gate is required')
    _guard_sources(prior['training_sources']);sources=evaluation_sources();out=validate_output(out);out.mkdir(parents=True,exist_ok=False)
    design=json.loads((ROOT/'protocols/rde_value_repair_evaluation_gate_v1.json').read_text());base=json.loads((ROOT/'protocols/rde_value_repair_efficacy_polypgen_v1.json').read_text())
    identity={'sources':sources,'runner_gate_results_sha256':prior_sha,'checkpoint_sha256':digest(CHECKPOINT_DEFAULT),'torch_version':torch.__version__,'threads':1}
    atomic_json(out/'identity.json',identity);atomic_json(out/'protocol_frozen.json',design)
    for n in sources:
        q=out/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/n).read_bytes())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);repo,revision=activate_author_repo()
    if revision!=base['model']['sam2_revision'] or identity['checkpoint_sha256']!=base['model']['checkpoint_sha256']:raise RuntimeError('Native model differs')
    model=build_author_predictor(argparse.Namespace(method='sam2',device='cpu',checkpoint=CHECKPOINT_DEFAULT,model_config=base['model']['configuration']),repo)
    model.requires_grad_(False).eval();core=_model_digest(model);videos=load_manifest(MANIFEST,'dev')
    trained=runner_gate/'uninterrupted';arm='segmentation_plus_consistency';arm_result,_=verify_receipt(trained/'training'/arm)
    path=trained/'training'/arm/'adapter_final.pt'
    if digest(path)!=arm_result['final_adapter_file_sha256']:raise RuntimeError('Trained fixture adapter differs')
    nonzero=load_adapter(path,digest(trained/'identity.json'),arm)
    if nonzero.mlp.layers[-1].weight.count_nonzero()==0:raise RuntimeError('Nonzero inference fixture is still identity')
    variants={'native':None,'zero':MemoryValueRepair().requires_grad_(False).eval(),'nonzero':nonzero};rows={};changed=0
    for name in design['sequences']:
        frames=videos[name];start=prompt_index(frames,{'initialization':'first_labeled_frame'});frames=frames[start:start+design['frames_from_first_prompt']]
        signatures={}
        for variant,adapter in variants.items():
            summary=evaluate_condition(model,frames,adapter,name,out/variant,digest(out/'identity.json'))
            cached=evaluate_condition(model,frames,adapter,name,out/variant,digest(out/'identity.json'))
            if summary!=cached:raise RuntimeError('Actual cached sequence summary differs')
            directory=out/variant/'evaluation'/name
            records=[json.loads(x) for x in (directory/'frames.jsonl').read_text().splitlines()]
            signatures[variant]=[(r['prediction_sha256'],r['mask_sha256']) for r in records]
        if signatures['native']!=signatures['zero']:raise RuntimeError('Evaluation zero/native raw logits or masks differ')
        changed+=sum(a[0]!=b[0] for a,b in zip(signatures['native'],signatures['nonzero']))
        rows[name]={'targets':len(signatures['native']),'native_zero_raw_and_mask_exact':True,'cache_rescore_exact':True}
        if _model_digest(model)!=core:raise RuntimeError('SAM2 parameters changed')
        _guard_sources(sources);atomic_json(out/'progress.json',{'completed_sequences':list(rows)})
    if not changed:raise RuntimeError('Nonzero fixture adapter never affected inference')
    result={'passed':True,'evaluation_safety_passed':True,'evaluation_sources':sources,'sequences':rows,
        'nonzero_changed_logit_frames':changed,'sam2_parameters_unchanged':True,'scope':design['scope'],'candidate_efficacy_started':False}
    atomic_json(out/'results.json',result);atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),'candidate_efficacy_started':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--runner-gate',type=Path,required=True);a=p.parse_args()
    try:run(a.output_dir,a.runner_gate)
    except Exception as e:
        if a.output_dir.exists():atomic_json(a.output_dir/'failure.json',{'error':str(e),'type':type(e).__name__})
        raise
