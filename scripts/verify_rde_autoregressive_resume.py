"""Fresh-process replay against complete golden own-rollout bank/cache/state."""
import argparse
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from modeling.rde_value_repair import MemoryValueRepair,author_consistency_loss
from scripts.mavos_value_utils import activate_author_repo,atomic_json
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT,PROTOCOL,_guard_sources,_model_digest,_tree_digest
from scripts.verify_rde_autoregressive_training import optimizer_for,replay
from scripts.verify_token_conditioned_key_adapter import require_tree_equal


def run(reference,out,rde_repository):
    c=json.loads((reference/'completion.json').read_text())
    if not c['passed'] or c['results_sha256']!=digest(reference/'results.json'):raise RuntimeError('Own-rollout reference gate incomplete or altered')
    identity=json.loads((reference/'identity.json').read_text());_guard_sources(identity['sources'])
    if digest(CHECKPOINT_DEFAULT)!=identity['checkpoint_sha256'] or digest(rde_repository/'model/model.py')!=identity['rde_source_sha256'] or torch.__version__!=identity['torch_version']:
        raise RuntimeError('Native model, author loss or runtime differs')
    out=validate_output(out);out.mkdir(parents=True,exist_ok=False)
    sources={**identity['sources'],'scripts/verify_rde_autoregressive_resume.py':digest(Path(__file__))}
    atomic_json(out/'identity.json',{'sources':sources,'reference_results_sha256':digest(reference/'results.json'),'threads':1})
    for n in sources:
        p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    repo,revision=activate_author_repo();base=json.loads(PROTOCOL.read_text())
    if revision!=base['model']['sam2_revision']:raise RuntimeError('SAM2 revision differs')
    model=build_author_predictor(argparse.Namespace(method='sam2',device='cpu',checkpoint=CHECKPOINT_DEFAULT,model_config=base['model']['configuration']),repo)
    model.requires_grad_(False).eval();core=_model_digest(model);criterion=author_consistency_loss(rde_repository)
    prior=json.loads((reference/'results.json').read_text());rows={}
    for name,seq in prior['sequences'].items():
        for arm,arm_result in seq['arms'].items():
            path=reference/f'{name}_{arm}_after8.pt';expected_path=reference/f'{name}_{arm}_expected.pt'
            if digest(path)!=arm_result['checkpoint_sha256'] or digest(expected_path)!=arm_result['expected_state_sha256']:raise RuntimeError('Checkpoint/golden state checksum differs')
            checkpoint=torch.load(path,weights_only=True,map_location='cpu');expected=torch.load(expected_path,weights_only=True,map_location='cpu')
            if _tree_digest(expected)!=arm_result['expected_tree_sha256']:raise RuntimeError('Golden state content differs')
            adapter=MemoryValueRepair();adapter.load_state_dict(checkpoint['adapter']);optimizer=optimizer_for(adapter)
            optimizer.load_state_dict(checkpoint['optimizer']);torch.random.set_rng_state(checkpoint['rng'])
            bank,cache,records=replay(model,adapter,optimizer,checkpoint['samples'],checkpoint['bank'],checkpoint['teacher_cache'],criterion,checkpoint['weight'])
            actual={'adapter':adapter.state_dict(),'optimizer':optimizer.state_dict(),'bank':bank,'teacher_cache':cache,'records':records,'rng':torch.random.get_rng_state()}
            require_tree_equal(expected,actual)
            if _model_digest(model)!=core:raise RuntimeError('SAM2 changed during resumed own-rollout')
            rows[f'{name}/{arm}']={'golden_tree_sha256':_tree_digest(actual),'records_exact':True,'bank_cache_adapter_optimizer_rng_exact':True}
    _guard_sources(sources)
    atomic_json(out/'results.json',{'passed':True,'scope':'Fresh-process two-update own-rollout resume, seq2/seq3 both arms; not whole-training runner safety',
        'fixtures':rows,'sam2_parameters_unchanged':True,'efficacy_authorized':False})
    atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),'efficacy_authorized':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--rde-repository',type=Path,required=True);a=p.parse_args()
    try:run(a.reference,a.output_dir,a.rde_repository)
    except Exception as e:
        if a.output_dir.exists():atomic_json(a.output_dir/'failure.json',{'error':str(e),'type':type(e).__name__})
        raise
