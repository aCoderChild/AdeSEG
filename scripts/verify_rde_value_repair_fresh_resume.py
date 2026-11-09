"""Fresh-process replay of both fixed seq2 optimizer checkpoints; no efficacy."""
import argparse
import copy
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from modeling.rde_value_repair import MemoryValueRepair,author_consistency_loss
from scripts.mavos_value_utils import activate_author_repo,atomic_json,atomic_torch_save
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT,PROTOCOL,_guard_sources,_model_digest,_tree_digest
from scripts.verify_rde_value_repair import update
from scripts.verify_token_conditioned_key_adapter import require_tree_equal


def run(reference,out,rde_repository):
    completion=json.loads((reference/'completion.json').read_text())
    if not completion['passed'] or completion['results_sha256']!=digest(reference/'results.json'):
        raise RuntimeError('Bounded reference gate is incomplete or altered')
    identity=json.loads((reference/'identity.json').read_text());_guard_sources(identity['sources'])
    design=json.loads((ROOT/'protocols/rde_value_repair_polypgen_v1.json').read_text())
    if digest(rde_repository/'model/model.py')!=design['prior_work']['source_sha256']:raise RuntimeError('RDE source differs')
    out=validate_output(out);out.mkdir(parents=True,exist_ok=False)
    own={'script':digest(Path(__file__)),'reference_results_sha256':digest(reference/'results.json'),
         'checkpoint_sha256':digest(CHECKPOINT_DEFAULT),'torch_version':torch.__version__,'threads':1}
    if own['checkpoint_sha256']!=identity['checkpoint_sha256'] or own['torch_version']!=identity['torch_version']:raise RuntimeError('Runtime/checkpoint differs')
    atomic_json(out/'identity.json',own)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    base=json.loads(PROTOCOL.read_text());repo,revision=activate_author_repo()
    if revision!=base['model']['sam2_revision']:raise RuntimeError('SAM2 revision differs')
    model=build_author_predictor(argparse.Namespace(method='sam2',device='cpu',checkpoint=CHECKPOINT_DEFAULT,model_config=base['model']['configuration']),repo)
    model.requires_grad_(False).eval();core=_model_digest(model);criterion=author_consistency_loss(rde_repository)
    reference_rows=json.loads((reference/'results.json').read_text())['sequences']['seq2']['arms'];results={}
    for arm,expected in reference_rows.items():
        path=reference/f'seq2_{arm}_resume.pt'
        if digest(path)!=expected['checkpoint_sha256']:raise RuntimeError('Saved optimizer checkpoint differs')
        replay_results=[]
        for repeat in range(2):
            checkpoint=torch.load(path,weights_only=True,map_location='cpu');bank_hash=_tree_digest(checkpoint['bank'])
            adapter=MemoryValueRepair();adapter.load_state_dict(checkpoint['adapter']);optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=1e-3)
            optimizer.load_state_dict(checkpoint['optimizer']);torch.random.set_rng_state(checkpoint['rng'])
            records=[update(model,adapter,optimizer,checkpoint['features'],checkpoint['bank'],checkpoint['target'],checkpoint['teacher'],checkpoint['baseline_keys'],criterion,checkpoint['weight']) for _ in range(2)]
            require_tree_equal(expected['records'][2:],records)
            final={'adapter':copy.deepcopy(adapter.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':torch.random.get_rng_state().clone(),'records':records}
            if _tree_digest(checkpoint['bank'])!=bank_hash or _model_digest(model)!=core:raise RuntimeError('Native state changed during resumed update')
            replay_results.append(final)
        require_tree_equal(*replay_results)
        atomic_torch_save(out/f'{arm}_final.pt',replay_results[0])
        results[arm]={'reference_update_records_bit_exact':True,'independent_reload_final_optimizer_and_adapter_exact':True,
                      'final_state_sha256':_tree_digest(replay_results[0]),'source_checkpoint_sha256':digest(path)}
    _guard_sources(identity['sources'])
    atomic_json(out/'results.json',{'passed':True,'arms':results,'sam2_parameters_unchanged':True,
        'scope':'Fresh-process fixed seq2 two-update replay from each saved optimizer checkpoint; not whole-training resume','efficacy_authorized':False})
    atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),'efficacy_authorized':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--rde-repository',type=Path,required=True);a=p.parse_args()
    try:run(a.reference,a.output_dir,a.rde_repository)
    except Exception as e:
        if a.output_dir.exists():atomic_json(a.output_dir/'failure.json',{'error':str(e),'type':type(e).__name__})
        raise
