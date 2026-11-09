"""Matched one-thread C0 using the existing native evaluator; no candidate run."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
from PIL import Image
import torch
from datasets.video import load_manifest
from modeling.complementary_memory.integration import NativeComplementaryRead
from scripts.mavos_value_utils import activate_author_repo,atomic_json
from scripts.memory_token_io import digest,validate_output,prompt_index
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT,MANIFEST,_source_hashes,_guard_sources,_model_digest,_evaluate_video,_validate_cached_evaluation

DESIGN=ROOT/'protocols/rde_value_repair_efficacy_polypgen_v1.json'


def run(out,old_reference,audit,ownership):
    complete=json.loads((audit/'completion.json').read_text());audited=json.loads((audit/'results.json').read_text())
    if not complete['passed'] or complete['results_sha256']!=digest(audit/'results.json') or not audited['passed']:
        raise RuntimeError('Native saved-mask audit incomplete or altered')
    if audited['reference_setup_sha256']!=digest(old_reference/'setup.json'):raise RuntimeError('Original native reference differs from audited reference')
    design=json.loads(DESIGN.read_text());sources=_source_hashes()
    for n in ['scripts/collect_rde_native_reference.py','protocols/rde_value_repair_efficacy_polypgen_v1.json','modeling/complementary_memory/integration.py','modeling/native_memory_layout.py']:
        sources[n]=digest(ROOT/n)
    for path in (ROOT/'external/sam2/sam2').rglob('*.py'):
        sources[str(path.relative_to(ROOT))]=digest(path)
    config='external/sam2/sam2/'+design['model']['configuration']
    sources[config]=digest(ROOT/config)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(design['training']['primary_seed'])
    if torch.__version__!=design['training']['torch_version']:raise RuntimeError('Frozen Torch version differs')
    out=validate_output(out);out.mkdir(parents=True,exist_ok=True)
    with (out/'writer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB)
        ownership['acquired']=True
        identity={'sources':sources,'checkpoint_sha256':digest(CHECKPOINT_DEFAULT),'old_reference_audit_sha256':digest(audit/'results.json'),
            'threads':1,'torch_version':torch.__version__,'scope':'Matched native validation only; no adapter training/candidate evaluation'}
        if (out/'identity.json').exists():
            if json.loads((out/'identity.json').read_text())!=identity:raise RuntimeError('Reference resume identity differs')
        else:
            atomic_json(out/'identity.json',identity);atomic_json(out/'protocol_frozen.json',design)
            for n in sources:
                p=out/'source'/n;p.parent.mkdir(parents=True,exist_ok=True);p.write_bytes((ROOT/n).read_bytes())
        setup_hash=digest(out/'identity.json');repo,revision=activate_author_repo()
        if revision!=design['model']['sam2_revision'] or identity['checkpoint_sha256']!=design['model']['checkpoint_sha256']:raise RuntimeError('Native model identity differs')
        model=build_author_predictor(argparse.Namespace(method='sam2',device='cpu',checkpoint=CHECKPOINT_DEFAULT,model_config=design['model']['configuration']),repo)
        model.requires_grad_(False).eval();model_hash=_model_digest(model);videos=load_manifest(MANIFEST,'dev');results={}
        for name in design['data']['validation_sequences']:
            frames=videos[name];directory=out/'evaluation'/name
            if directory.exists():
                summary=_validate_cached_evaluation(directory,frames,prompt_index(frames,{'initialization':'first_labeled_frame'}),'native',model_hash,setup_hash)
                trace=json.loads((directory/'native_read_audit.json').read_text())
                if trace['frames_sha256']!=digest(directory/'frames.jsonl') or trace['summary_sha256']!=digest(directory/'summary.json'):
                    raise RuntimeError('Cached native read audit differs from its prediction/summary')
                if len(trace['calls'])!=summary['target_frames'] or any(r['layers']!=[0,1,2,3] or not r['native_positions_verified'] or not r['stored_mask_alignment'] for r in trace['calls']):
                    raise RuntimeError('Cached native four-layer alignment coverage differs')
            else:
                with NativeComplementaryRead(model,None,trace_only=True) as observer:
                    summary=_evaluate_video(model,frames,None,None,name,out,setup_hash,sources)
                if len(observer.calls)!=summary['target_frames'] or any(r['layers']!=[0,1,2,3] for r in observer.calls):raise RuntimeError('Native full-video layer coverage differs')
                atomic_json(directory/'native_read_audit.json',{'calls':observer.calls,'all_four_layers':True,
                    'frames_sha256':digest(directory/'frames.jsonl'),'summary_sha256':digest(directory/'summary.json'),'efficacy_method':False})
            prior=old_reference/'native_validation/evaluation'/name
            if digest(prior/'frames.jsonl')!=audited['receipts'][name]['frames_sha256']:raise RuntimeError('Original frame reference changed')
            old_rows=[json.loads(line) for line in (prior/'frames.jsonl').read_text().splitlines()]
            new_rows=[json.loads(line) for line in (directory/'frames.jsonl').read_text().splitlines()]
            if [r['frame'] for r in old_rows]!=[r['frame'] for r in new_rows]:raise RuntimeError('Native frame schedule differs')
            mask_matches=0;logit_matches=0
            for old,new in zip(old_rows,new_rows):
                path=prior/'masks'/f"{old['frame']:06d}.png"
                if digest(path)!=old['mask_sha256']:raise RuntimeError('Old prediction mask changed after audit')
                with Image.open(path) as a,Image.open(directory/'masks'/f"{new['frame']:06d}.png") as b:
                    mask_matches+=int(np.array_equal(np.asarray(a),np.asarray(b)))
                logit_matches+=int(old['prediction_sha256']==new['prediction_sha256'])
            results[name]={'summary':summary,'prior_mask_matches':mask_matches,'prior_logit_hash_matches':logit_matches,
                'target_frames':len(new_rows),'summary_sha256':digest(directory/'summary.json'),'frames_sha256':digest(directory/'frames.jsonl'),
                'read_audit_sha256':digest(directory/'native_read_audit.json')}
            atomic_json(out/'progress.json',{'completed_sequences':list(results),'current_results':results,'candidate_efficacy_started':False})
            print(json.dumps({'native_completed':name,'mask_matches':mask_matches,'targets':len(new_rows)}),flush=True)
        _guard_sources(sources)
        atomic_json(out/'results.json',{'passed':True,'sequences':results,'identity_sha256':setup_hash,'scope':identity['scope'],
            'reuse_this_matched_reference':True,'candidate_efficacy_started':False})
        atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),'candidate_efficacy_started':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--old-reference',type=Path,required=True);p.add_argument('--audit',type=Path,required=True);a=p.parse_args()
    ownership={}
    try:run(a.output_dir,a.old_reference,a.audit,ownership)
    except Exception as e:
        if ownership.get('acquired'):atomic_json(a.output_dir/'failure.json',{'error':str(e),'type':type(e).__name__})
        raise
