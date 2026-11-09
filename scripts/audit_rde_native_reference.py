"""Read-only audit/re-score of the existing compatible native validation masks."""
import argparse
import ast
import json
from pathlib import Path
import subprocess
import sys
import zipfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from datasets.video import load_manifest
from scripts.mavos_value_utils import activate_author_repo,atomic_json
from scripts.memory_token_io import digest,validate_output,prompt_index
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT,MANIFEST,_guard_sources,_model_digest,_validate_cached_evaluation


def native_source_equivalent(old,new):
    """Only the non-neural tree-checksum implementation may differ."""
    def normalized(text):
        tree=ast.parse(text)
        tree.body=[n for n in tree.body if not isinstance(n,ast.FunctionDef) or n.name!='_tree_digest']
        return ast.dump(tree,include_attributes=False)
    return normalized(old)==normalized(new)


def run(reference,out):
    design_path=ROOT/'protocols/rde_value_repair_efficacy_polypgen_v1.json';design=json.loads(design_path.read_text())
    setup=json.loads((reference/'setup.json').read_text());core=setup['core']
    relative='scripts/run_token_conditioned_key_adapter.py'
    other={n:h for n,h in core['sources'].items() if n!=relative};_guard_sources(other)
    with zipfile.ZipFile(reference/'source_snapshot.zip') as archive:
        name=next(n for n in archive.namelist() if n.endswith(relative))
        old=archive.read(name)
    import hashlib
    if hashlib.sha256(old).hexdigest()!=core['sources'][relative]:raise RuntimeError('Original native evaluator snapshot differs')
    if not native_source_equivalent(old.decode(),(ROOT/relative).read_text()):raise RuntimeError('Native evaluator code changed beyond the checksum-only helper')
    current_runner_sha=digest(ROOT/relative)
    if setup['torch']!=torch.__version__:raise RuntimeError('Native reference Torch version differs')
    old_threads=[n.args[0].value for n in ast.walk(ast.parse(old.decode())) if isinstance(n,ast.Call)
        and isinstance(n.func,ast.Attribute) and n.func.attr=='set_num_threads' and len(n.args)==1 and isinstance(n.args[0],ast.Constant)]
    if len(old_threads)!=1:raise RuntimeError('Reference thread configuration is ambiguous')
    if core['checkpoint_sha256']!=digest(CHECKPOINT_DEFAULT) or core['manifest_sha256']!=digest(MANIFEST):raise RuntimeError('Reference inputs/model identity differ')
    if core['checkpoint_sha256']!=design['model']['checkpoint_sha256'] or core['sam2_revision']!=design['model']['sam2_revision']:raise RuntimeError('New protocol model differs from reference')
    old_protocol=json.loads((reference/'protocol.json').read_text())
    if old_protocol['model']['configuration']!=design['model']['configuration']:
        raise RuntimeError('Native reference model configuration differs')
    out=validate_output(out);out.mkdir(parents=True,exist_ok=False)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
    repo,revision=activate_author_repo()
    subprocess.run(['git','diff','--exit-code','HEAD','--','sam2'],cwd=repo,check=True,capture_output=True)
    if revision!=core['sam2_revision']:raise RuntimeError('Native author revision differs')
    model=build_author_predictor(argparse.Namespace(method='sam2',device='cpu',checkpoint=CHECKPOINT_DEFAULT,model_config=design['model']['configuration']),repo)
    model.requires_grad_(False).eval();model_hash=_model_digest(model);videos=load_manifest(MANIFEST,'dev');summaries={};receipts={}
    for name in design['data']['validation_sequences']:
        directory=reference/'native_validation/evaluation'/name;frames=videos[name]
        summary=_validate_cached_evaluation(directory,frames,prompt_index(frames,{'initialization':'first_labeled_frame'}),'native',model_hash,setup['core_sha256'])
        summaries[name]=summary
        receipts[name]={'summary_sha256':digest(directory/'summary.json'),'frames_sha256':digest(directory/'frames.jsonl'),
            'prediction_masks':{p.name:digest(p) for p in (directory/'masks').glob('*.png')}}
    _guard_sources(other)
    if digest(ROOT/relative)!=current_runner_sha:raise RuntimeError('Native evaluator changed during audit')
    result={'passed':True,'scope':'Existing native validation saved-mask/input/provenance/metric audit; no CNN inference rerun',
        'reference_setup_sha256':digest(reference/'setup.json'),'protocol_sha256':digest(design_path),'script_sha256':digest(Path(__file__)),
        'sam2_parameter_sha256':model_hash,'summaries':summaries,'receipts':receipts,
        'original_native_threads':old_threads[0],'new_protocol_threads':1,
        'runtime_matched_for_reuse':old_threads[0]==1,
        'checksum_only_source_difference_verified':current_runner_sha!=core['sources'][relative],
        'current_native_evaluator_sha256':current_runner_sha,
        'native_inference_rerun':False,'efficacy_authorized':False}
    atomic_json(out/'results.json',result);atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),'efficacy_authorized':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--reference',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args()
    try:run(a.reference,a.output_dir)
    except Exception as e:
        if a.output_dir.exists():atomic_json(a.output_dir/'failure.json',{'error':str(e),'type':type(e).__name__})
        raise
