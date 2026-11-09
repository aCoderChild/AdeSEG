"""Recover two original Run155 visual examples; do not rerun a cohort.

Native logits come from the archived bank. Regional uses the existing author
runner, stopping at46, and is accepted only if EVERY prefix output/memory hash
and score agrees with Run155. Original missing source snapshot is not replaced.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
import torch.nn.functional as F
from PIL import Image

from datasets.video import load_manifest
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, validate_output
from scripts.mavos_value_utils import atomic_json, atomic_torch_save
from scripts.run_author_baseline import tensor_hash
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT
import scripts.run_region_constrained_tome_gate as author_runner

DRIVE=CHECKPOINT_DEFAULT.parents[1]
RUN=DRIVE/'155_region_constrained_tome_polypgen_v1'
TARGETS=(25,46)


class PrefixComplete(Exception):
    pass


def run(out):
    out=validate_output(out);out.mkdir(parents=True,exist_ok=False)
    setup=json.loads((RUN/'full/setup.json').read_text())
    protocol=json.loads((RUN/'full/protocol.json').read_text())
    if digest(CHECKPOINT_DEFAULT)!=setup['checkpoint_sha256']:
        raise RuntimeError('Archived checkpoint differs')
    if digest(ROOT/'data/polypgen_sequence.jsonl')!=setup['manifest_sha256']:
        raise RuntimeError('Archived manifest differs')
    frames=load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')['seq18']
    refs={a:{x['frame']:x for x in json.loads((RUN/'full'/a/'seq18/frames.json').read_text())}
          for a in ['native','region_constrained_tome']}
    provenance={'scope':'Bounded seq18 visual reconstruction only; no new efficacy',
        'targets':list(TARGETS),'checkpoint_sha256':setup['checkpoint_sha256'],
        'threads':torch.get_num_threads(),'source_sha256':{p:digest(ROOT/p) for p in [
            'scripts/capture_run155_seq18_overlays.py','scripts/run_region_constrained_tome_gate.py',
            'modeling/region_constrained_tome.py','external/ToMe/tome/merge.py']},
        'original_module_source_matches':digest(ROOT/'modeling/region_constrained_tome.py')==setup['method_module_sha256'],
        'acceptance':'Every regional propagated logit and stored-memory hash, score metrics and object score through46 must match archive; no tolerance fallback.'}
    atomic_json(out/'identity.json',provenance)
    for relative in provenance['source_sha256']:
        dest=out/'source'/relative;dest.parent.mkdir(parents=True,exist_ok=True);dest.write_bytes((ROOT/relative).read_bytes())
    bank=torch.load(DRIVE/'143_official_native_feature_audit_v1/full/seq18/bank.pt',map_location='cpu',weights_only=False)['outputs']
    for frame in TARGETS:
        stored=bank['non_cond_frame_outputs'][frame]
        with Image.open(frames[frame].image_path) as im:shape=(im.height,im.width)
        with torch.inference_mode():logits=F.interpolate(stored['pred_masks'],size=shape,mode='bilinear',align_corners=False)
        if tensor_hash(logits)!=refs['native'][frame]['prediction_sha256'] or fingerprint(stored)['sha256']!=refs['native'][frame]['stored_sha256']:
            raise RuntimeError('Native archive reconstruction differs')
        atomic_torch_save(out/f'native_frame_{frame}.pt',logits.cpu())
    del bank
    repo,revision=author_runner.activate_author_repo()
    if revision!=setup['sam2_revision']:raise RuntimeError('Native source differs')
    model_holder={};original_build=author_runner.build_model;original_score=author_runner.score_frame;original_fingerprint=author_runner.fingerprint
    captured={};checks=[];current={}
    def build(*args,**kwargs):
        model=original_build(*args,**kwargs);model_holder['model']=model
        model_holder['parameter_digest']=author_runner.parameter_digest(model);return model
    def score(logits,frames,frame):
        row=original_score(logits,frames,frame);ref=refs['region_constrained_tome'][frame]
        for key in ('prediction_sha256','scores','present','absent_fp'):
            if row[key]!=ref[key]:raise RuntimeError(f'Regional archive mismatch at frame{frame}: {key}')
        current['frame']=frame
        if frame in TARGETS:captured[frame]=logits.detach().cpu().clone()
        return row
    def memory(value):
        result=original_fingerprint(value);frame=current['frame'];ref=refs['region_constrained_tome'][frame]
        if result['sha256']!=ref['stored_sha256'] or float(value['object_score_logits'].item())!=ref['object_score']:
            raise RuntimeError(f'Regional stored memory or presence differs at frame{frame}')
        checks.append({'frame':frame,'prediction_sha256':ref['prediction_sha256'],'stored_sha256':result['sha256']})
        atomic_json(out/'prefix_checks.json',checks)
        if frame==max(TARGETS):raise PrefixComplete()
        return result
    author_runner.build_model=build;author_runner.score_frame=score;author_runner.fingerprint=memory
    try:
        author_runner.run_condition('seq18','region_constrained_tome',frames,0,refs['region_constrained_tome'],
            CHECKPOINT_DEFAULT,protocol['model']['configuration'],repo,out)
        raise RuntimeError('Bounded replay did not stop as planned')
    except PrefixComplete:
        if [c['frame'] for c in checks]!=list(range(1,47)) or set(captured)!=set(TARGETS):
            raise RuntimeError('Incomplete replay evidence')
        if author_runner.parameter_digest(model_holder['model'])!=model_holder['parameter_digest']:
            raise RuntimeError('Pretrained weights changed')
        for frame,logits in captured.items():atomic_torch_save(out/f'regional_frame_{frame}.pt',logits)
        atomic_json(out/'completion.json',{'passed':True,'native_target_count':2,'regional_prefix_targets_verified':46,
            'prefix_checks_sha256':digest(out/'prefix_checks.json'),
            'outputs':{p.name:digest(p) for p in out.glob('*_frame_*.pt')},'scope':provenance['scope']})
    finally:
        author_runner.build_model=original_build;author_runner.score_frame=original_score;author_runner.fingerprint=original_fingerprint


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);args=parser.parse_args()
    existed=args.output_dir.exists()
    try:run(args.output_dir)
    except Exception as error:
        if not existed and args.output_dir.exists():atomic_json(args.output_dir/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise
