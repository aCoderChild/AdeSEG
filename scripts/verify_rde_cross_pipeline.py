"""Compare actual native-reference and adapter evaluation paths on fixed TRAIN clips."""
import argparse
from contextlib import contextmanager
from functools import wraps
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
from modeling.rde_value_repair import MemoryValueRepair
from scripts.evaluate_rde_value_repair import evaluation_sources
from scripts.mavos_value_utils import activate_author_repo,atomic_json
from scripts.memory_token_io import digest,validate_output,prompt_index
from scripts.rde_value_evaluation import evaluate_condition
from scripts.run_author_baseline import build_author_predictor,tensor_hash
from scripts.run_rde_value_training import verify_receipt
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT,MANIFEST,_guard_sources,_model_digest,_evaluate_video

PROTOCOL=ROOT/'protocols/rde_value_repair_cross_pipeline_gate_v1.json'


def gate_sources():
    sources=evaluation_sources()
    for name in ['scripts/verify_rde_cross_pipeline.py','protocols/rde_value_repair_cross_pipeline_gate_v1.json']:
        sources[name]=digest(ROOT/name)
    return sources


@contextmanager
def capture_logits(model):
    original=model.propagate_in_video;records=[]
    @wraps(original)
    def observed(*args,**kwargs):
        iterator=original(*args,**kwargs)
        try:
            for frame,ids,logits in iterator:
                records.append({'frame':frame,'ids':list(ids),'shape':list(logits.shape),'raw_sha256':tensor_hash(logits)})
                yield frame,ids,logits
        finally:iterator.close()
    model.propagate_in_video=observed
    try:yield records
    finally:model.propagate_in_video=original


def run(out,native_reference,ownership):
    native,native_sha=verify_receipt(native_reference);native_identity=json.loads((native_reference/'identity.json').read_text())
    _guard_sources(native_identity['sources'])
    if not native.get('reuse_this_matched_reference') or native_identity['threads']!=1 or native_identity['torch_version']!=torch.__version__:
        raise RuntimeError('Matched native runtime unavailable')
    sources=gate_sources();out=validate_output(out);out.mkdir(parents=True,exist_ok=False)
    ownership['created']=True
    design=json.loads(PROTOCOL.read_text());base=json.loads((ROOT/'protocols/rde_value_repair_efficacy_polypgen_v1.json').read_text())
    identity={'sources':sources,'matched_native_results_sha256':native_sha,'checkpoint_sha256':digest(CHECKPOINT_DEFAULT),'torch_version':torch.__version__}
    if identity['checkpoint_sha256']!=base['model']['checkpoint_sha256']:raise RuntimeError('Checkpoint differs')
    atomic_json(out/'identity.json',identity);atomic_json(out/'protocol_frozen.json',design)
    for name in sources:
        path=out/'source'/name;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/name).read_bytes())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(base['training']['primary_seed'])
    repo,revision=activate_author_repo()
    if revision!=base['model']['sam2_revision']:raise RuntimeError('SAM2 revision differs')
    model=build_author_predictor(argparse.Namespace(method='sam2',device='cpu',checkpoint=CHECKPOINT_DEFAULT,model_config=base['model']['configuration']),repo)
    model.requires_grad_(False).eval();core=_model_digest(model);videos=load_manifest(MANIFEST,'dev');results={}
    fields=['frame','layers','memory_frames','spatial_keys','pointer_keys','pointer_input_sha256','pointer_position_sha256','native_positions_verified','stored_mask_alignment']
    for name in design['sequences']:
        all_frames=videos[name];prompt=prompt_index(all_frames,{'initialization':'first_labeled_frame'});frames=all_frames[:prompt+design['frames_from_first_prompt']]
        with capture_logits(model) as reference_logits,NativeComplementaryRead(model,None,trace_only=True) as observer:
            reference_summary=_evaluate_video(model,frames,None,None,name,out/'native_collector',digest(out/'identity.json'),sources)
        reference_rows=[json.loads(x) for x in (out/'native_collector/evaluation'/name/'frames.jsonl').read_text().splitlines()]
        if len(reference_rows)!=10 or len(observer.calls)!=10:raise RuntimeError('Fixed target coverage differs')
        if max(r['spatial_keys'] for r in observer.calls)!=28672:raise RuntimeError('Full native capacity was not exercised')
        comparisons={}
        for condition,adapter in [('adapter_runner_native',None),('adapter_runner_zero',MemoryValueRepair().requires_grad_(False).eval())]:
            with capture_logits(model) as actual_logits:
                summary=evaluate_condition(model,frames,adapter,name,out/condition,digest(out/'identity.json'))
            if actual_logits!=reference_logits:raise RuntimeError(f'Raw-logit parity failed for {name}/{condition}')
            directory=out/condition/'evaluation'/name;rows=[json.loads(x) for x in (directory/'frames.jsonl').read_text().splitlines()]
            if len(rows)!=len(reference_rows):raise RuntimeError('Per-frame coverage differs')
            for original,row in zip(reference_rows,rows):
                if any(row[k]!=original[k] for k in ['frame','present','scores','absent_fp']):raise RuntimeError('Per-frame metric ownership/value differs')
                a=out/'native_collector/evaluation'/name/'masks'/f"{row['frame']:06d}.png";b=directory/'masks'/a.name
                with Image.open(a) as x,Image.open(b) as y:
                    if not np.array_equal(np.asarray(x),np.asarray(y)):raise RuntimeError('Saved native mask pixels differ')
            if adapter is not None:
                for left,row in zip(observer.calls,rows):
                    if {k:left[k] for k in fields}!={k:row['memory_read'][k] for k in fields}:raise RuntimeError('Cross-pipeline memory trace differs')
            for metric in ['present_dice','present_iou','present_boundary_f','gt_empty_false_alarm_rate','propagation']:
                if summary[metric]!=reference_summary[metric]:raise RuntimeError(f'Summary aggregation differs: {metric}')
            comparisons[condition]={'common_raw_logits_exact':True,'saved_masks_and_metrics_exact':True,'targets':len(rows)}
        if _model_digest(model)!=core:raise RuntimeError('Frozen SAM2 changed')
        results[name]={'targets':len(reference_rows),'maximum_spatial_keys':28672,'comparisons':comparisons,'four_layer_trace_exact':True}
        atomic_json(out/f'{name}_common_logits.json',reference_logits);_guard_sources(sources)
        print(json.dumps({'cross_pipeline_passed':name}),flush=True)
    result={'passed':True,'cross_pipeline_parity_passed':True,'sources':sources,'sequences':results,'matched_native_results_sha256':native_sha,'scope':design['scope'],'candidate_outcomes_examined':False}
    atomic_json(out/'results.json',result);atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),'candidate_outcomes_examined':False})


if __name__=='__main__':
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output-dir',type=Path,required=True);parser.add_argument('--native-reference',type=Path,required=True);args=parser.parse_args()
    ownership={}
    try:run(args.output_dir,args.native_reference,ownership)
    except Exception as error:
        if ownership.get('created'):atomic_json(args.output_dir/'failure.json',{'type':type(error).__name__,'error':str(error)})
        raise
