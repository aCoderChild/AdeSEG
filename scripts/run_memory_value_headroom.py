"""Fixed28-case TRAIN diagnostic after source-bound real-frame read gate.

Native trajectories only; source GT replaces spatial V during isolated query
replays. Resume replays each unfinished video's native prefix and verifies saved
case outputs; complete videos are reused only with identical input/source hashes.
"""
import argparse
from collections import OrderedDict
import fcntl
import json
from pathlib import Path
import resource
import sys
import tempfile
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
import numpy as np
import torch
import torch.nn.functional as F

from datasets.io import load_label_mask
from datasets.video import load_manifest
from evaluation.segmentation import calculate_scores
from evaluation.davis2017.metrics import db_eval_boundary
from modeling.complementary_memory.training import capture_current_features, clone_tensors
from modeling.native_memory_layout import selected_memory_frames
from scripts.analyze_memory_value_headroom import ARMS, summarize
from scripts.mask_memory_teacher import capture_native_mask_inputs, same_frame_teachers
from scripts.memory_value_query import MEMORY_FIELDS, query, spatial_values
from scripts.mavos_value_utils import activate_author_repo, atomic_json, prepare
from scripts.memory_token_io import digest, validate_output
from scripts.prepare_mask_memory_headroom import DESIGN
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT, MANIFEST, PROTOCOL, _guard_sources, _model_digest, _tree_digest


def score(output, truth):
    with torch.inference_mode():
        pred = F.interpolate(output['pred_masks'].float(), size=truth.shape,
            mode='bilinear', align_corners=False)[0, 0].cpu().numpy() > 0
    if not np.isfinite(output['pred_masks'].float().cpu().numpy()).all():
        raise RuntimeError('Nonfinite query mask')
    scores = calculate_scores(pred, truth)
    scores['boundary_f'] = float(db_eval_boundary(pred, truth))
    return {'scores': scores, 'prediction_nonempty': bool(pred.any()),
            'object_score': float(output['object_score_logits'].item())}


def load_case(out, case_id, identity_hash):
    path, receipt = out/'cases'/f'{case_id}.json', out/'cases'/f'{case_id}.done.json'
    if not path.exists() and not receipt.exists(): return None
    if not path.exists() or not receipt.exists():
        raise RuntimeError(f'Unreceipted partial case requires audit before resume: {case_id}')
    done = json.loads(receipt.read_text())
    if done['identity_sha256'] != identity_hash or done['result_sha256'] != digest(path):
        raise RuntimeError('Saved case is altered or belongs to another implementation')
    row = json.loads(path.read_text())
    if row['id'] != case_id or row.get('identity_sha256') != identity_hash:
        raise RuntimeError('Case identifier differs from its receipt')
    return row


def run(args):
    gate = json.loads((args.read_gate/'results.json').read_text())
    complete = json.loads((args.read_gate/'completion.json').read_text())
    audit = json.loads((args.read_gate/'audit_results.json').read_text())
    if (not complete['passed'] or not audit['passed'] or not complete['full_train_diagnostic_integrity_passed']
        or complete['results_sha256'] != digest(args.read_gate/'results.json')
        or audit['results_sha256'] != complete['results_sha256']):
        raise RuntimeError('Full source-bound read gate and independent audit required')
    _guard_sources(gate['identity']['sources'])
    plan = json.loads(args.case_plan.read_text())
    if digest(args.case_plan) != gate['identity']['case_plan_sha256']:
        raise RuntimeError('Read gate belongs to another frozen case plan')
    design, protocol = json.loads(DESIGN.read_text()), json.loads(PROTOCOL.read_text())
    if digest(DESIGN) != plan['identity']['protocol_sha256']:
        raise RuntimeError('Frozen diagnostic protocol changed')
    sources = dict(gate['identity']['sources'])
    for name in ('scripts/run_memory_value_headroom.py', 'scripts/analyze_memory_value_headroom.py',
                 'tests/test_memory_value_headroom_analysis.py'):
        sources[name] = digest(ROOT/name)
    identity = {'sources': sources, 'gate_results_sha256': digest(args.read_gate/'results.json'),
        'gate_audit_sha256': digest(args.read_gate/'audit_results.json'),
        'case_plan_sha256': digest(args.case_plan), 'checkpoint_sha256': digest(CHECKPOINT_DEFAULT),
        'threads': 1, 'torch_version': torch.__version__, 'torch_build': torch.__config__.show()}
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    # Lock the exact attempt; never start a second trajectory writer on resume.
    with (out/'run.lock').open('a+') as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        identity_path = out/'identity.json'
        if identity_path.exists():
            if json.loads(identity_path.read_text()) != identity:
                raise RuntimeError('Resume identity differs; preserve this attempt')
        else:
            atomic_json(identity_path, identity)
            atomic_json(out/'protocol_frozen.json', design)
            atomic_json(out/'case_plan_frozen.json', plan)
            for relative in sources:
                path=out/'source'/relative;path.parent.mkdir(parents=True,exist_ok=True);path.write_bytes((ROOT/relative).read_bytes())
        identity_hash = digest(identity_path)
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(design['seed'])
        repo, revision = activate_author_repo()
        if revision != protocol['model']['sam2_revision'] or identity['checkpoint_sha256'] != protocol['model']['checkpoint_sha256']:
            raise RuntimeError('Native model identity changed')
        videos = load_manifest(MANIFEST, 'dev')
        all_cases = []
        for name in design['data']['sequences']:
            selected = [c for c in plan['cases'] if c['sequence']==name]
            if not selected: continue
            if name not in protocol['data']['train_sequences']:
                raise RuntimeError('Non-TRAIN data requested')
            start=plan['sequences'][name]['prompt_manifest_offset'];frames=videos[name][start:]
            files=[{'frame':f.frame_index,'image_sha256':digest(f.image_path),'mask_sha256':digest(f.mask_path)} for f in frames]
            source_path=out/'inputs'/f'{name}.json'
            if source_path.exists() and json.loads(source_path.read_text()) != files:
                raise RuntimeError('Sequence images/labels changed on resume')
            atomic_json(source_path,files)
            existing={c['id']:load_case(out,c['id'],identity_hash) for c in selected}
            if all(v is not None for v in existing.values()):
                all_cases.extend(existing.values());continue
            model = build_author_predictor(argparse.Namespace(method='sam2',device='cpu',
                checkpoint=CHECKPOINT_DEFAULT,model_config=protocol['model']['configuration']),repo)
            model.requires_grad_(False).eval();core=_model_digest(model)
            chosen={c['target_local_frame']:c for c in selected}
            with tempfile.TemporaryDirectory(dir=out) as temp:
                state,prompt=prepare(model,frames,Path(temp))
                if prompt != 0:raise RuntimeError('Frozen prompt changed')
                masks=OrderedDict();iterator=model.propagate_in_video(state,start_frame_idx=0)
                try:
                    while True:
                        t0=time.monotonic()
                        # Observe only a native propagation step, never teacher/oracle calls.
                        with capture_native_mask_inputs(model) as capture, torch.inference_mode():
                            try:frame,ids,_=next(iterator)
                            except StopIteration:break
                        if ids != [1] or capture['count'] != 1:
                            raise RuntimeError('Native source capture/frame/owner association changed')
                        masks[frame]=clone_tensors(capture['latest'])
                        while len(masks)>8:masks.popitem(last=False)
                        if frame not in chosen:continue
                        case=chosen[frame];native_bank=state['output_dict_per_obj'][0]
                        target={f:native_bank['non_cond_frame_outputs'][frame][f] for f in MEMORY_FIELDS}
                        target_hash=_tree_digest(target)
                        old=existing[case['id']]
                        if old is not None:
                            if old['native_propagation_sha256'] != target_hash:
                                raise RuntimeError('Native prefix differs from saved case during resume')
                            continue
                        bank=clone_tensors({'cond_frame_outputs':native_bank['cond_frame_outputs'],
                            'non_cond_frame_outputs':{k:v for k,v in native_bank['non_cond_frame_outputs'].items() if k<frame}})
                        bank_hash=_tree_digest(bank);spans=selected_memory_frames(model,frame,bank,len(frames))
                        expected=spatial_values(bank,spans)
                        if expected.shape != (28672,1,64):raise RuntimeError('Native full capacity changed')
                        features=capture_current_features(model,state,frame)
                        qstart=time.monotonic()
                        baseline=query(model,features,bank,frame,len(frames))
                        query_seconds={'native':time.monotonic()-qstart}
                        if baseline['output_sha256'] != target_hash:raise RuntimeError('Native query replay differs')
                        teachers={};source_rows=[];teacher_start=time.monotonic()
                        for span in spans:
                            source=span['frame']
                            if source in bank['cond_frame_outputs']:continue
                            visual=capture_current_features(model,state,source)
                            gt=load_label_mask(frames[source].mask_path,[1])==1
                            stored=bank['non_cond_frame_outputs'][source]
                            teachers[source]=same_frame_teachers(model,visual,stored,
                                torch.from_numpy(gt.copy())[None,None],masks[source])
                            source_rows.append({'local_source_frame':source,'native_reencoding_exact':True,
                                'source_prediction':score(stored,gt),
                                'native_presence_logit':teachers[source]['native_presence_logit'],
                                'gt_presence_logit':teachers[source]['gt_presence_logit']})
                        teacher_seconds=time.monotonic()-teacher_start
                        raw={'native':baseline}
                        for arm in ARMS[1:]:
                            qstart=time.monotonic()
                            raw[arm]=query(model,features,bank,frame,len(frames),expected,
                                spatial_values(bank,spans,teachers,arm))
                            query_seconds[arm]=time.monotonic()-qstart
                            if raw[arm]['keys'] != baseline['keys'] or raw[arm]['calls'] != baseline['calls']:
                                raise RuntimeError('Oracle changed native keys/positions/spans/pointers')
                        if raw['identity_values']['output_sha256'] != target_hash:raise RuntimeError('Identity parity differs')
                        after=query(model,features,bank,frame,len(frames))
                        if after['output_sha256'] != target_hash or _tree_digest(bank)!=bank_hash:
                            raise RuntimeError('Oracle changed native replay or its bank')
                        truth=load_label_mask(frames[frame].mask_path,[1])==1
                        row={**case,'present':bool(truth.any()),'identity_sha256':identity_hash,
                            'native_propagation_sha256':target_hash,'native_bank_sha256':bank_hash,
                            'spans':spans,'sources':source_rows,'arms':{},
                            'source_teacher_encoding_wall_seconds':teacher_seconds,
                            'query_wall_seconds_with_integrity_instrumentation':query_seconds,
                            'wall_seconds_case_including_last_native_step':time.monotonic()-t0,
                            'process_peak_rss_bytes':resource.getrusage(resource.RUSAGE_SELF).ru_maxrss,
                            'native_bank_unchanged':True,'native_after_oracle_exact':True}
                        for arm,result in raw.items():
                            row['arms'][arm]={**score(result['output'],truth),
                                **{k:v for k,v in result.items() if k!='output'}}
                        if _model_digest(model)!=core:raise RuntimeError('Pretrained weights changed')
                        _guard_sources(sources)
                        path=out/'cases'/f"{case['id']}.json";atomic_json(path,row)
                        atomic_json(path.with_name(f"{case['id']}.done.json"),
                            {'identity_sha256':identity_hash,'result_sha256':digest(path)})
                        existing[case['id']]=row
                        atomic_json(out/'progress.json',{'last_case':case['id'],
                            'completed_cases':len(all_cases)+sum(v is not None for v in existing.values()),
                            'planned_cases':len(plan['cases']),'training_started':False,'efficacy_authorized':False})
                        print(json.dumps({'case_completed':case['id']}),flush=True)
                finally:iterator.close()
                model.reset_state(state)
            if _model_digest(model)!=core:raise RuntimeError('Pretrained model changed')
            all_cases.extend(existing.values());del model
        if len(all_cases)!=len(plan['cases']) or any(x is None for x in all_cases):
            raise RuntimeError('Planned case coverage incomplete')
        _guard_sources(sources)
        results={'identity':identity,'case_ids':[c['id'] for c in all_cases],
            'summary':summarize(all_cases,design['seed']),
            'case_receipts':{c['id']:digest(out/'cases'/f"{c['id']}.json") for c in all_cases}}
        atomic_json(out/'results.json',results)
        atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),
            'scope':'Complete fixed28-case TRAIN oracle diagnostic; not a learned method',
            'efficacy_authorized':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--output-dir',required=True,type=Path);p.add_argument('--read-gate',required=True,type=Path)
    p.add_argument('--case-plan',required=True,type=Path);args=p.parse_args()
    try:run(args)
    except Exception as error:
        if args.output_dir.exists():
            atomic_json(args.output_dir/f'failure_{time.time_ns()}.json',{'type':type(error).__name__,'error':str(error)})
        raise
