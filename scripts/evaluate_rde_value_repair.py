"""Primary value-repair validation; strict fixed cohort, no test-set entry point."""
import argparse
import fcntl
import json
from pathlib import Path
import sys
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from datasets.video import load_manifest
from scripts.mavos_value_utils import activate_author_repo,atomic_json
from scripts.memory_token_io import digest,validate_output,prompt_index
from scripts.rde_value_evaluation import evaluate_condition,load_adapter
from scripts.rde_value_results import summarize,ARMS
from scripts.run_author_baseline import build_author_predictor
from scripts.run_rde_value_training import frozen_sources,verify_receipt
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT,MANIFEST,_guard_sources,_model_digest,_validate_cached_evaluation

DESIGN=ROOT/'protocols/rde_value_repair_efficacy_polypgen_v1.json'


def evaluation_sources():
    sources=frozen_sources()
    for n in ['scripts/evaluate_rde_value_repair.py','scripts/rde_value_evaluation.py','scripts/rde_value_results.py',
        'scripts/verify_rde_value_evaluation.py','tests/test_rde_value_evaluation.py','tests/test_rde_value_results.py',
        'protocols/rde_value_repair_evaluation_gate_v1.json','scripts/run_mavos_value_contrastive.py']:
        sources[n]=digest(ROOT/n)
    return sources


def report(result,path):
    names=list(result['per_sequence']['native']);means=result['equal_sequence_means']
    fmt=lambda v:'N/A' if v is None else f'{v:.6f}'
    lines=['# Value-repair primary validation','','Exploratory PolypGen seq13–15. Fixed final epoch3; no epoch or seed selection. Historical test not opened.','','| Condition | Present Dice | IoU | Boundary F | Equal-sequence false-alarm rate |','|---|---:|---:|---:|---:|']
    for a in ARMS:lines.append('| '+a+' | '+' | '.join(fmt(means[a][m]) for m in ['present_dice','present_iou','present_boundary_f','gt_empty_false_alarm_rate'])+' |')
    lines+=['','| Sequence | Native | Segmentation-only | Segmentation + consistency |','|---|---:|---:|---:|']
    for n in names:lines.append('| '+n+' | '+' | '.join(fmt(result['per_sequence'][a][n]['present_dice']) for a in ARMS)+' |')
    lines+=['','| C2 contrast | Mean Δ Dice | 95% sequence-bootstrap CI | Wins / ties / losses | Holm p |','|---|---:|---|---:|---:|']
    for label,c in result['primary_contrasts_C2_minus'].items():
        ci=c['ci95'];lines.append(f"| C2 − {label} | {fmt(c['mean_delta'])} | {ci} | {c['wins']} / {c['ties']} / {c['losses']} | {fmt(c['holm_p'])} |")
    lines+=['','| Condition | Re-detected / return events | Missed returns | False alarms / empty frames | Instrumented FPS |','|---|---:|---:|---:|---:|']
    for a,t in result['temporal_and_compute'].items():lines.append(f"| {a} | {t['detected_returns']}/{t['return_events']} | {t['missed_returns']} | {t['gt_empty_false_alarms']}/{t['gt_empty_frames']} | {fmt(t['instrumented_fps'])} |")
    screen=result['validation_screen'];lines+=['',f"Predeclared primary screen: **{'PASS' if screen['passed'] else 'FAIL'}**.",
        'A pass permits the fixed replication stage only; it is not independent confirmation or a publication claim. A failure stops this candidate without seed search or historical-test evaluation.',
        '',result['limits'],'','Per-sequence inputs, masks, layer reads, complete contrasts and leave-one-sequence-out sensitivity are in the JSON/sequence receipts.']
    Path(path).write_text('\n'.join(lines)+'\n')


def run(args,ownership):
    design=json.loads(DESIGN.read_text());sources=evaluation_sources()
    training,training_sha=verify_receipt(args.training_root)
    if training.get('engineering') is not False:raise RuntimeError('Engineering fixture weights cannot be used as efficacy weights')
    train_identity=json.loads((args.training_root/'identity.json').read_text());_guard_sources(train_identity['sources'])
    if train_identity['epochs']!=3 or train_identity['seed']!=design['training']['primary_seed']:
        raise RuntimeError('Only the fixed primary final-epoch checkpoint is accepted')
    gate,gate_sha=verify_receipt(args.evaluation_gate)
    if not gate.get('evaluation_safety_passed') or gate['evaluation_sources']!=sources:raise RuntimeError('Evaluation gate/source identity differs')
    native,native_sha=verify_receipt(args.native_reference);native_identity=json.loads((args.native_reference/'identity.json').read_text())
    _guard_sources(native_identity['sources'])
    if not native.get('reuse_this_matched_reference') or native_identity['threads']!=1 or native_identity['torch_version']!=torch.__version__:
        raise RuntimeError('Matched native reference is unavailable')
    if digest(CHECKPOINT_DEFAULT)!=design['model']['checkpoint_sha256'] or digest(MANIFEST)!=design['data']['manifest_sha256'] or torch.__version__!=design['training']['torch_version']:
        raise RuntimeError('Frozen model/data/runtime mismatch')
    if native_identity['checkpoint_sha256']!=digest(CHECKPOINT_DEFAULT):raise RuntimeError('Native checkpoint differs')
    out=validate_output(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/'writer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ownership['acquired']=True
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        checkpoint_hashes={a:digest(args.training_root/'training'/a/'adapter_final.pt') for a in ARMS[1:]}
        for a,h in checkpoint_hashes.items():
            arm_result,_=verify_receipt(args.training_root/'training'/a)
            if h!=arm_result['final_adapter_file_sha256'] or arm_result!=training['arms'][a]:raise RuntimeError('Final trained arm checkpoint/receipt differs')
        identity={'sources':sources,'training_results_sha256':training_sha,'training_identity_sha256':digest(args.training_root/'identity.json'),
            'native_results_sha256':native_sha,'evaluation_gate_sha256':gate_sha,'final_checkpoints':checkpoint_hashes,'protocol_sha256':digest(DESIGN)}
        if (out/'identity.json').exists():
            if json.loads((out/'identity.json').read_text())!=identity:raise RuntimeError('Validation resume identity differs')
        else:
            atomic_json(out/'identity.json',identity);atomic_json(out/'protocol_frozen.json',design)
            for n in sources:
                q=out/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/n).read_bytes())
        setup_hash=digest(out/'identity.json');repo,revision=activate_author_repo()
        if revision!=design['model']['sam2_revision']:raise RuntimeError('SAM2 source revision differs')
        model=build_author_predictor(argparse.Namespace(method='sam2',device='cpu',checkpoint=CHECKPOINT_DEFAULT,model_config=design['model']['configuration']),repo)
        model.requires_grad_(False).eval();model_hash=_model_digest(model);videos=load_manifest(MANIFEST,'dev');groups={a:{} for a in ARMS};names=design['data']['validation_sequences']
        if set(native['sequences'])!=set(names):raise RuntimeError('Native validation cohort differs')
        for name in names:
            directory=args.native_reference/'evaluation'/name
            prior=native['sequences'][name]
            if digest(directory/'summary.json')!=prior['summary_sha256'] or digest(directory/'frames.jsonl')!=prior['frames_sha256']:raise RuntimeError('Matched native sequence changed')
            groups['native'][name]=_validate_cached_evaluation(directory,videos[name],prompt_index(videos[name],{'initialization':'first_labeled_frame'}),'native',model_hash,digest(args.native_reference/'identity.json'))
        for arm in ARMS[1:]:
            adapter=load_adapter(args.training_root/'training'/arm/'adapter_final.pt',identity['training_identity_sha256'],arm)
            for name in names:
                groups[arm][name]=evaluate_condition(model,videos[name],adapter,name,out/arm,setup_hash)
                _guard_sources(sources);atomic_json(out/'progress.json',{'arm':arm,'completed_sequence':name,'historical_test_started':False})
                print(json.dumps({'validation_completed':arm,'sequence':name}),flush=True)
        result=summarize(groups,names,design['training']['primary_seed'],True)
        result.update({'technical_evaluation_passed':True,'historical_test_started':False,'training_seed':train_identity['seed'],
            'identity_sha256':setup_hash,'primary_release_for_replication':result['validation_screen']['passed']})
        _guard_sources(sources);atomic_json(out/'results.json',result);report(result,out/'results.md')
        atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),
            'validation_screen_passed':result['validation_screen']['passed'],'historical_test_started':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--training-root',type=Path,required=True);p.add_argument('--native-reference',type=Path,required=True);p.add_argument('--evaluation-gate',type=Path,required=True);p.add_argument('--output-dir',type=Path,required=True);a=p.parse_args();ownership={}
    try:run(a,ownership)
    except Exception as e:
        if ownership.get('acquired'):atomic_json(a.output_dir/f'failure_{time.time_ns()}.json',{'error':str(e),'type':type(e).__name__})
        raise
