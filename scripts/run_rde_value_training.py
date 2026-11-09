"""Fixed RDE value-repair training using project native trainer/checkpoint tools.

Engineering mode runs the same executor on fixed TRAIN prefixes, with a planned
interruption before a checkpoint commit. No validation or test images are opened.
"""
import argparse
import copy
import fcntl
import json
from pathlib import Path
import random
import sys
import tempfile
import time
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch
from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.complementary_memory.training import capture_current_features,clone_tensors
from modeling.rde_value_repair import MemoryValueRepair,author_consistency_loss
from scripts.mavos_value_utils import activate_author_repo,atomic_json,atomic_torch_save,prepare,target_for
from scripts.memory_token_io import digest,validate_output,prompt_index
from scripts.rde_training_state import load_checkpoint,publish_checkpoint,append_frame,verify_journal,next_sequence
from scripts.rde_value_training import training_step
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT,MANIFEST,_capture_rng,_restore_rng,_guard_sources,_source_hashes,_model_digest,_tree_digest,_sequence_source_hashes

DESIGN=ROOT/'protocols/rde_value_repair_efficacy_polypgen_v1.json'
ENGINEERING=ROOT/'protocols/rde_value_repair_runner_gate_v1.json'
ARMS={'segmentation_only':0,'segmentation_plus_consistency':10}


class ControlledInterruption(Exception):pass


def verify_receipt(directory):
    complete=json.loads((directory/'completion.json').read_text());result=directory/'results.json'
    if not complete['passed'] or complete['results_sha256']!=digest(result):raise RuntimeError(f'Invalid prerequisite: {directory}')
    return json.loads(result.read_text()),digest(result)


def frozen_sources():
    sources=_source_hashes()
    for n in ['scripts/run_rde_value_training.py','scripts/verify_rde_training_runner.py','scripts/rde_training_state.py','scripts/rde_value_training.py',
        'modeling/rde_value_repair.py','modeling/mavos_focal_value_modulation.py','modeling/complementary_memory/integration.py',
        'modeling/native_memory_layout.py','scripts/mask_memory_teacher.py','scripts/memory_value_query.py',
        'protocols/rde_value_repair_efficacy_polypgen_v1.json','protocols/rde_value_repair_runner_gate_v1.json',
        'protocols/rde_value_repair_polypgen_v1.json','protocols/rde_value_repair_autoregressive_gate_v1.json','tests/test_rde_training_state.py','tests/test_rde_training_runner.py','tests/test_rde_value_training.py']:
        sources[n]=digest(ROOT/n)
    for path in (ROOT/'external/sam2/sam2').rglob('*.py'):sources[str(path.relative_to(ROOT))]=digest(path)
    d=json.loads(DESIGN.read_text());config='external/sam2/sam2/'+d['model']['configuration'];sources[config]=digest(ROOT/config)
    return sources


def selected_frames(videos,design,engineering):
    fixture=json.loads(ENGINEERING.read_text())
    names=design['data']['train_sequences'] if not engineering else fixture['sequences']
    result={}
    for name in names:
        frames=videos[name]
        if engineering:
            offset=prompt_index(frames,{'initialization':'first_labeled_frame'});frames=frames[offset:offset+fixture['frames_from_first_prompt']]
            if len(frames)!=fixture['frames_from_first_prompt']:raise RuntimeError('Engineering prefix length changed')
        result[name]=frames
    return result


def train_arm(model,adapter,optimizer,frames_by_name,epochs,arm,weight,out,identity_sha,sources,criterion,
              interruption,invocation,initial_sha):
    directory=out/'training'/arm;directory.mkdir(parents=True,exist_ok=True);journal=directory/'frames'
    names=list(frames_by_name);state=load_checkpoint(directory,identity_sha)
    if state is None:
        if list(journal.glob('*.json')):raise RuntimeError('Frame journal exists without an initial checkpoint')
        state={'identity_sha256':identity_sha,'arm':arm,'initial_adapter_sha256':initial_sha,
            'cursor':{'epoch':0,'sequence_index':0,'next_frame':None},'frames_seen':0,'updates':0,'journal_tip':None,
            'adapter':copy.deepcopy(adapter.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
            'rng':_capture_rng(),'bank':None,'teacher_cache':{},'active_inputs':None}
        publish_checkpoint(directory,state)
    if state['arm']!=arm or state['initial_adapter_sha256']!=initial_sha:raise RuntimeError('Arm initialization identity mismatch')
    adapter.load_state_dict(state['adapter']);optimizer.load_state_dict(state['optimizer']);_restore_rng(state['rng'])
    verify_journal(journal,identity_sha,state['frames_seen'],state['journal_tip'])
    core=_model_digest(model)
    while state['cursor']['epoch']<epochs:
        cursor=state['cursor'];name=names[cursor['sequence_index']];frames=frames_by_name[name]
        inputs=_sequence_source_hashes(frames)
        with tempfile.TemporaryDirectory(dir=directory) as temp:
            inference,prompt=prepare(model,frames,Path(temp))
            with torch.inference_mode():model.propagate_in_video_preflight(inference)
            initial_bank=clone_tensors(inference['output_dict_per_obj'][0])
            if cursor['next_frame'] is None:
                state['bank']=initial_bank;state['teacher_cache']={};state['active_inputs']=inputs
                cursor={**cursor,'next_frame':prompt+1};state['cursor']=cursor
                state['adapter']=copy.deepcopy(adapter.state_dict());state['optimizer']=copy.deepcopy(optimizer.state_dict());state['rng']=_capture_rng()
                publish_checkpoint(directory,state)
            else:
                if state['active_inputs']!=inputs or not prompt+1<=cursor['next_frame']<=len(frames):raise RuntimeError('Resume sequence inputs or frame cursor differ')
                if _tree_digest(state['bank']['cond_frame_outputs'])!=_tree_digest(initial_bank['cond_frame_outputs']):raise RuntimeError('Prompt bank differs on resume')
                for source,entry in state['teacher_cache'].items():
                    if entry['gt_source_sha256']!=inputs[source]['mask_sha256']:raise RuntimeError('Cached teacher label source changed')
                # The staged image decoder is deterministic; initialization must
                # not consume training randomness across a process restart.
                _restore_rng(state['rng'])
            for frame in range(cursor['next_frame'],len(frames)):
                actual={'frame':frame,'image_sha256':digest(frames[frame].image_path),'mask_sha256':digest(frames[frame].mask_path)}
                if actual!=inputs[frame]:raise RuntimeError('Active frame/image/label changed')
                visual=capture_current_features(model,inference,frame)
                sample={'frame':frame,'num_frames':len(frames),'vision':visual['vision'],'positions':visual['positions'],'sizes':visual['sizes'],
                    'target':target_for(frames[frame])[0],'teacher_target':torch.from_numpy((load_label_mask(frames[frame].mask_path,[1])==1).copy())[None,None],
                    'teacher_target_sha256':actual['mask_sha256'],'object_id':1,'mask_inputs':None,'point_inputs':None}
                started=time.monotonic()
                bank,cache,row=training_step(model,adapter,optimizer,sample,state['bank'],state['teacher_cache'],criterion,weight)
                step=state['frames_seen']+1
                row={'epoch':cursor['epoch'],'sequence':name,'source_frame_index':frames[frame].frame_index,'input':actual,**row}
                tip=append_frame(journal,identity_sha,step,state['journal_tip'],row)
                invocation['frames']+=1
                if interruption and invocation['frames']==interruption:
                    raise ControlledInterruption('Planned gate interruption after frame receipt, before checkpoint publish')
                state={**state,'cursor':{**cursor,'next_frame':frame+1},'frames_seen':step,'journal_tip':tip,
                    'updates':state['updates']+int(row['optimizer_updated']),'bank':bank,'teacher_cache':cache,
                    'adapter':copy.deepcopy(adapter.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'rng':_capture_rng()}
                publish_checkpoint(directory,state)
                _guard_sources(sources)
                telemetry={'arm':arm,'epoch':cursor['epoch'],'sequence':name,'frame':frame,'frames_seen':step,'updates':state['updates'],
                    'step_wall_seconds':time.monotonic()-started,'loss':row['loss'],'segmentation_loss':row['segmentation_loss'],'consistency_loss':row['consistency_loss']}
                atomic_json(directory/'progress.json',telemetry);print(json.dumps(telemetry),flush=True)
            with torch.inference_mode():model.reset_state(inference)
        if _model_digest(model)!=core:raise RuntimeError('Frozen SAM2 changed during training sequence')
        state['cursor']=next_sequence(cursor['epoch'],cursor['sequence_index'],len(names),epochs)
        state['bank']=None;state['teacher_cache']={};state['active_inputs']=None;state['rng']=_capture_rng()
        publish_checkpoint(directory,state)
    expected=epochs*sum(len(frames)-prompt_index(frames,{'initialization':'first_labeled_frame'})-1 for frames in frames_by_name.values())
    if state['frames_seen']!=expected or verify_journal(journal,identity_sha,expected,state['journal_tip']):raise RuntimeError('Final training frame coverage differs')
    final=directory/'adapter_final.pt';atomic_torch_save(final,{'adapter':adapter.state_dict(),'identity_sha256':identity_sha,'arm':arm,'frames_seen':expected})
    result={'passed':True,'arm':arm,'frames_seen':expected,'updates':state['updates'],'journal_tip':state['journal_tip'],
        'initial_adapter_sha256':initial_sha,'final_adapter_tree_sha256':_tree_digest(adapter.state_dict()),'final_adapter_file_sha256':digest(final),
        'final_checkpoint_tree_sha256':_tree_digest(state),'sam2_parameters_unchanged':True,'scope':'Training only; no segmentation efficacy result'}
    atomic_json(directory/'results.json',result);atomic_json(directory/'completion.json',{'passed':True,'results_sha256':digest(directory/'results.json')})
    return result


def run(args,ownership):
    design=json.loads(DESIGN.read_text());sources=frozen_sources();gates={}
    for name in ['engineering_attempt_02','multi_object_attempt_01','autoregressive_gate_attempt_01','autoregressive_resume_attempt_01']:
        result,h=verify_receipt(args.gates_root/name);gates[name]=h
        identity_path=args.gates_root/name/'identity.json'
        if identity_path.exists():
            prior=json.loads(identity_path.read_text());_guard_sources(prior.get('sources',{}))
    if not args.engineering:
        if args.interrupt_before_checkpoint:raise RuntimeError('Interruption injection is engineering-only')
        if args.runner_gate is None or args.native_reference is None:raise RuntimeError('Full training requires runner gate and matched native reference')
        runner,h=verify_receipt(args.runner_gate)
        if not runner.get('runner_safety_passed') or runner['training_sources']!=sources:raise RuntimeError('Full runner safety/source identity differs')
        native,h2=verify_receipt(args.native_reference)
        if not native.get('reuse_this_matched_reference'):raise RuntimeError('Matched native reference is unavailable')
        native_identity=json.loads((args.native_reference/'identity.json').read_text())
        _guard_sources(native_identity['sources'])
        if (native_identity['threads']!=1 or native_identity['torch_version']!=torch.__version__
                or native_identity['checkpoint_sha256']!=digest(CHECKPOINT_DEFAULT)):
            raise RuntimeError('Native runtime/model mismatch')
        if set(native['sequences'])!=set(design['data']['validation_sequences']):
            raise RuntimeError('Native validation cohort differs')
        gates.update({'runner':h,'matched_native':h2})
    if args.seed!=design['training']['primary_seed']:raise RuntimeError('Replication seeds require the separately audited primary validation release; not implemented by this primary runner')
    if torch.__version__!=design['training']['torch_version'] or digest(CHECKPOINT_DEFAULT)!=design['model']['checkpoint_sha256'] or digest(MANIFEST)!=design['data']['manifest_sha256']:
        raise RuntimeError('Runtime, checkpoint or manifest differs')
    rde=Path(args.rde_repository)
    architecture=json.loads((ROOT/'protocols/rde_value_repair_polypgen_v1.json').read_text())
    if digest(rde/'model/model.py')!=architecture['prior_work']['source_sha256']:raise RuntimeError('RDE loss source changed')
    out=validate_output(args.output_dir);out.mkdir(parents=True,exist_ok=True)
    with (out/'writer.lock').open('a+') as lock:
        fcntl.flock(lock,fcntl.LOCK_EX|fcntl.LOCK_NB);ownership['acquired']=True
        torch.set_num_threads(1);torch.use_deterministic_algorithms(True)
        frames=selected_frames(load_manifest(MANIFEST,'dev'),design,args.engineering);epochs=json.loads(ENGINEERING.read_text())['epochs'] if args.engineering else design['training']['epochs']
        identity={'sources':sources,'gates':gates,'seed':args.seed,'epochs':epochs,'engineering':args.engineering,
            'input_files':{name:_sequence_source_hashes(f) for name,f in frames.items()},'torch_version':torch.__version__,
            'checkpoint_sha256':digest(CHECKPOINT_DEFAULT),'rde_source_sha256':digest(rde/'model/model.py')}
        if (out/'identity.json').exists():
            if json.loads((out/'identity.json').read_text())!=identity:raise RuntimeError('Run identity differs on resume')
        else:
            atomic_json(out/'identity.json',identity);atomic_json(out/'protocol_frozen.json',design)
            for n in sources:
                q=out/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/n).read_bytes())
        identity_sha=digest(out/'identity.json');repo,revision=activate_author_repo()
        if revision!=design['model']['sam2_revision']:raise RuntimeError('SAM2 revision differs')
        model=build_author_predictor(argparse.Namespace(method='sam2',device='cpu',checkpoint=CHECKPOINT_DEFAULT,model_config=design['model']['configuration']),repo)
        model.requires_grad_(False).eval();criterion=author_consistency_loss(rde);invocation={'frames':0};arms={};initial_sha=None
        for name,weight in ARMS.items():
            random.seed(args.seed);np.random.seed(args.seed);torch.manual_seed(args.seed)
            adapter=MemoryValueRepair();current_initial=_tree_digest(adapter.state_dict())
            if initial_sha is not None and initial_sha!=current_initial:raise RuntimeError('Arms do not share exact initialization')
            initial_sha=current_initial
            cfg=design['training']['optimizer']
            if cfg['gradient_clip_norm']!=3:raise RuntimeError('Trainer clipping differs from the frozen protocol')
            optimizer=torch.optim.AdamW(adapter.parameters(),lr=cfg['lr'],weight_decay=cfg['weight_decay'],betas=tuple(cfg['betas']),eps=cfg['eps'])
            arms[name]=train_arm(model,adapter,optimizer,frames,epochs,name,weight,out,identity_sha,sources,criterion,
                args.interrupt_before_checkpoint,invocation,initial_sha)
        _guard_sources(sources)
        atomic_json(out/'results.json',{'passed':True,'scope':'Engineering training fixture' if args.engineering else 'Fixed primary training completed; evaluation pending',
            'arms':arms,'engineering':args.engineering,'candidate_efficacy_result_available':False})
        atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),'candidate_efficacy_result_available':False})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--gates-root',type=Path,required=True);p.add_argument('--rde-repository',type=Path,required=True)
    p.add_argument('--seed',type=int,default=20261109);p.add_argument('--engineering',action='store_true');p.add_argument('--interrupt-before-checkpoint',type=int,default=0);p.add_argument('--runner-gate',type=Path);p.add_argument('--native-reference',type=Path);args=p.parse_args();ownership={}
    try:run(args,ownership)
    except ControlledInterruption as e:print(str(e),flush=True);sys.exit(75)
    except Exception as e:
        if ownership.get('acquired'):atomic_json(args.output_dir/f'failure_{time.time_ns()}.json',{'error':str(e),'type':type(e).__name__})
        raise
