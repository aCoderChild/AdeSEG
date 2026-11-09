"""Fixed TRAIN own-rollout/teacher-association gate, reusing native SAM2 helpers."""
import argparse
import copy
import json
from pathlib import Path
import sys
import tempfile
ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import torch
from datasets.io import load_label_mask
from datasets.video import load_manifest
from modeling.complementary_memory.training import capture_current_features,clone_tensors
from modeling.native_memory_layout import selected_memory_frames
from modeling.rde_value_repair import MemoryValueRepair,author_consistency_loss
from scripts.mask_memory_teacher import capture_native_mask_inputs,same_frame_teachers
from scripts.memory_value_query import MEMORY_FIELDS,query
from scripts.mavos_value_utils import activate_author_repo,atomic_json,atomic_torch_save,prepare,target_for
from scripts.memory_token_io import digest,validate_output
from scripts.rde_value_training import training_step
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT,MANIFEST,PROTOCOL,_guard_sources,_model_digest,_tree_digest
from scripts.verify_token_conditioned_key_adapter import require_tree_equal

DESIGN=ROOT/'protocols/rde_value_repair_autoregressive_gate_v1.json'


def optimizer_for(adapter):
    return torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=1e-3)


def replay(model,adapter,optimizer,samples,bank,cache,criterion,weight):
    rows=[]
    for sample in samples:
        bank,cache,row=training_step(model,adapter,optimizer,sample,bank,cache,criterion,weight)
        rows.append(row)
    return bank,cache,rows


def run(out,reference,case_plan,rde_repository):
    prior=json.loads((reference/'completion.json').read_text());prior_path=reference/'results.json'
    if not prior['passed'] or prior['results_sha256']!=digest(prior_path):raise RuntimeError('Bounded engineering reference differs')
    identity=json.loads((reference/'identity.json').read_text());_guard_sources(identity['sources'])
    out=validate_output(out);out.mkdir(parents=True,exist_ok=False)
    sources={n:digest(ROOT/n) for n in ['scripts/rde_value_training.py','scripts/verify_rde_autoregressive_training.py',
        'tests/test_rde_value_training.py','protocols/rde_value_repair_autoregressive_gate_v1.json']}
    sources={**identity['sources'],**sources}
    own={'sources':sources,'reference_results_sha256':digest(prior_path),'case_plan_sha256':digest(case_plan),
        'checkpoint_sha256':digest(CHECKPOINT_DEFAULT),'rde_source_sha256':digest(rde_repository/'model/model.py'),
        'threads':1,'torch_version':torch.__version__}
    if own['rde_source_sha256']!=identity['rde_source_sha256'] or own['checkpoint_sha256']!=identity['checkpoint_sha256']:raise RuntimeError('Author source or checkpoint differs')
    atomic_json(out/'identity.json',own)
    for n in sources:
        q=out/'source'/n;q.parent.mkdir(parents=True,exist_ok=True);q.write_bytes((ROOT/n).read_bytes())
    design=json.loads(DESIGN.read_text());atomic_json(out/'protocol_frozen.json',design)
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(design['fixtures']['seed'])
    base=json.loads(PROTOCOL.read_text());plan=json.loads(case_plan.read_text());videos=load_manifest(MANIFEST,'dev');repo,revision=activate_author_repo()
    if revision!=base['model']['sam2_revision']:raise RuntimeError('SAM2 revision differs')
    criterion=author_consistency_loss(rde_repository);results={'scope':design['scope'],'efficacy_authorized':False,'sequences':{}}
    for name in design['fixtures']['sequences']:
        offset=plan['sequences'][name]['prompt_manifest_offset'];frames=videos[name][offset:offset+11]
        model=build_author_predictor(argparse.Namespace(method='sam2',device='cpu',checkpoint=CHECKPOINT_DEFAULT,model_config=base['model']['configuration']),repo)
        model.requires_grad_(False).eval();core=_model_digest(model)
        with tempfile.TemporaryDirectory(dir=out) as temp:
            state,prompt=prepare(model,frames,Path(temp));masks={};bank=None
            with capture_native_mask_inputs(model) as capture,torch.inference_mode():
                iterator=model.propagate_in_video(state,start_frame_idx=0)
                try:
                    for frame,ids,_ in iterator:
                        if ids!=[1] or capture['count']!=frame+1:raise RuntimeError('Native source capture ID/frame mismatch')
                        masks[frame]=clone_tensors(capture['latest']);current=state['output_dict_per_obj'][0]
                        if frame==7:bank=clone_tensors(current)
                        if frame==8:break
                finally:iterator.close()
            if bank is None or 8 in bank['non_cond_frame_outputs']:raise RuntimeError('Warm-up bank scope differs')
            samples=[]
            for frame in [8,9,10]:
                f=capture_current_features(model,state,frame);gt=torch.from_numpy((load_label_mask(frames[frame].mask_path,[1])==1).copy())[None,None]
                samples.append({'frame':frame,'num_frames':11,'vision':f['vision'],'positions':f['positions'],'sizes':f['sizes'],
                    'target':target_for(frames[frame])[0],'teacher_target':gt,'teacher_target_sha256':digest(frames[frame].mask_path),
                    'object_id':1,'mask_inputs':None,'point_inputs':None})
            spans=selected_memory_frames(model,8,bank,11);cache={}
            for span in spans:
                frame=span['frame']
                if frame in bank['cond_frame_outputs']:continue
                features=capture_current_features(model,state,frame);stored=bank['non_cond_frame_outputs'][frame]
                gt=torch.from_numpy((load_label_mask(frames[frame].mask_path,[1])==1).copy())[None,None]
                t=same_frame_teachers(model,features,stored,gt,masks[frame])
                cache[frame]={'frame':frame,'object_id':1,'native_memory_sha256':_tree_digest(stored['maskmem_features']),
                    'native_score_sha256':_tree_digest(stored['object_score_logits']),'value':t['mask_only'],
                    'gt_source_sha256':digest(frames[frame].mask_path)}
            baseline=query(model,{'vision':samples[0]['vision'],'positions':samples[0]['positions'],'sizes':samples[0]['sizes']},bank,8,11)
            initial=MemoryValueRepair().state_dict();bank_hash=_tree_digest(bank);cache_hash=_tree_digest(cache);arms={}
            for arm,weight in design['updates']['arms'].items():
                adapter=MemoryValueRepair();adapter.load_state_dict(initial);optimizer=optimizer_for(adapter)
                b,c,row=training_step(model,adapter,optimizer,samples[0],bank,cache,criterion,weight)
                if row['prediction_sha256']!=baseline['output_sha256']:raise RuntimeError('First own write differs from native zero-residual prediction')
                checkpoint={'adapter':copy.deepcopy(adapter.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
                    'rng':torch.random.get_rng_state().clone(),'bank':clone_tensors(b),'teacher_cache':clone_tensors(c),
                    'samples':clone_tensors(samples[1:]),'weight':weight}
                path=out/f'{name}_{arm}_after8.pt';atomic_torch_save(path,checkpoint)
                final_bank,final_cache,rows=replay(model,adapter,optimizer,samples[1:],b,c,criterion,weight)
                expected={'adapter':copy.deepcopy(adapter.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
                    'bank':clone_tensors(final_bank),'teacher_cache':clone_tensors(final_cache),'records':rows,'rng':torch.random.get_rng_state().clone()}
                # Preserve complete expected state for later independent process comparison.
                final_path=out/f'{name}_{arm}_expected.pt';atomic_torch_save(final_path,expected)
                restored=torch.load(path,weights_only=True,map_location='cpu');adapter.load_state_dict(restored['adapter']);optimizer.load_state_dict(restored['optimizer']);torch.random.set_rng_state(restored['rng'])
                b2,c2,rows2=replay(model,adapter,optimizer,restored['samples'],restored['bank'],restored['teacher_cache'],criterion,weight)
                require_tree_equal(expected,{'adapter':adapter.state_dict(),'optimizer':optimizer.state_dict(),'bank':b2,'teacher_cache':c2,'records':rows2,'rng':torch.random.get_rng_state()})
                arms[arm]={'native_zero_first_write_exact':True,'own_rollout_checkpoint_replay_exact':True,'records':[row,*rows],
                    'checkpoint_sha256':digest(path),'expected_state_sha256':digest(final_path),'expected_tree_sha256':_tree_digest(expected)}
            # Alter only auxiliary teacher targets, leaving current segmentation
            # supervision fixed. The current student write must remain identical.
            adapter=MemoryValueRepair();adapter.load_state_dict(initial);optimizer=optimizer_for(adapter)
            altered={f:{**entry,'value':torch.zeros_like(entry['value'])} for f,entry in cache.items()}
            changed={**samples[0],'teacher_target':~samples[0]['teacher_target']}
            _,_,leak=training_step(model,adapter,optimizer,changed,bank,altered,criterion,0)
            if leak['prediction_sha256']!=baseline['output_sha256']:raise RuntimeError('Teacher targets leaked into the current student forward')
            # The first postprompt read has only conditioning values, so no
            # adapter update is permitted; its predicted source still seeds a
            # detached teacher for the next frame.
            first_features=capture_current_features(model,state,1)
            first={**samples[0],'frame':1,'vision':first_features['vision'],'positions':first_features['positions'],
                'sizes':first_features['sizes'],'target':target_for(frames[1])[0],
                'teacher_target':torch.from_numpy((load_label_mask(frames[1].mask_path,[1])==1).copy())[None,None],
                'teacher_target_sha256':digest(frames[1].mask_path)}
            cold=MemoryValueRepair();cold_optimizer=optimizer_for(cold)
            _,cold_cache,cold_row=training_step(model,cold,cold_optimizer,first,
                {'cond_frame_outputs':clone_tensors(bank['cond_frame_outputs']),'non_cond_frame_outputs':{}},{},criterion,10)
            if cold_row['optimizer_updated'] or cold_optimizer.state or set(cold_cache)!={1}:
                raise RuntimeError('Conditioning-only first read updated adapter or omitted next-source teacher')
            if _tree_digest(bank)!=bank_hash or _tree_digest(cache)!=cache_hash or _model_digest(model)!=core:raise RuntimeError('Fixture/native state mutated')
            results['sequences'][name]={'arms':arms,'teacher_target_noninterference':True,'conditioning_only_read_no_update':True,'native_bank_unchanged':True,'sam2_parameters_unchanged':True,
                'input_files':[{'image':digest(f.image_path),'mask':digest(f.mask_path)} for f in frames]}
            model.reset_state(state)
        _guard_sources(sources);atomic_json(out/'progress.json',results);print(json.dumps({'checked':name}),flush=True)
    atomic_json(out/'results.json',results);atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),
        'efficacy_authorized':False,'pending':['fresh-process own-rollout resume','full frozen efficacy protocol and runner verification']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--reference',type=Path,required=True);p.add_argument('--case-plan',type=Path,required=True);p.add_argument('--rde-repository',type=Path,required=True);a=p.parse_args()
    try:run(a.output_dir,a.reference,a.case_plan,a.rde_repository)
    except Exception as e:
        if a.output_dir.exists():atomic_json(a.output_dir/'failure.json',{'error':str(e),'type':type(e).__name__})
        raise
