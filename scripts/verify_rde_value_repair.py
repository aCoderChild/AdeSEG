"""Bounded TRAIN value-repair parity/gradient/checkpoint gate; no Dice scoring."""
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
from modeling.complementary_memory.integration import NativeComplementaryRead
from modeling.complementary_memory.training import capture_current_features,clone_tensors,reader_clip_loss
from modeling.mavos_focal_value_modulation import MAVOSMemoryValueModulation
from modeling.native_memory_layout import selected_memory_frames
from modeling.rde_value_repair import MemoryValueRepair,author_consistency_loss,value_consistency,bind_native_reads
from scripts.mask_memory_teacher import capture_native_mask_inputs,same_frame_teachers
from scripts.memory_value_query import MEMORY_FIELDS,query,spatial_values
from scripts.mavos_value_utils import activate_author_repo,atomic_json,atomic_torch_save,prepare,target_for
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_token_conditioned_key_adapter import CHECKPOINT_DEFAULT,MANIFEST,PROTOCOL,_guard_sources,_model_digest,_tree_digest
from scripts.verify_token_conditioned_key_adapter import require_tree_equal

DESIGN=ROOT/'protocols/rde_value_repair_polypgen_v1.json'


def update(model,adapter,optimizer,features,bank,target,teacher,baseline_keys,criterion,weight):
    optimizer.zero_grad(set_to_none=True)
    keys=[];handles=[]
    try:
        for i,layer in enumerate(model.memory_attention.layers):
            def observe(module,args,i=i):keys.append({'layer':i,'key_input_sha256':_tree_digest(args[0])})
            handles.append(layer.cross_attn_image.k_proj.register_forward_pre_hook(observe))
        observer=NativeComplementaryRead(model,None,trace_only=True)
        hook=MAVOSMemoryValueModulation(model,[adapter]*4,record_value_receipts=True,capture_adapted_values=True)
        records=[]
        def auxiliary(sample):
            if len(hook.adapted_values)!=4:raise RuntimeError('Missing four-layer adapted values')
            terms=[]
            for item in hook.adapted_values:
                student=item['adapted'][:teacher.shape[0]]
                if not torch.equal(student[:4096],teacher[:4096]):raise RuntimeError('Conditioning values changed')
                terms.append(value_consistency(student,teacher,adapter._repair_indices,criterion))
            value=torch.stack(terms).mean()
            records.append(float(value.detach()))
            return weight*value
        sample={'frame':8,'num_frames':9,'vision':features['vision'],'positions':features['positions'],
                'sizes':features['sizes'],'target':target,'mask_inputs':None,'point_inputs':None}
        with bind_native_reads(model,adapter),observer,hook:
            loss,_,writes=reader_clip_loss(model,[sample],bank,auxiliary=auxiliary)
        if keys!=baseline_keys:raise RuntimeError('Native keys/positions changed')
        if len(observer.calls)!=1 or observer.calls[0]['layers']!=[0,1,2,3] or observer.calls[0]['spatial_keys']!=28672:
            raise RuntimeError('Native full-capacity read differs')
        if len(hook.value_calls)!=4 or not all(c['pointer_tail_bit_identical'] for c in hook.value_calls):
            raise RuntimeError('Pointer values or four-layer coverage changed')
        loss.backward()
        gradients={n:float(p.grad.abs().sum()) for n,p in adapter.named_parameters() if p.grad is not None}
        if any(not torch.isfinite(p.grad).all() for p in adapter.parameters() if p.grad is not None) or gradients.get('mlp.layers.1.weight',0)<=0:
            raise RuntimeError('Missing/nonfinite adapter gradient')
        if any(p.grad is not None for p in model.parameters()):raise RuntimeError('SAM2 received gradients')
        torch.nn.utils.clip_grad_norm_(adapter.parameters(),3)
        optimizer.step()
        return {'loss':float(loss.detach()),'kl':records,'gradient_l1':gradients,'keys':keys,
                'reads':observer.calls,'values':hook.value_calls,'writes':writes}
    finally:
        for handle in handles:handle.remove()


def run(out,plan_path,rde_repository,headroom_root):
    audit_path=headroom_root/'receipt_audit_complete_01/audit_results.json'
    audit=json.loads(audit_path.read_text())
    results_path=headroom_root/'train_diagnostic_attempt_01/results.json'
    if not audit['passed'] or not audit['complete_coverage_and_summary_match'] or not audit['primary_headroom_screen_passed'] or audit['final_results_sha256']!=digest(results_path):
        raise RuntimeError('Completed audited headroom prerequisite differs')
    out=validate_output(out);out.mkdir(parents=True,exist_ok=False)
    design=json.loads(DESIGN.read_text());base=json.loads(PROTOCOL.read_text());plan=json.loads(plan_path.read_text())
    sources={n:digest(ROOT/n) for n in ['protocols/rde_value_repair_polypgen_v1.json','modeling/rde_value_repair.py',
        'scripts/verify_rde_value_repair.py','modeling/mavos_focal_value_modulation.py',
        'modeling/complementary_memory/training.py','modeling/complementary_memory/integration.py',
        'scripts/mask_memory_teacher.py','scripts/memory_value_query.py','modeling/native_memory_layout.py',
        'external/sam2/sam2/modeling/sam2_utils.py','tests/test_rde_value_repair.py','data/polypgen_sequence.jsonl']}
    if digest(Path(rde_repository)/'model/model.py')!=design['prior_work']['source_sha256']:raise RuntimeError('RDE source hash differs')
    identity={'sources':sources,'checkpoint_sha256':digest(CHECKPOINT_DEFAULT),'case_plan_sha256':digest(plan_path),
              'headroom_audit_sha256':digest(audit_path),
              'rde_source_sha256':design['prior_work']['source_sha256'],'torch_version':torch.__version__,'threads':1}
    if identity['checkpoint_sha256']!=base['model']['checkpoint_sha256']:raise RuntimeError('Checkpoint changed')
    atomic_json(out/'identity.json',identity);atomic_json(out/'protocol_frozen.json',design)
    for n in sources:
        dst=out/'source'/n;dst.parent.mkdir(parents=True,exist_ok=True);dst.write_bytes((ROOT/n).read_bytes())
    torch.set_num_threads(1);torch.use_deterministic_algorithms(True);torch.manual_seed(20261109)
    repo,revision=activate_author_repo()
    if revision!=base['model']['sam2_revision']:raise RuntimeError('SAM2 revision differs')
    videos=load_manifest(MANIFEST,'dev');results={'scope':'Bounded PolypGen engineering only','efficacy_authorized':False,
        'multi_object_gate_passed':False,'fresh_process_resume_passed':False,'sequences':{}}
    criterion=author_consistency_loss(rde_repository)
    for name in ['seq2','seq3']:
        offset=plan['sequences'][name]['prompt_manifest_offset'];frames=videos[name][offset:offset+9]
        model=build_author_predictor(argparse.Namespace(method='sam2',device='cpu',checkpoint=CHECKPOINT_DEFAULT,
            model_config=base['model']['configuration']),repo);model.requires_grad_(False).eval();core=_model_digest(model)
        with tempfile.TemporaryDirectory(dir=out) as temp:
            state,prompt=prepare(model,frames,Path(temp));masks={};bank=None
            with capture_native_mask_inputs(model) as capture,torch.inference_mode():
                iterator=model.propagate_in_video(state,start_frame_idx=0)
                try:
                    for frame,ids,_ in iterator:
                        if ids!=[1] or capture['count']!=frame+1:raise RuntimeError('Native source/ID association differs')
                        masks[frame]=clone_tensors(capture['latest'])
                        current=state['output_dict_per_obj'][0]
                        if frame==7:bank=clone_tensors(current)
                        if frame==8:native_target=clone_tensors({k:current['non_cond_frame_outputs'][8][k] for k in MEMORY_FIELDS})
                finally:iterator.close()
            bank_hash=_tree_digest(bank);features=capture_current_features(model,state,8);spans=selected_memory_frames(model,8,bank,9)
            baseline=query(model,features,bank,8,9)
            if baseline['output_sha256']!=_tree_digest(native_target):raise RuntimeError('Native replay differs from propagation')
            teachers={}
            for span in spans:
                f=span['frame']
                if f in bank['cond_frame_outputs']:continue
                visual=capture_current_features(model,state,f)
                gt=torch.from_numpy((load_label_mask(frames[f].mask_path,[1])==1).copy())[None,None]
                teachers[f]=same_frame_teachers(model,visual,bank['non_cond_frame_outputs'][f],gt,masks[f])
            teacher=spatial_values(bank,spans,teachers,'mask_only');adapter=MemoryValueRepair();initial=copy.deepcopy(adapter.state_dict())
            with bind_native_reads(model,adapter),MAVOSMemoryValueModulation(model,[adapter]*4,record_value_receipts=True) as hook:
                zero=query(model,features,bank,8,9)
            if zero['output_sha256']!=baseline['output_sha256'] or zero['keys']!=baseline['keys']:
                raise RuntimeError('Zero adapter native parity failed')
            if len(hook.value_calls)!=4 or any(not v['pointer_tail_bit_identical'] for v in hook.value_calls):raise RuntimeError('Zero read pointer coverage failed')
            arms={}
            for arm,weight in [('segmentation_only',0),('segmentation_plus_consistency',10)]:
                adapter=MemoryValueRepair();adapter.load_state_dict(initial)
                optimizer=torch.optim.AdamW(adapter.parameters(),lr=1e-4,weight_decay=1e-3)
                records=[];checkpoint=None
                for step in range(4):
                    records.append(update(model,adapter,optimizer,features,bank,target_for(frames[8])[0],teacher,baseline['keys'],criterion,weight))
                    if step==1:
                        checkpoint={'adapter':copy.deepcopy(adapter.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),
                            'rng':torch.random.get_rng_state().clone(),'bank':clone_tensors(bank),'features':features,'teacher':teacher,
                            'target':target_for(frames[8])[0],'baseline_keys':baseline['keys'],'weight':weight}
                        atomic_torch_save(out/f'{name}_{arm}_resume.pt',checkpoint)
                expected={'adapter':copy.deepcopy(adapter.state_dict()),'optimizer':copy.deepcopy(optimizer.state_dict()),'records':records[2:]}
                restored=torch.load(out/f'{name}_{arm}_resume.pt',weights_only=True,map_location='cpu')
                require_tree_equal(restored['bank'],bank)
                adapter.load_state_dict(restored['adapter']);optimizer.load_state_dict(restored['optimizer']);torch.random.set_rng_state(restored['rng'])
                replay=[update(model,adapter,optimizer,features,restored['bank'],restored['target'],restored['teacher'],baseline['keys'],criterion,weight) for _ in range(2)]
                require_tree_equal(expected,{'adapter':adapter.state_dict(),'optimizer':optimizer.state_dict(),'records':replay})
                if _tree_digest(bank)!=bank_hash or _model_digest(model)!=core:raise RuntimeError('Native bank or SAM2 parameters changed')
                if any(records[-1]['gradient_l1'].get(n,0)<=0 for n,_ in adapter.named_parameters()):raise RuntimeError('MLP input/output parameters did not all acquire gradients')
                arms[arm]={'records':records,'checkpoint_restore_replay_exact':True,'trainable_parameters':sum(p.numel() for p in adapter.parameters()),'checkpoint_sha256':digest(out/f'{name}_{arm}_resume.pt')}
            results['sequences'][name]={'spans':spans,'native_zero_parity_exact':True,'native_bank_unchanged':True,
                'sam2_parameters_unchanged':True,'arms':arms,'input_files':[{'image':digest(f.image_path),'mask':digest(f.mask_path)} for f in frames]}
            model.reset_state(state)
        _guard_sources(sources);atomic_json(out/'progress.json',results)
        print(json.dumps({'checked':name}),flush=True)
    atomic_json(out/'results.json',results);atomic_json(out/'completion.json',{'passed':True,'results_sha256':digest(out/'results.json'),
        'scope':results['scope'],'efficacy_authorized':False,'pending':['persistent multi-object gate','fresh-process optimizer resume','full efficacy protocol']})


if __name__=='__main__':
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output-dir',type=Path,required=True);p.add_argument('--case-plan',type=Path,required=True);p.add_argument('--rde-repository',type=Path,required=True);p.add_argument('--headroom-root',type=Path,required=True);a=p.parse_args()
    try:run(a.output_dir,a.case_plan,a.rde_repository,a.headroom_root)
    except Exception as e:
        if a.output_dir.exists():atomic_json(a.output_dir/'failure.json',{'error':str(e),'type':type(e).__name__})
        raise
