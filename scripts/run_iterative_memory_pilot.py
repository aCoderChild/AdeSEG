"""Run174: frozen protocol -> real-frame engineering -> validated clip pilot.

Reuses official Cutie blocks, SAM2's author clip sampler and the existing
autoregressive training/propagation/metric code. Only experiment orchestration
and the SAM2-to-Cutie bridge are project adaptations.
"""
import argparse
import copy
import gc
import hashlib
import json
import os
from pathlib import Path
import random
import subprocess
import sys
import time
import zipfile

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
import numpy as np
import torch

from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from evaluation.statistics import sequence_bootstrap_interval, paired_wilcoxon, holm_adjust
from modeling.complementary_memory import ComplementaryMemoryReader
from modeling.complementary_memory.iterative_reader import IterativeMemoryReader
from scripts import run_complementary_memory_pilot as base
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.verify_complementary_memory_pilot import rollout, compare_exact
from scripts.verify_complementary_memory import multi_object_gate

PROTOCOL = 'protocols/iterative_memory_pilot_v1.json'
OWN_SOURCES = [PROTOCOL,'modeling/complementary_memory/iterative_reader.py',
               'scripts/run_iterative_memory_pilot.py','tests/test_iterative_memory_reader.py',
               'tests/test_iterative_memory_pilot.py']
ARMS = ('single_regional','iterative_generic','iterative_regional')


def status(out,phase,**values):
    record={'phase':phase,'pid':os.getpid(),'updated_at':time.time(),**values}
    base.atomic_json(out/'run_status.json',record)
    print(json.dumps(record),flush=True)


def require(condition,message):
    if not condition:
        raise RuntimeError(message)


def mode_for(arm):
    return 'generic' if arm=='iterative_generic' else 'regional'


def new_reader(arm,seed):
    require(arm in ARMS,'Unknown arm')
    torch.manual_seed(seed)
    return ComplementaryMemoryReader() if arm=='single_regional' else IterativeMemoryReader()


def optimizer_for(reader,protocol):
    config=protocol['training']
    optimizer=torch.optim.AdamW(reader.parameters(),lr=config['learning_rate'],
                                 weight_decay=config['weight_decay'])
    def clip(optimizer,args,kwargs):
        torch.nn.utils.clip_grad_norm_(reader.parameters(),config['clip_grad_norm'],
                                       error_if_nonfinite=True)
    optimizer.register_step_pre_hook(clip)
    return optimizer


def training_plan(videos,protocol):
    """Same author-sampled clips and order for every condition; no outcomes used."""
    data,config=protocol['data'],protocol['training']
    plan=[]
    for epoch in range(1,config['epochs']+1):
        epoch_rows=[]
        for name in data['train']:
            for draw in range(config['clips_per_sequence_per_epoch']):
                seed=protocol['seed']+epoch*100000+int(name[3:])*100+draw
                selected=sample_clip(videos[name],config['clip_length'],seed)
                positions={f.frame_index:i for i,f in enumerate(videos[name])}
                indices=[positions[f.frame_index] for f in selected]
                require(indices==list(range(indices[0],indices[0]+config['clip_length'])),
                        'Author sampler changed contiguous clip geometry')
                epoch_rows.append({'epoch':epoch,'sequence':name,'draw':draw,
                                   'sampler_seed':seed,'indices':indices})
        random.Random(protocol['seed']+epoch).shuffle(epoch_rows)
        plan.extend(epoch_rows)
    return plan


def guard(core,videos,name=None):
    for path,sha in core['sources'].items():
        require(digest(ROOT/path)==sha,f'Source changed: {path}')
    if name is not None:
        require(base.input_hashes(videos,[name])[name]==core['inputs'][name],f'Input changed: {name}')


def setup(args,videos,out):
    protocol=json.loads((ROOT/PROTOCOL).read_text())
    head=base.git_text('rev-parse','HEAD')
    for relative in OWN_SOURCES:
        committed=subprocess.check_output(['git','show',f'{head}:{relative}'],cwd=ROOT)
        require(committed==(ROOT/relative).read_bytes(),f'Uncommitted study source: {relative}')
    require(digest(args.checkpoint)==protocol['checkpoint_sha256'],'Wrong checkpoint')
    require(digest(ROOT/base.MANIFEST_REL)==protocol['manifest_sha256'],'Changed manifest')
    train,val,test=[protocol['data'][k] for k in ('train','validation','test')]
    require(not(set(train)&set(val) or set(train)&set(test) or set(val)&set(test)), 'Split leakage')
    repo,revision=base.activate_author_repo()
    require(revision==protocol['sam2_revision'],'Wrong SAM2 revision')
    sources=list(dict.fromkeys(list(base.SOURCE_REL)+OWN_SOURCES+[
        'scripts/verify_complementary_memory.py','datasets/sam2_clip_sampler.py',
        'datasets/instance_masks.py','external/sam2/training/dataset/vos_sampler.py',
        'external/Cutie/cutie/config/model/base.yaml','external/Cutie/cutie/config/train_config.yaml']))
    # Constructing the reader verifies the clean pinned upstream checkout.
    identities={arm:base.state_digest(new_reader(arm,protocol['seed']).state_dict()) for arm in ARMS}
    require(identities['iterative_generic']==identities['iterative_regional'],'Unmatched iterative initialization')
    core={'protocol':protocol,'sources':{p:digest(ROOT/p) for p in sources},
          'checkpoint_sha256':digest(args.checkpoint),'inputs':base.input_hashes(videos,train+val+test),
          'sam2_revision':revision,'torch':torch.__version__,'initial_states':identities,
          'plan':training_plan(videos,protocol)}
    core_hash=hashlib.sha256(json.dumps(core,sort_keys=True).encode()).hexdigest()
    if args.resume:
        saved=json.loads((out/'setup.json').read_text())
        require(saved['core_sha256']==core_hash,'Resume source/data/protocol mismatch')
        require(digest(out/'source_snapshot.zip')==saved['snapshot_sha256'],'Snapshot changed')
    else:
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for path in sources:
                archive.write(ROOT/path,path)
        base.atomic_json(out/'setup.json',{'commit':head,'core':core,'core_sha256':core_hash,
            'snapshot_sha256':digest(out/'source_snapshot.zip'),'command':sys.argv,
            'scope':'Exploratory PolypGen pilot; original foundation weights frozen'})
    return core,core_hash,repo


def verify_native_archive(native,name,drive):
    path=drive/'outputs/experiment/143_official_native_feature_audit_v1/full'
    completion=json.loads((path/'completion.json').read_text())
    audit=json.loads((path/'independent_audit.json').read_text())
    require(completion['passed'] and audit['passed'],'Native archive not verified')
    folder=Path(completion['entries'][name])
    require(digest(folder/'receipt.json')==audit['sequences'][name]['receipt_sha256'],'Changed native receipt')
    receipt=json.loads((folder/'receipt.json').read_text())
    expected={r['frame']:r['sha256'] for r in receipt['predictions']}
    require(all(r['prediction_sha256']==expected[r['frame']] for r in native['rows']
                if r['frame']!=native['prompt_frame']), 'Native full-logit archive mismatch')


def engineering(args,videos,out,core,core_hash,repo,drive):
    folder=out/'gate';folder.mkdir(exist_ok=True)
    receipt=folder/'completion.json'
    if receipt.exists():
        saved=json.loads(receipt.read_text())
        require(saved['passed'] and saved['core_sha256']==core_hash,'Gate receipt mismatch')
        require(digest(folder/'results.json')==saved['results_sha256'],'Gate result changed')
        return
    protocol=core['protocol']; seed=protocol['seed']; parity=[]; native_records={}
    for name in ('seq2','seq3'):
        status(out,'engineering_native',sequence=name)
        native=rollout(args.checkpoint,repo,videos[name],folder)
        verify_native_archive(native,name,drive)
        native_records[name]=native
        base.atomic_json(folder/f'{name}_native.json',native)
        for arm in ARMS:
            status(out,'engineering_identity',sequence=name,arm=arm)
            reader=new_reader(arm,seed).eval()
            adapted=rollout(args.checkpoint,repo,videos[name],folder,reader,mode_for(arm))
            parity.append(compare_exact(native,adapted,name,arm))
            base.atomic_json(folder/f'{name}_{arm}_identity.json',adapted)
    learning={}; trained=None
    for arm in ARMS:
        status(out,'engineering_gradients',arm=arm)
        model=base.init_predictor(args.checkpoint,repo);original=parameter_digest(model)
        reader=new_reader(arm,seed).train();optimizer=optimizer_for(reader,protocol)
        records=[]
        for name in ('seq2','seq3'):
            records.append(base.train_video(model,reader,mode_for(arm),optimizer,videos[name],name,
                folder,out/'progress.json',seed,arm,max_post_frames=8))
        require(parameter_digest(model)==original,'Frozen original weights changed')
        groups={key:sum(chunk['gradient_groups_l1'][key] for row in records for chunk in row['chunks'])
                for key in records[0]['chunks'][0]['gradient_groups_l1']}
        require(all(value>0 for value in groups.values()),'An added component never received gradients')
        # Native operations are deterministic; round-trip state and optimizer before another real clip.
        save=folder/f'{arm}_resume.pt'
        base.atomic_torch_save(save,{'reader':reader.state_dict(),'optimizer':optimizer.state_dict(),
                                    'rng':torch.random.get_rng_state()})
        first=copy.deepcopy(reader);first_optimizer=optimizer_for(first,protocol)
        first_optimizer.load_state_dict(optimizer.state_dict())
        checkpoint=torch.load(save,map_location='cpu',weights_only=True)
        second=new_reader(arm,seed);second.load_state_dict(checkpoint['reader'])
        second_optimizer=optimizer_for(second,protocol);second_optimizer.load_state_dict(checkpoint['optimizer'])
        for candidate,opt in ((first,first_optimizer),(second,second_optimizer)):
            torch.random.set_rng_state(checkpoint['rng'])
            base.train_video(model,candidate,mode_for(arm),opt,videos['seq3'],'seq3',folder,
                             out/'progress.json',seed,arm,max_post_frames=4)
        require(base.state_digest(first.state_dict())==base.state_digest(second.state_dict()),
                'Real-frame checkpoint resume mismatch')
        learning[arm]={'gradients':groups,'training':records,'real_frame_resume_exact':True,
                       'parameters':sum(p.numel() for p in reader.parameters()),
                       'sam2_parameters_unchanged':True}
        active=rollout(args.checkpoint,repo,videos['seq3'],folder,reader.eval(),mode_for(arm),max_targets=8)
        require(any(r['reader_update_l1']>0 for r in active['reader_calls']), 'Learned module inactive')
        base.atomic_json(folder/f'{arm}_active.json',active)
        if arm=='iterative_regional':
            trained=reader.eval()
        del model,first,second,optimizer,first_optimizer,second_optimizer
        gc.collect()
    status(out,'engineering_multi_object')
    model=base.init_predictor(args.checkpoint,repo)
    prompt=core['inputs']['seq2']['prompt']
    multi=multi_object_gate(model,trained,videos['seq2'][prompt:],folder)
    del model;gc.collect()
    guard(core,videos)
    results={'passed':True,'parity':parity,'learning':learning,'multi_object':multi,
             'native_archive_matches':True,'scope':'Engineering only; no efficacy claim'}
    base.atomic_json(folder/'results.json',results)
    base.atomic_json(receipt,{'passed':True,'core_sha256':core_hash,'results_sha256':digest(folder/'results.json')})
    status(out,'engineering_passed')


def evaluate(args,reader,arm,videos,names,folder,repo,core_hash,out,phase):
    model=base.init_predictor(args.checkpoint,repo)
    if reader is not None:
        reader.eval()
    result={}
    for name in names:
        status(out,phase,arm=arm,sequence=name)
        result[name]=base.evaluate_video(model,videos[name],reader,mode_for(arm),folder,
                                          name,20261107,core_hash)
    del model;gc.collect()
    return result


def train(args,arm,videos,out,core,core_hash,repo):
    protocol=core['protocol']; config=protocol['training'];seed=protocol['seed']
    folder=out/'training'/arm;folder.mkdir(parents=True,exist_ok=True)
    model=base.init_predictor(args.checkpoint,repo);original=parameter_digest(model)
    reader=new_reader(arm,seed).train();optimizer=optimizer_for(reader,protocol)
    checkpoint=folder/'resume.pt'; completed=0; validation={};best=-float('inf');best_epoch=None
    clip_hashes=[]
    if checkpoint.exists():
        require(args.resume,'Training checkpoint exists without --resume')
        receipt=json.loads((folder/'resume_receipt.json').read_text())
        require(digest(checkpoint)==receipt['sha256'],'Resume checkpoint bytes changed')
        saved=torch.load(checkpoint,map_location='cpu',weights_only=True)
        require(saved['core_sha256']==core_hash and saved['arm']==arm,'Incompatible training checkpoint')
        reader.load_state_dict(saved['reader']);optimizer.load_state_dict(saved['optimizer'])
        torch.random.set_rng_state(saved['rng'])
        completed=saved['completed'];validation=saved['validation'];best=saved['best'];best_epoch=saved['best_epoch']
        clip_hashes=saved['clip_hashes']
        require(len(clip_hashes)==completed and receipt['completed']==completed,'Resume coverage mismatch')
        for index,sha in enumerate(clip_hashes):
            require(digest(folder/'clips'/f'{index:04d}.json')==sha,'Completed-clip receipt changed')
        if best_epoch is not None:
            require(digest(folder/'best.pt')==saved['best_sha256'],'Selected checkpoint changed')
    plan=core['plan'];per_epoch=len(protocol['data']['train'])*config['clips_per_sequence_per_epoch']
    for index in range(completed,len(plan)):
        row=plan[index];name=row['sequence'];reader.train()
        if row['draw']==0:
            guard(core,videos,name)
        status(out,'training',arm=arm,epoch=row['epoch'],clip=index+1,total_clips=len(plan),sequence=name)
        frames=[videos[name][i] for i in row['indices']]
        record=base.train_video(model,reader,mode_for(arm),optimizer,frames,name,folder,
                               out/'progress.json',seed,arm)
        require(record['post_prompt_frames']==7 and record['optimizer_updates']==2,'Training clip coverage changed')
        clip_path=folder/'clips'/f'{index:04d}.json'
        payload={'plan':row,'record':record}
        if clip_path.exists():
            require(json.loads(clip_path.read_text())==payload,'Unsaved clip does not replay exactly on resume')
        else:
            base.atomic_json(clip_path,payload)
        clip_hashes.append(digest(clip_path))
        # Validation checkpoints are fixed in advance. Exact ties retain the earliest epoch.
        epoch_done=(index+1)%per_epoch==0
        if epoch_done and row['epoch'] in config['validation_epochs']:
            require(parameter_digest(model)==original,'Training modified original SAM2')
            epoch=row['epoch'];epoch_file=folder/f'epoch_{epoch:02d}.pt'
            base.atomic_torch_save(epoch_file,reader.state_dict())
            summaries=evaluate(args,reader,arm,videos,protocol['data']['validation'],
                out/'validation'/arm/f'epoch_{epoch:02d}',repo,core_hash,out,'validation')
            score=float(np.mean([s['present_dice'] for s in summaries.values()]))
            validation[str(epoch)]={'present_dice':score,'summaries':summaries,'checkpoint_sha256':digest(epoch_file)}
            if score>best:
                best,best_epoch=score,epoch
                base.atomic_torch_save(folder/'best.pt',reader.state_dict())
        if (index+1)%config['checkpoint_every_clips']==0 or index+1==len(plan):
            base.atomic_torch_save(checkpoint,{'core_sha256':core_hash,'arm':arm,'reader':reader.state_dict(),
                'optimizer':optimizer.state_dict(),'rng':torch.random.get_rng_state(),'completed':index+1,
                'validation':validation,'best':best,'best_epoch':best_epoch,'clip_hashes':clip_hashes,
                'best_sha256':digest(folder/'best.pt') if best_epoch is not None else None})
            base.atomic_json(folder/'resume_receipt.json',{'sha256':digest(checkpoint),'completed':index+1})
    require(best_epoch is not None,'No validation-selected checkpoint')
    require(parameter_digest(model)==original,'Original SAM2 changed')
    reader.load_state_dict(torch.load(folder/'best.pt',weights_only=True,map_location='cpu'))
    base.atomic_json(folder/'completion.json',{'complete':True,'clips':len(plan),'updates':2*len(plan),
        'validation':validation,'best_epoch':best_epoch,'best_validation_present_dice':best,
        'best_checkpoint_sha256':digest(folder/'best.pt'),'original_parameters_sha256':original,
        'convergence_caution':'A bounded schedule and validation selection do not by themselves demonstrate convergence; inspect the learning curves.'})
    del model;gc.collect()
    return reader.eval()


def report(summaries,protocol):
    names=protocol['data']['test'];metrics={}
    for arm,records in summaries.items():
        require(set(records)==set(names),'Incomplete test cohort')
        metrics[arm]={key:float(np.mean([r[key] for r in records.values()])) for key in
                      ('present_dice','present_iou','present_boundary_f','elapsed_seconds')}
        metrics[arm]['false_alarms']=sum(r['gt_empty_false_alarm_count'] for r in records.values())
        metrics[arm]['empty_frames']=sum(r['gt_empty_frames'] for r in records.values())
    contrasts={};pvalues={}
    for control in ('native','iterative_generic','single_regional'):
        left={n:summaries['iterative_regional'][n]['present_dice'] for n in names}
        right={n:summaries[control][n]['present_dice'] for n in names}
        delta={n:left[n]-right[n] for n in names}
        contrasts[control]={'mean_dice_gain':float(np.mean(list(delta.values()))),
                           'bootstrap_95_ci':sequence_bootstrap_interval(delta,seed=protocol['seed']),
                           'wins':sum(v>0 for v in delta.values()),'per_sequence':delta}
        pvalues[control]=paired_wilcoxon(left,right)['p']
    checks={}
    for control,minimum in (('native',.01),('iterative_generic',.005),('single_regional',.002)):
        c=contrasts[control]
        checks[control]=c['mean_dice_gain']>=minimum and c['bootstrap_95_ci'][0]>0 and c['wins']>=5
    native=metrics['native'];regional=metrics['iterative_regional']
    checks['false_alarm_safety']=(regional['empty_frames']==native['empty_frames'] and
        (regional['empty_frames']==0 or regional['false_alarms']/regional['empty_frames']<=
         native['false_alarms']/native['empty_frames']+.02))
    return {'scope':'Exploratory eight-sequence full-propagation pilot; cohort previously examined',
            'per_sequence':summaries,'means':metrics,'contrasts':contrasts,'holm_p':holm_adjust(pvalues),
            'continuation_checks':checks,'continuation_passed':all(checks.values())}


def main():
    parser=argparse.ArgumentParser(__doc__)
    parser.add_argument('--checkpoint',type=Path,required=True)
    parser.add_argument('--output-dir',type=Path,required=True)
    parser.add_argument('--resume',action='store_true')
    parser.add_argument('--gate-only',action='store_true')
    args=parser.parse_args();args.checkpoint=args.checkpoint.resolve()
    out=validate_output(args.output_dir)
    from scripts.diagnose_complementary_memory import drive_root
    drive=drive_root(out)
    require(out.parent.name=='174_iterative_memory_reader_v1' and out.name.startswith('attempt_'),'Wrong output path')
    out.mkdir(parents=True,exist_ok=args.resume)
    started=time.monotonic()
    try:
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        videos=load_manifest(ROOT/base.MANIFEST_REL,'dev')|load_manifest(ROOT/base.MANIFEST_REL,'test')
        core,core_hash,repo=setup(args,videos,out)
        engineering(args,videos,out,core,core_hash,repo,drive)
        if args.gate_only:
            status(out,'engineering_complete_pilot_not_launched');return
        summaries={}
        # Finish every arm's training/checkpoint selection before inspecting pilot-test results.
        for arm in ARMS:
            reader=train(args,arm,videos,out,core,core_hash,repo)
            del reader;gc.collect()
        for arm in ARMS:
            reader=new_reader(arm,core['protocol']['seed'])
            reader.load_state_dict(torch.load(out/'training'/arm/'best.pt',weights_only=True,map_location='cpu'))
            summaries[arm]=evaluate(args,reader,arm,videos,core['protocol']['data']['test'],
                                     out/'evaluation'/arm,repo,core_hash,out,'test_evaluation')
            guard(core,videos)
        summaries['native']=evaluate(args,None,'native',videos,core['protocol']['data']['test'],
                                     out/'evaluation'/'native',repo,core_hash,out,'native_evaluation')
        results=report(summaries,core['protocol']);base.atomic_json(out/'results.json',results)
        guard(core,videos)
        all_sequences=sum([core['protocol']['data'][key] for key in ('train','validation','test')],[])
        require(base.input_hashes(videos,all_sequences)==core['inputs'],'Input changed during study')
        base.atomic_json(out/'completion.json',{'complete':True,'core_sha256':core_hash,
            'results_sha256':digest(out/'results.json'),'elapsed_seconds':time.monotonic()-started,
            'continuation_passed':results['continuation_passed']})
        status(out,'complete',continuation_passed=results['continuation_passed'])
    except BaseException as error:
        base.atomic_json(out/'failure.json',{'error':repr(error),'time':time.time()})
        status(out,'failed',error=repr(error));raise


if __name__=='__main__':
    main()
