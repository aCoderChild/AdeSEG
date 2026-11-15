#!/usr/bin/env python3
"""Exploratory, source-bound nine-arm channel-value fixed-read feasibility.

Plan is CPU-only and label-free. Parity and training may open training labels
only. Replay opens no target labels. Score is permitted only after a separate
complete replay. This is not a live video integration or independent test.
"""
import argparse
import json
import os
from pathlib import Path
import random
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import save_binary_mask
from modeling.foreground_conditioned_refiner import ForegroundConditionedRefiner
from modeling.lean_channel_attention import author_sha256
from modeling.cmsa_boundary_training_loss import cmsa_boundary_loss
from modeling.multiscale_channel_value_refiner_v1 import MultiscaleChannelValueRefinerV1
from scripts.memory_token_io import digest
from scripts.resume_source_group_capture_v1 import BASE
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.source_group_training_label_guard_v1 import install_training_mask_guard
from scripts.source_group_training_label_guard_v1 import mask_path_sets
from scripts.source_group_refiner_shared_v2 import (EPOCHS, SEED, dependencies as previous_dependencies,
    ensure_target, initial_hash, load_prepared, read, schedule)
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture, to_video
from scripts.train_region_mixed_loss_v1 import environment_identity
from scripts.score_source_group_boundary_v1 import metric, summary

ARMS = ('mask_only', 'diagonal', 'global', 'global_contrast', 'foreground',
        'foreground_background_no_contrast', 'foreground_background',
        'foreground_background_asinh', 'region_reference')
PARITY_KEYS = (('seq2', 4), ('seq4', 18), ('seq4', 24), ('seq13', 8))
SOURCES = ('scripts/channel_value_fixed_read_v1.py',
    'scripts/run_channel_value_fixed_read_v1.py',
    'modeling/multiscale_channel_value_refiner_v1.py',
    'scripts/score_source_group_boundary_v1.py', 'modeling/robust_segmentation_loss.py',
    'scripts/audit_boundary_video_factorial_v1.py',
    'modeling/cmsa_boundary_training_loss.py',
    'scripts/resume_source_group_capture_v1.py', 'scripts/freeze_source_group_runtime_v1.py',
    'scripts/run_source_group_capture_v1.py', 'scripts/audit_source_group_capture_v1.py',
    'data/polypgen_sequence.jsonl',
    'protocols/MULTISCALE_CHANNEL_VALUE_REFINER_V1.md')
CONFIG = {'early_channels': 32, 'deep_channels': 256, 'hidden': 64, 'attention_size': 32}
LOSS = 'CMSA corrected weighted BCE + aggregate weighted IoU + project soft Dice'


def require(ok, message):
    if not ok: raise ValueError(message)


def require_environment_flags():
    expected={'PYTHONHASHSEED':'0','PYTORCH_ENABLE_MPS_FALLBACK':'0','PYTORCH_MPS_FAST_MATH':'0'}
    require(all(os.environ.get(k)==v for k,v in expected.items()), 'Required hash seed or MPS flags differ')
    return expected


def install_phase_mask_guard(manifest):
    train,protected=mask_path_sets(manifest);blocked=train|protected
    state={'active':True,'blocked_paths':len(blocked),'open_attempts':0}
    def audit(event,args):
        if (state['active'] and event=='open' and args and isinstance(args[0],(str,bytes,os.PathLike))
                and os.path.realpath(os.fsdecode(args[0])) in blocked):
            state['open_attempts']+=1
            raise PermissionError('Label open blocked before verification')
    sys.addaudithook(audit)
    return state


def outpath(path):
    p = Path(path).resolve()
    require(p.is_relative_to(BASE.resolve(strict=True)) and not p.exists(), 'Output must be a new Drive experiment path')
    return p


def deps():
    return {'current': {p: digest(ROOT/p) for p in SOURCES}, 'prior': previous_dependencies()}


def model_for(arm):
    if arm == 'region_reference': return ForegroundConditionedRefiner(gate_mode='region', contrast_head=True)
    require(arm in ARMS, 'Unknown arm')
    return MultiscaleChannelValueRefinerV1(mode=arm, **CONFIG)


def payload_for(model, arm, manifest, manifest_sha, plan_path, parity_path, steps):
    common = {'research_only': True, 'training_steps': steps,
        'state_dict': {k: v.detach().cpu() for k, v in model.state_dict().items()},
        'backbone_sha256': manifest['checkpoint_sha256'], 'feature_pipeline_sha256': manifest['feature_pipeline_sha256'],
        'train_manifest_sha256': manifest_sha, 'study_arm': arm, 'plan_sha256': digest(plan_path),
        'parity_sha256': digest(parity_path), 'dependency_sha256': deps()}
    if arm == 'region_reference':
        common.update(format='foreground_conditioned_refiner_v1', author_sha256=author_sha256(),
            conditioned_source_sha256=digest(ROOT/'modeling/foreground_conditioned_refiner.py'),
            base_source_sha256=digest(ROOT/'modeling/multilevel_contrast_refiner.py'),
            gate_mode='region', contrast_head=True, arm='multilevel')
    else:
        common.update(format='multiscale_channel_value_refiner_v1',
            source_sha256=digest(ROOT/'modeling/multiscale_channel_value_refiner_v1.py'), mode=arm, config=CONFIG)
    return common


def load_model(payload, arm, manifest, manifest_sha):
    model = model_for(arm)
    model.load_trained_state(payload, backbone_sha256=manifest['checkpoint_sha256'],
        feature_pipeline_sha256=manifest['feature_pipeline_sha256'], train_manifest_sha256=manifest_sha)
    return model


def build_plan(manifest, manifest_sha):
    identities, counts = {}, {}
    for arm in ARMS:
        torch.manual_seed(SEED); model = model_for(arm)
        identities[arm] = initial_hash(model); counts[arm] = sum(p.numel() for p in model.parameters())
    require(len(set(identities[a] for a in ARMS[:-1])) == 1 and
            all(counts[a] == 63809 for a in ARMS[:-1]) and counts['region_reference'] == 48005,
            'Matched initialization or parameter count differs')
    return {'passed': True, 'scope': 'Exploratory fixed-read feasibility; reused development videos.',
        'manifest_sha256': manifest_sha, 'feature_pipeline_sha256': manifest['feature_pipeline_sha256'],
        'dependency_sha256': deps(), 'environment': manifest['feature_pipeline']['runtime_environment'],
        'arms': list(ARMS), 'initial_state_sha256': identities, 'parameter_count': counts,
        'environment_flags':require_environment_flags(),
        'seed': SEED, 'epochs': EPOCHS, 'training_order': schedule(manifest['samples']),
        'loss': LOSS, 'optimizer': 'Adam', 'learning_rate': 0.0003, 'weight_decay': 0.0001,
        'parity_keys': [list(x) for x in PARITY_KEYS], 'config': CONFIG,
        'limitations': ['Region reference is not parameter-matched.', 'Diagonal changes normalization as well as mixing.',
                        'Reused validation outcomes are not independent.', 'No live propagation is evaluated.']}


def require_plan(path, manifest, sha):
    saved = read(path)
    require(saved == build_plan(manifest, sha), 'Frozen plan or sources changed')
    return saved


def phase_inputs(run, plan_path, parity_path=None):
    run=Path(run).resolve(strict=True)
    require(run.is_relative_to(BASE.resolve(strict=True)), 'Study outside Drive experiment root')
    phases=[json.loads(x) for x in (run/'phases.jsonl').read_text().splitlines() if x.strip()]
    names=('plan',) if parity_path is None else ('plan','parity')
    require([x.get('phase') for x in phases[:len(names)]]==list(names),'Plan or parity phase missing')
    paths=(Path(plan_path),) if parity_path is None else (Path(plan_path),Path(parity_path))
    expected_paths=(run/'plan/plan.json',) if parity_path is None else (run/'plan/plan.json',run/'parity/parity.json')
    hashes=[]
    for phase,name,path,expected_path in zip(phases,names,paths,expected_paths,strict=False):
        require(path.resolve(strict=True)==expected_path.resolve(strict=True) and phase.get('passed') is True and
                phase.get('log_sha256')==digest(run/(name+'.log')) and
                phase.get('artifact_sha256')==digest(path),f'{name} artifact or phase changed')
        hashes.append(phase['artifact_sha256'])
    return hashes


def inputs(item, device):
    early, deep, logits, shape = read_capture(item)
    require(all(digest(item[k])==item[k+'_sha256'] for k in ('feature','snapshot')),
            'Captured features changed during load')
    return (early.to(device), deep.to(device), logits.to(device)), shape


def checked_target(item,shape):
    target=ensure_target(item,shape)
    require(digest(item['label'])==item['label_sha256'],'Label changed during decoding')
    return target


def parity(manifest, sha, plan_path, output, guard):
    expected=phase_inputs(Path(output).parent,plan_path)
    plan = require_plan(plan_path, manifest, sha)
    require(digest(plan_path)==expected[0], 'Plan changed during parsing')
    require(environment_identity() == plan['environment'], 'Capture environment differs')
    checks = []
    for arm in ARMS:
        torch.manual_seed(SEED); cpu = model_for(arm).train()
        original_mps=model_for(arm).train();original_mps.load_state_dict(cpu.state_dict());original_mps.to('mps')
        for key in (PARITY_KEYS[0],('seq4',24)):
            item=next(x for x in manifest['samples'] if x['split']=='train' and (x['sequence'],x['frame'])==key)
            for net,device in ((cpu,'cpu'),(original_mps,'mps')):
                identity_args,_=inputs(item,device)
                require(torch.equal(net(*identity_args),identity_args[2]), f'{device} zero-residual identity failed')
        del original_mps
        torch.nn.init.normal_(cpu.head[-1].weight, mean=0, std=0.01)
        mps = model_for(arm).train(); mps.load_state_dict(cpu.state_dict()); mps.to('mps')
        for seq, frame in PARITY_KEYS:
            item = next(x for x in manifest['samples'] if x['split'] == 'train' and (x['sequence'], x['frame']) == (seq, frame))
            values, target = [], None
            for net, device in ((cpu, 'cpu'), (mps, 'mps')):
                net.zero_grad(set_to_none=True); args, shape = inputs(item, device)
                if target is None: target = checked_target(item, shape)
                if (seq,frame)==('seq4',24):
                    require(bool(torch.all(args[2]==-1024)) and bool(torch.any(target>0)),
                            'Training sentinel probe is not suppressed and present')
                pred = net(*args); loss = cmsa_boundary_loss(to_video(pred, shape), target.to(device))
                require(torch.isfinite(pred).all() and torch.isfinite(loss), 'Nonfinite parity values')
                loss.backward()
                grads = [None if p.grad is None else p.grad.detach().cpu().clone() for p in net.parameters()]
                require(all(g is None or torch.isfinite(g).all() for g in grads), 'Nonfinite parity gradient')
                values.append((pred.detach().cpu(), loss.detach().cpu(), grads, (to_video(pred.detach(), shape)>0).cpu()))
            torch.testing.assert_close(values[0][0], values[1][0], rtol=0, atol=0.001)
            torch.testing.assert_close(values[0][1], values[1][1], rtol=0.001, atol=0.001)
            grad_ratio = 0.0
            for x, y in zip(values[0][2], values[1][2], strict=True):
                require((x is None) == (y is None), 'Parity gradient presence differs')
                if x is not None:
                    torch.testing.assert_close(x, y, rtol=0.001, atol=0.001)
                    grad_ratio = max(grad_ratio, float(((x.double()-y.double()).abs()/(0.001+0.001*y.double().abs())).max()))
            changed = int((values[0][3] != values[1][3]).sum()); pixels = values[0][3].numel()
            require(changed <= 20 and changed/pixels <= 1e-5, 'CPU/MPS native masks differ')
            checks.append({'arm': arm, 'sequence': seq, 'frame': frame, 'pixels': pixels, 'mask_difference': changed,
                'logit_difference': float((values[0][0]-values[1][0]).abs().max()),
                'loss_ratio': float((values[0][1]-values[1][1]).abs()/(0.001+0.001*values[1][1].abs())),
                'gradient_ratio': grad_ratio})
    require(guard['blocked_open_attempts']==0, 'Protected label opened during parity')
    out = outpath(output); out.mkdir(parents=True)
    (out/'parity.json').write_text(json.dumps({'passed':True, 'manifest_sha256':sha,
        'plan_sha256':digest(plan_path), 'environment':environment_identity(),
        'environment_flags':require_environment_flags(), 'checks':checks,
        'label_guard':dict(guard)}, indent=2)+'\n')


def require_parity(path, sha, plan_path, plan):
    r = read(path); checks = r.get('checks', [])
    train_paths,protected_paths=mask_path_sets(ROOT/'data/polypgen_sequence.jsonl')
    require(r.get('passed') is True and r.get('manifest_sha256') == sha and r.get('plan_sha256') == digest(plan_path)
        and r.get('label_guard',{}).get('blocked_open_attempts')==0
        and r['label_guard'].get('train_paths')==len(train_paths)
        and r['label_guard'].get('blocked_paths')==len(protected_paths)
        and r.get('environment') == plan['environment'] and r.get('environment_flags')==plan['environment_flags']
        and len(checks) == len(ARMS)*len(PARITY_KEYS)
        and {(x['arm'], x['sequence'], x['frame']) for x in checks} == {(a,s,f) for a in ARMS for s,f in PARITY_KEYS}
        and all(x['pixels'] > 0 and x['mask_difference'] <= 20 and x['mask_difference']/x['pixels'] <= 1e-5
                and np.isfinite(x['logit_difference']) and x['logit_difference'] <= 0.001
                and np.isfinite(x['loss_ratio']) and x['loss_ratio'] <= 1.00001
                and np.isfinite(x['gradient_ratio']) and x['gradient_ratio'] <= 1.00001 for x in checks),
        'Parity receipt differs')


def train(manifest, sha, arm, plan_path, parity_path, output, guard):
    expected=phase_inputs(Path(output).parent.parent,plan_path,parity_path)
    plan = require_plan(plan_path, manifest, sha); require_parity(parity_path, sha, plan_path, plan)
    require([digest(plan_path),digest(parity_path)]==expected, 'Plan or parity changed during parsing')
    require(environment_identity() == plan['environment'], 'Capture environment differs')
    rows = {(x['sequence'],x['frame']):x for x in manifest['samples'] if x['split']=='train'}
    val = sorted((x for x in manifest['samples'] if x['split']=='validation'), key=lambda x:(x['sequence'],x['frame']))
    out = outpath(output); out.mkdir(parents=True)
    torch.manual_seed(SEED); torch.mps.manual_seed(SEED); random.seed(SEED); np.random.seed(SEED)
    model = model_for(arm).train(); require(initial_hash(model) == plan['initial_state_sha256'][arm], 'Initial state differs')
    model.to('mps'); optimizer = torch.optim.Adam(model.parameters(), lr=0.0003, weight_decay=0.0001)
    history = []
    for epoch, order in enumerate(plan['training_order']):
        losses = []; peak_gradient = -1.; peak_example = None
        for seq, frame in order:
            item = rows[(seq,frame)]; args, shape = inputs(item, 'mps'); target = checked_target(item, shape).to('mps')
            optimizer.zero_grad(set_to_none=True); loss = cmsa_boundary_loss(to_video(model(*args), shape), target)
            require(torch.isfinite(loss), 'Nonfinite loss'); loss.backward()
            require(all(p.grad is None or torch.isfinite(p.grad).all() for p in model.parameters()), 'Nonfinite gradient')
            squared=sum((p.grad.float().square().sum() for p in model.parameters() if p.grad is not None),
                        torch.zeros((),device='mps'))
            gradient=float(torch.sqrt(squared).detach().cpu())
            require(np.isfinite(gradient), 'Nonfinite gradient norm')
            if gradient>peak_gradient:peak_gradient=gradient;peak_example=[seq,frame]
            optimizer.step(); require(all(torch.isfinite(p).all() for p in model.parameters()), 'Nonfinite parameter')
            losses.append(float(loss.detach().cpu()))
        history.append({'epoch':epoch+1, 'samples':len(losses), 'order':order, 'mean_loss':float(np.mean(losses)),
                        'peak_gradient_l2':peak_gradient,'peak_gradient_read':peak_example})
        (out/'progress.json').write_text(json.dumps({'history':history},indent=2)+'\n')
        print(f'{arm}: epoch {epoch+1}/{EPOCHS}', flush=True)
    ck = out/'checkpoint.pt'; torch.save(payload_for(model,arm,manifest,sha,plan_path,parity_path,len(rows)*EPOCHS),ck)
    fitted = load_model(torch.load(ck,map_location='cpu',weights_only=True),arm,manifest,sha).to('mps').eval()
    folder=out/'frozen_predictions'; folder.mkdir(); predictions=[]
    with torch.no_grad():
        for item in val:
            args,shape=inputs(item,'mps'); output_logits=fitted(*args)
            require(torch.isfinite(output_logits).all(), 'Nonfinite frozen output')
            masks={'baseline':(to_video(args[2],shape)>0).cpu().numpy()[0,0],
                   'refined':(to_video(output_logits,shape)>0).cpu().numpy()[0,0]}
            row={'sequence':item['sequence'],'frame':item['frame'],'shape':list(shape)}
            for name,mask in masks.items():
                path=folder/f"{item['sequence']}_{item['frame']:06d}_{name}.png"; save_binary_mask(mask,path)
                row[name]=str(path);row[name+'_sha256']=digest(path)
            predictions.append(row)
    require(guard['blocked_open_attempts']==0, 'Protected label was opened')
    frozen=out/'frozen_predictions.json'; frozen.write_text(json.dumps({'checkpoint_sha256':digest(ck),'rows':predictions},indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'arm':arm,'manifest_sha256':sha,
        'plan_sha256':digest(plan_path),'parity_sha256':digest(parity_path),'checkpoint_sha256':digest(ck),
        'frozen_sha256':digest(frozen),'initial_state_sha256':plan['initial_state_sha256'][arm],
        'history':history,'label_guard':dict(guard),'environment_flags':require_environment_flags()},indent=2)+'\n')


def validated_runs(run, manifest, sha, plan_path, parity_path, plan):
    """Validate all outputs and load identities without opening label masks."""
    run = Path(run).resolve(strict=True)
    require(run.is_relative_to(BASE.resolve(strict=True)), 'Study outside Drive experiment root')
    phases = [json.loads(x) for x in (run/'phases.jsonl').read_text().splitlines() if x.strip()]
    training = [x for x in phases if x['phase'].startswith('train_')]
    expected_phases = ['plan','parity'] + ['train_'+a for a in ARMS]
    require([x['phase'] for x in phases[:len(expected_phases)]] == expected_phases and
            all(x.get('passed') is True and digest(run/(x['phase']+'.log')) == x.get('log_sha256')
                for x in phases[:len(expected_phases)]) and
            [x['phase'] for x in training] == ['train_'+a for a in ARMS] and
            all(x.get('passed') is True and digest(run/(x['phase']+'.log')) == x.get('log_sha256') for x in training),
            'All training phases and logs must complete')
    train_paths, protected_paths = mask_path_sets(ROOT/'data/polypgen_sequence.jsonl')
    targets={(x['sequence'],x['frame']) for x in manifest['samples'] if x['split']=='validation'}
    result={}
    for arm, phase in zip(ARMS,training,strict=True):
        folder=run/'runs'/arm; completion=folder/'completion.json'; expected=phase['completion_sha256']
        require(digest(completion)==expected,'Completion changed after training')
        rec=read(completion); require(digest(completion)==expected,'Completion changed while parsing')
        ck=folder/'checkpoint.pt'; frozen=folder/'frozen_predictions.json'
        require(rec.get('passed') is True and rec.get('arm')==arm and rec.get('manifest_sha256')==sha and
            rec.get('plan_sha256')==digest(plan_path) and rec.get('parity_sha256')==digest(parity_path) and
            rec.get('checkpoint_sha256')==digest(ck) and rec.get('frozen_sha256')==digest(frozen) and
            rec.get('initial_state_sha256')==plan['initial_state_sha256'][arm] and
            rec.get('environment_flags')==plan['environment_flags'] and
            rec.get('label_guard',{}).get('blocked_open_attempts')==0 and
            rec['label_guard'].get('train_paths')==len(train_paths) and
            rec['label_guard'].get('blocked_paths')==len(protected_paths) and
            len(rec.get('history',[]))==EPOCHS and
            all(h.get('epoch')==i+1 and h.get('samples')==164 and h.get('order')==plan['training_order'][i]
                and np.isfinite(h.get('mean_loss',float('nan'))) and np.isfinite(h.get('peak_gradient_l2',float('nan')))
                and h.get('peak_gradient_l2')>=0 and h.get('peak_gradient_read') in plan['training_order'][i]
                for i,h in enumerate(rec['history'])),
            f'{arm}: training identity differs')
        frozen_rec=read(frozen); require(digest(frozen)==rec['frozen_sha256'] and frozen_rec['checkpoint_sha256']==digest(ck),
                                        'Frozen prediction identity differs')
        payload=torch.load(ck,map_location='cpu',weights_only=True)
        require(payload.get('study_arm')==arm and payload.get('training_steps')==164*EPOCHS and
                payload.get('plan_sha256')==digest(plan_path) and payload.get('parity_sha256')==digest(parity_path) and
                payload.get('dependency_sha256')==deps(), 'Checkpoint provenance differs')
        load_model(payload,arm,manifest,sha)
        mapping={(x['sequence'],x['frame']):x for x in frozen_rec['rows']}
        require(len(frozen_rec['rows'])==50 and len(mapping)==50 and set(mapping)==targets,'Validation coverage differs')
        for row in mapping.values():
            for kind in ('baseline','refined'):
                path=Path(row[kind]).resolve(strict=True)
                require(path.is_relative_to(folder.resolve(strict=True)) and digest(path)==row[kind+'_sha256'],
                        'Frozen prediction changed')
        result[arm]={'folder':folder,'completion_sha256':expected,'checkpoint_sha256':digest(ck),
                     'frozen_sha256':digest(frozen),'payload':payload,'mapping':mapping}
    return result


def factorial_effects(rows):
    """Descriptive paired Dice effects on identical labeled-present reads."""
    present=[r for r in rows if r['present']]
    terms={name:[] for name in ('matrix_without_contrast','matrix_with_contrast',
                               'contrast_with_global_matrix','contrast_with_region_matrix','interaction',
                               'asinh_head_input_vs_raw')}
    for r in present:
        g=r['global']['dice'];gc=r['global_contrast']['dice']
        f=r['foreground_background_no_contrast']['dice'];fc=r['foreground_background']['dice']
        for name,value in (('matrix_without_contrast',f-g),('matrix_with_contrast',fc-gc),
                           ('contrast_with_global_matrix',gc-g),('contrast_with_region_matrix',fc-f),
                           ('interaction',fc-f-gc+g),
                           ('asinh_head_input_vs_raw',r['foreground_background_asinh']['dice']-fc)):
            terms[name].append(value)
    return {'present_frames':len(present),**{k:(float(np.mean(v)) if v else None) for k,v in terms.items()}}


def replay(run, manifest, sha, plan_path, parity_path, output, guard):
    expected=phase_inputs(run,plan_path,parity_path)
    plan=require_plan(plan_path,manifest,sha);require_parity(parity_path,sha,plan_path,plan)
    require([digest(plan_path),digest(parity_path)]==expected,'Plan or parity changed during parsing')
    require(environment_identity()==plan['environment'],'Capture environment differs')
    runs=validated_runs(run,manifest,sha,plan_path,parity_path,plan)
    targets=sorted((x for x in manifest['samples'] if x['split']=='validation'),key=lambda x:(x['sequence'],x['frame']))
    baseline={}
    for arm, record in runs.items():
        net=load_model(record['payload'],arm,manifest,sha).to('mps').eval()
        with torch.no_grad():
            for item in targets:
                args,shape=inputs(item,'mps'); result=net(*args)
                require(torch.isfinite(result).all(),'Nonfinite replay')
                masks={'baseline':(to_video(args[2],shape)>0).cpu().numpy()[0,0],
                       'refined':(to_video(result,shape)>0).cpu().numpy()[0,0]}
                key=(item['sequence'],item['frame']);row=record['mapping'][key]
                require(list(shape)==row['shape'],'Frozen shape differs')
                for kind, predicted in masks.items():
                    path=Path(row[kind]);require(digest(path)==row[kind+'_sha256'],'Frozen mask changed')
                    with Image.open(path) as image: saved=np.asarray(image.convert('L'))>0
                    require(digest(path)==row[kind+'_sha256'] and np.array_equal(saved,predicted),'Mask replay differs')
                if key in baseline: require(baseline[key]==row['baseline_sha256'],'Baseline differs across arms')
                baseline[key]=row['baseline_sha256']
    require(guard['open_attempts']==0,'Label open attempted during replay')
    out=outpath(output);out.mkdir(parents=True)
    result={'passed':True,'scope':'All frozen baseline/refined masks replayed without target labels.',
        'manifest_sha256':sha,'plan_sha256':digest(plan_path),'parity_sha256':digest(parity_path),
        'dependency_sha256':deps(),'label_guard':dict(guard),'environment_flags':require_environment_flags(),
        'runs':{a:{k:r[k] for k in ('completion_sha256','checkpoint_sha256','frozen_sha256')} for a,r in runs.items()}}
    (out/'verification.json').write_text(json.dumps(result,indent=2)+'\n')


def score(run, manifest, sha, plan_path, parity_path, verification, output, guard):
    out=outpath(output)
    expected=phase_inputs(run,plan_path,parity_path)
    plan=require_plan(plan_path,manifest,sha);require_parity(parity_path,sha,plan_path,plan)
    require([digest(plan_path),digest(parity_path)]==expected,'Plan or parity changed during parsing')
    phase_file=Path(run)/'phases.jsonl'
    phases=[json.loads(x) for x in phase_file.read_text().splitlines() if x.strip()]
    require([p['phase'] for p in phases] == ['plan','parity']+['train_'+a for a in ARMS]+['replay'] and
            phases[-1].get('passed') is True and phases[-1].get('verification_sha256')==digest(verification) and
            phases[-1].get('log_sha256')==digest(Path(run)/'replay.log'), 'Replay phase not frozen')
    proof=read(verification)
    require(phases[-1]['verification_sha256']==digest(verification),'Replay receipt changed during parsing')
    require(proof.get('passed') is True and proof.get('manifest_sha256')==sha and
            proof.get('plan_sha256')==digest(plan_path) and proof.get('parity_sha256')==digest(parity_path) and
            proof.get('dependency_sha256')==deps() and proof.get('environment_flags')==plan['environment_flags'] and
            proof.get('label_guard',{}).get('open_attempts')==0 and
            proof.get('label_guard',{}).get('blocked_paths')==sum(map(len,mask_path_sets(ROOT/'data/polypgen_sequence.jsonl'))) and
            set(proof.get('runs',{}))==set(ARMS),'Replay verification differs')
    runs=validated_runs(run,manifest,sha,plan_path,parity_path,plan)
    for arm, record in runs.items():
        require(proof['runs'][arm]=={k:record[k] for k in ('completion_sha256','checkpoint_sha256','frozen_sha256')},
                'Frozen arm changed after replay')
    require(guard['open_attempts']==0,'Validation label opened before frozen verification')
    guard['active']=False
    targets=sorted((x for x in manifest['samples'] if x['split']=='validation'),key=lambda x:(x['sequence'],x['frame']))
    rows=[]
    for item in targets:
        key=(item['sequence'],item['frame']);shape=tuple(runs[ARMS[0]]['mapping'][key]['shape'])
        truth=checked_target(item,shape)[0,0].numpy().astype(bool)
        row={'sequence':key[0],'frame':key[1],'present':bool(truth.any())}
        for arm,record in runs.items():
            saved=record['mapping'][key]
            for kind in ('baseline','refined'):
                name='baseline' if kind=='baseline' else arm
                if name in row: continue
                path=Path(saved[kind]);require(digest(path)==saved[kind+'_sha256'],'Prediction changed before score')
                with Image.open(path) as image: pred=np.asarray(image.convert('L'))>0
                require(digest(path)==saved[kind+'_sha256'],'Prediction changed during score')
                row[name]=metric(pred,truth)
        rows.append(row)
    names=('baseline',*ARMS)
    summaries={name:{'pooled':summary(rows,name),
        'per_video':{seq:summary([r for r in rows if r['sequence']==seq],name) for seq in sorted({r['sequence'] for r in rows})}}
        for name in names}
    for result in summaries.values():
        means=[x['mean_present_dice'] for x in result['per_video'].values() if x['mean_present_dice'] is not None]
        result['equal_video_mean_present_dice']=float(np.mean(means)) if means else None
    effects={'pooled':factorial_effects(rows),
             'per_video':{seq:factorial_effects([r for r in rows if r['sequence']==seq])
                          for seq in sorted({r['sequence'] for r in rows})}}
    out.mkdir(parents=True)
    lines=['# Exploratory fixed-read channel-value comparison','',
      'Reused seq9/15 validation reads; not a live-video or independent test. The region reference has 48,005 parameters; other trained arms have 63,809.','',
      '| Arm | Present-frame Dice | Empty false alarms |', '|---|---:|---:|']
    for name in names:
        s=summaries[name]['pooled'];dice='n/a' if s['mean_present_dice'] is None else f"{s['mean_present_dice']:.4f}"
        lines.append(f"| {name} | {dice} | {s['empty_false_alarms']}/{s['empty_frames']} |")
    lines+=['','Paired factorial Dice effects (present reads only; descriptive, not independent evidence):',
            '', '| Contrast | Mean paired Dice effect |','|---|---:|']
    for name,value in effects['pooled'].items():
        if name!='present_frames':lines.append(f"| {name} | {'n/a' if value is None else f'{value:+.4f}'} |")
    lines+=['','FG/BG uses p² spatial weighting; the diagonal control changes normalization as well as attention mixing. No causal video-effect claim is supported.']
    report=out/'report.md';report.write_text('\n'.join(lines)+'\n')
    result={'passed':True,'scope':'Previously inspected fixed-read development only; region reference is smaller.',
        'manifest_sha256':sha,'plan_sha256':digest(plan_path),'verification_sha256':digest(verification),
        'prelabel_guard':dict(guard),'report_sha256':digest(report),'summaries':summaries,
        'factorial_effects':effects,'rows':rows}
    (out/'scores.json').write_text(json.dumps(result,indent=2)+'\n')


def independent_audit(run,manifest,sha,plan_path,parity_path,verification,output,guard):
    """Separate CPU metrics; shares the frozen structural verifier."""
    from scripts.audit_boundary_video_factorial_v1 import label as independent_label, metrics as independent_metrics, same
    out=outpath(output);run=Path(run)
    expected=phase_inputs(run,plan_path,parity_path)
    plan=require_plan(plan_path,manifest,sha);require_parity(parity_path,sha,plan_path,plan)
    require([digest(plan_path),digest(parity_path)]==expected,'Plan or parity changed')
    phases=[json.loads(x) for x in (run/'phases.jsonl').read_text().splitlines() if x.strip()]
    names=['plan','parity']+['train_'+a for a in ARMS]+['replay','score']
    require([p['phase'] for p in phases]==names and phases[-1].get('passed') is True and
            phases[-1].get('log_sha256')==digest(run/'score.log') and
            phases[-1].get('artifact_sha256')==digest(run/'scores/scores.json') and
            phases[-2].get('verification_sha256')==digest(verification),'Scoring artifacts differ')
    score_sha=phases[-1]['artifact_sha256'];primary=read(run/'scores/scores.json')
    require(digest(run/'scores/scores.json')==score_sha and primary.get('passed') is True and
            primary.get('manifest_sha256')==sha and primary.get('plan_sha256')==digest(plan_path) and
            primary.get('verification_sha256')==digest(verification) and
            primary.get('prelabel_guard',{}).get('open_attempts')==0 and
            primary.get('report_sha256')==digest(run/'scores/report.md'),'Primary score identity differs')
    proof_sha=digest(verification);proof=read(verification)
    require(digest(verification)==proof_sha and proof.get('passed') is True and
            proof.get('dependency_sha256')==deps() and proof.get('label_guard',{}).get('open_attempts')==0,
            'Replay identity differs')
    runs=validated_runs(run,manifest,sha,plan_path,parity_path,plan)
    for arm,record in runs.items():
        require(proof['runs'][arm]=={k:record[k] for k in ('completion_sha256','checkpoint_sha256','frozen_sha256')},
                'Frozen arm changed after replay')
    require(guard['open_attempts']==0,'Label opened before audit verification');guard['active']=False
    targets=sorted((x for x in manifest['samples'] if x['split']=='validation'),key=lambda x:(x['sequence'],x['frame']))
    require(len(primary['rows'])==len(targets),'Primary score frame coverage differs')
    errors=[];count=0;audit_rows=[]
    for item,expected_row in zip(targets,primary['rows'],strict=True):
        key=(item['sequence'],item['frame']);label_path=Path(item['label'])
        require(digest(label_path)==item['label_sha256'],'Label changed before audit')
        truth=independent_label(label_path)
        require(digest(label_path)==item['label_sha256'],'Label changed during audit')
        require((expected_row['sequence'],expected_row['frame'])==key and expected_row['present']==bool(truth.any()),
                'Primary label or frame differs')
        audit_row={'sequence':key[0],'frame':key[1],'present':bool(truth.any())}
        for name in ('baseline',*ARMS):
            arm=ARMS[0] if name=='baseline' else name
            kind='baseline' if name=='baseline' else 'refined'
            row=runs[arm]['mapping'][key];path=Path(row[kind])
            require(digest(path)==row[kind+'_sha256'],'Prediction changed before audit')
            with Image.open(path) as image:pred=np.asarray(image.convert('L'))>0
            require(digest(path)==row[kind+'_sha256'],'Prediction changed during audit')
            value=independent_metrics(pred,truth);value.pop('present');value.pop('iou')
            value['zero_overlap']=(value['dice']==0) if truth.any() else None
            audit_row[name]=value
            for field,v in value.items():
                if not same(v,expected_row[name][field]):errors.append([*key,name,field,v,expected_row[name][field]])
            count+=1
        audit_rows.append(audit_row)
    def independent_summary(items,arm):
        import math
        present=[r[arm] for r in items if r['present']];empty=[r[arm] for r in items if not r['present']]
        def mean(values):
            values=list(values)
            return math.fsum(values)/len(values) if values else None
        precision=[x['precision'] for x in present if x['precision'] is not None]
        return {'present_frames':len(present),'mean_present_dice':mean(x['dice'] for x in present),
            'mean_present_boundary_f_5px':mean(x['boundary_f_5px'] for x in present),
            'present_zero_overlap_frames':sum(x['zero_overlap'] for x in present),
            'mean_present_precision_when_defined':mean(precision),'precision_defined_frames':len(precision),
            'mean_present_recall':mean(x['recall'] for x in present),'empty_frames':len(empty),
            'empty_false_alarms':sum(x['false_alarm'] for x in empty),
            'mean_empty_predicted_pixels':mean(x['predicted_pixels'] for x in empty)}
    for name in ('baseline',*ARMS):
        actual={'pooled':independent_summary(audit_rows,name),
            'per_video':{seq:independent_summary([x for x in audit_rows if x['sequence']==seq],name)
                         for seq in sorted({x['sequence'] for x in audit_rows})}}
        means=[x['mean_present_dice'] for x in actual['per_video'].values() if x['mean_present_dice'] is not None]
        actual['equal_video_mean_present_dice']=sum(means)/len(means) if means else None
        def compare(path,left,right):
            if isinstance(left,dict) and isinstance(right,dict):
                if set(left)!=set(right):errors.append([*path,'keys'])
                for k in left.keys() & right.keys():compare([*path,k],left[k],right[k])
            elif not same(left,right):errors.append([*path,left,right])
        compare([name,'summary'],actual,primary['summaries'][name])
    def audit_effect(items):
        import math
        rows=[x for x in items if x['present']]
        terms={k:[] for k in ('matrix_without_contrast','matrix_with_contrast',
                               'contrast_with_global_matrix','contrast_with_region_matrix','interaction',
                               'asinh_head_input_vs_raw')}
        for r in rows:
            g=r['global']['dice'];gc=r['global_contrast']['dice']
            f=r['foreground_background_no_contrast']['dice'];fc=r['foreground_background']['dice']
            for k,v in (('matrix_without_contrast',f-g),('matrix_with_contrast',fc-gc),
                        ('contrast_with_global_matrix',gc-g),('contrast_with_region_matrix',fc-f),
                        ('interaction',fc-f-gc+g),
                        ('asinh_head_input_vs_raw',r['foreground_background_asinh']['dice']-fc)):terms[k].append(v)
        return {'present_frames':len(rows),**{k:(math.fsum(v)/len(v) if v else None) for k,v in terms.items()}}
    effect={'pooled':audit_effect(audit_rows),'per_video':{
        seq:audit_effect([r for r in audit_rows if r['sequence']==seq])
        for seq in sorted({r['sequence'] for r in audit_rows})}}
    compare(['factorial_effects'],effect,primary['factorial_effects'])
    out.mkdir(parents=True)
    result={'passed':not errors,'scope':'Independent CPU/SciPy per-read metrics; structural verifier shared. Reused validation, not video.',
        'score_sha256':score_sha,'prediction_masks_checked':count,'label_files_checked':len(targets),
        'prelabel_guard':dict(guard),'error_count':len(errors),'errors':errors}
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    require(not errors,'Independent metric audit found discrepancies')


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--mode',choices=('plan','parity','train','replay','score','audit'),required=True)
    p.add_argument('--manifest',type=Path,required=True);p.add_argument('--output',type=Path,required=True)
    p.add_argument('--plan',type=Path);p.add_argument('--parity',type=Path);p.add_argument('--run',type=Path)
    p.add_argument('--verification',type=Path);p.add_argument('--arm',choices=ARMS)
    a=p.parse_args();require_environment_flags()
    outpath(a.output)  # Reject invalid output before any target label can open.
    if a.mode in ('plan','replay'): guard=install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl')
    elif a.mode in ('parity','train'): guard=install_training_mask_guard(ROOT/'data/polypgen_sequence.jsonl')
    else: guard=install_phase_mask_guard(ROOT/'data/polypgen_sequence.jsonl')
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    manifest,sha=load_prepared(a.manifest)
    if a.mode=='plan':
        plan=build_plan(manifest,sha);require(guard['open_attempts']==0,'Label opened during plan')
        out=outpath(a.output);out.mkdir(parents=True);(out/'plan.json').write_text(json.dumps(plan,indent=2)+'\n')
    elif a.mode=='parity':
        require(a.plan is not None,'Parity requires plan');parity(manifest,sha,a.plan,a.output,guard)
        require(guard['blocked_open_attempts']==0,'Protected label opened in parity')
    elif a.mode=='train':
        require(a.plan is not None and a.parity is not None and a.arm is not None,'Training arguments missing')
        train(manifest,sha,a.arm,a.plan,a.parity,a.output,guard)
    elif a.mode=='replay':
        require(a.plan is not None and a.parity is not None and a.run is not None,'Replay arguments missing')
        replay(a.run,manifest,sha,a.plan,a.parity,a.output,guard)
    elif a.mode=='score':
        require(all(x is not None for x in (a.plan,a.parity,a.run,a.verification)),'Scoring arguments missing')
        score(a.run,manifest,sha,a.plan,a.parity,a.verification,a.output,guard)
    else:
        require(all(x is not None for x in (a.plan,a.parity,a.run,a.verification)),'Audit arguments missing')
        independent_audit(a.run,manifest,sha,a.plan,a.parity,a.verification,a.output,guard)


if __name__=='__main__': main()
