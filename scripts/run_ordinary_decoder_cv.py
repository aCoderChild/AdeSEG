"""Gate and evaluate ordinary decoder-core adaptation with native propagation."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import sys
import tempfile
import time
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.sam2_clip_sampler import sample_clip
from datasets.video import load_manifest
from modeling.decoder_adaptation import train_decoder_core
from modeling.decoder_core_training import decoder_clip_loss
from modeling.native_history_cache import capture_cached_block, load_bank
from modeling.native_stage_capture import NativeStageCapture
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import activate_author_repo, prepare
from scripts.run_prediction_conditioned_affinity_cv import SEQUENCES, summarize_arm, verify_folds
from scripts.run_regional_refinement import load, save_tensor, write
from scripts.run_soft_role_affinity_cv import paired_delta, score_frame, summarize
from training.tracking import start_wandb

PROTOCOL='protocols/ordinary_decoder_cv_v1.json'


def schedule(parent, fold):
    settings=parent['ordinary_baseline']
    verify_folds(settings['folds'])
    validation=settings['folds'][fold-1]
    training=[n for n in parent['sequences'] if n not in validation]
    if set(training)&set(validation) or set(parent['sequences'])!=set(SEQUENCES):
        raise RuntimeError('Training/validation partition differs')
    return [(epoch,name,settings['seed']+fold*100000+epoch*len(training)+index)
        for epoch in range(settings['epochs']) for index,name in enumerate(training)]


def restore_core(model, weights):
    parameters=dict(train_decoder_core(model))
    if set(parameters)!=set(weights):
        raise RuntimeError('Saved decoder-core parameter scope differs')
    with torch.no_grad():
        for name,p in parameters.items():
            if weights[name].shape!=p.shape or not torch.isfinite(weights[name]).all():
                raise RuntimeError('Invalid decoder-core weights')
            p.copy_(weights[name])


def evaluate(model, frames, name, reference, output, *, check_native=False):
    from sam2.modeling.position_encoding import PositionEmbeddingSine
    for module in model.modules():
        if isinstance(module,PositionEmbeddingSine):
            module.cache.clear()
    before=parameter_digest(model)
    lookup={r['frame']:r for r in reference}
    rows=[]
    with tempfile.TemporaryDirectory(prefix='ordinary_decoder_',dir=output) as temp, torch.inference_mode():
        state,prompt=prepare(model,frames,Path(temp))
        started=time.monotonic()
        try:
            with NativeStageCapture(model) as observer:
                trajectory=iter(model.propagate_in_video(state,start_frame_idx=prompt))
                for expected in range(prompt,len(frames)):
                    observer.begin(False)
                    frame,ids,logits=next(trajectory)
                    if frame!=expected or ids!=[1] or not torch.isfinite(logits).all():
                        raise RuntimeError('Invalid autonomous trajectory')
                    if frame==prompt:
                        continue
                    layers=observer.layers
                    if ([r['layer'] for r in layers]!=[0,1,2,3] or
                            any((r['keys']-r['pointer_keys'])%4096 for r in layers)):
                        raise RuntimeError('Invalid native layer/token schedule')
                    stored=state['output_dict_per_obj'][0]['non_cond_frame_outputs'][frame]
                    row=score_frame(logits,frames,frame)
                    row.update(stored_sha256=fingerprint(stored)['sha256'],attention_layers=layers,
                        object_score=float(stored['object_score_logits'].item()),
                        prediction_nonempty=bool((logits>0).any()))
                    native=lookup[frame]
                    if check_native and any(row[k]!=native[k] for k in
                            ('prediction_sha256','stored_sha256','scores','present','absent_fp')):
                        raise RuntimeError(f'Fresh decoder replay differs: {name}/{frame}')
                    rows.append(row)
                    write(output/'progress.json',{'phase':'evaluation','sequence':name,'frame':frame})
        finally:
            model.reset_state(state)
        elapsed=time.monotonic()-started
    if [r['frame'] for r in rows]!=list(range(prompt+1,len(frames))) or parameter_digest(model)!=before:
        raise RuntimeError('Incomplete evaluation or mutated weights')
    return rows,{**summarize(rows),'inference_wall_seconds':elapsed}


def result_table(result):
    lines=['# Ordinary decoder-core adaptation — exploratory PolypGen CV','',
        '| Sequence | Native Dice | Adapted Dice | Difference |','|---|---:|---:|---:|']
    for name in SEQUENCES:
        a,b=[result[mode][name]['present_dice'] for mode in ('native','ordinary')]
        lines.append(f'| {name} | {a:.6f} | {b:.6f} | {b-a:+.6f} |')
    lines+=['','| Arm | Mean Dice | IoU | Boundary F | False alarms / empty frames |',
        '|---|---:|---:|---:|---:|']
    for mode,row in result['summary'].items():
        lines.append(f"| {mode} | {row['present_dice']:.6f} | {row['iou']:.6f} | {row['boundary_f']:.6f} | {row['gt_empty_false_alarm_count']} / {row['gt_empty_targets']} |")
    delta=result['ordinary_vs_native']
    lines+=['',f"Mean paired Δ={delta['mean_delta']:+.6f}; 95% sequence-bootstrap CI={delta['ci95']}; Wilcoxon p={delta['wilcoxon']['p']:.6g}.",
        f"Frozen meaningful-gain screen passed: {result['meaningful_signal_screen']}.",
        'Sequence grouping does not establish patient independence. No external confirmation.']
    return '\n'.join(lines)+'\n'


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('native-run','training-gate','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--gate-only',action='store_true')
    parser.add_argument('--propagation-gate',type=Path)
    parser.add_argument('--wandb-mode',choices=['offline','disabled'],default='offline')
    args=parser.parse_args()
    if not args.gate_only and args.propagation_gate is None:
        parser.error('CV requires --propagation-gate')
    out=validate_output(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    run=None
    try:
        parent=json.loads((ROOT/'protocols/encoder_decoder_bottleneck_v1.json').read_text())
        protocol=json.loads((ROOT/PROTOCOL).read_text())
        training_gate=json.loads((args.training_gate/'completion.json').read_text())
        training_audit=json.loads((args.training_gate/'independent_receipt_audit.json').read_text())
        native=json.loads((args.native_run/'completion.json').read_text())
        audit=json.loads((args.native_run/'independent_audit.json').read_text())
        setup=json.loads((args.native_run/'setup.json').read_text())
        if (not training_gate.get('passed') or not training_audit.get('passed') or
                training_audit['completion_sha256']!=digest(args.training_gate/'completion.json') or
                not native.get('passed') or not audit.get('passed') or
                (not args.gate_only and native['scope']!='cohort')):
            raise RuntimeError('Verified native/training prerequisites missing')
        if (training_audit['training_sha256']!=digest(args.training_gate/'training.json') or
                training_audit['replay_sha256']!=digest(args.training_gate/'replay.json')):
            raise RuntimeError('Audited training evidence changed')
        if audit['scope']!=native['scope'] or set(audit['sequences'])!=set(native['entries']):
            raise RuntimeError('Native audit covers different evidence')
        sources={**training_gate['sources'],**{p:digest(ROOT/p) for p in
            [PROTOCOL,'scripts/run_ordinary_decoder_cv.py','tests/test_ordinary_decoder_cv.py']}}
        if any(digest(ROOT/n)!=h for n,h in sources.items()) or any(sources.get(n)!=h for n,h in native['sources'].items()):
            raise RuntimeError('Verified source inventory differs')
        if digest(Path(parent['checkpoint']))!=setup['checkpoint_sha256'] or setup['checkpoint_sha256']!=parent['checkpoint_sha256']:
            raise RuntimeError('Official checkpoint mismatch')
        if digest(ROOT/parent['manifest'])!=setup['manifest_sha256']:
            raise RuntimeError('Manifest mismatch')
        if setup['model_config']!=parent['model_config']:
            raise RuntimeError('Native architecture differs')
        for name,path in native['entries'].items():
            folder=Path(path);entry=audit['sequences'][name]
            if digest(folder/'receipt.json')!=entry['receipt_sha256'] or digest(folder/'completion.json')!=entry['completion_sha256']:
                raise RuntimeError('Native receipt changed')
        if args.propagation_gate:
            gate=json.loads((args.propagation_gate/'completion.json').read_text())
            if not gate.get('passed') or gate['sources']!=sources or not gate.get('engineering_only'):
                raise RuntimeError('Propagation gate missing or sources changed')
            if gate['training_gate_sha256']!=digest(args.training_gate/'completion.json'):
                raise RuntimeError('Propagation gate used different diagnostic weights')
            for path,sha in gate['artifacts'].items():
                if digest(args.propagation_gate/path)!=sha:
                    raise RuntimeError('Propagation evidence changed')
        for path,sha in sources.items():
            target=out/'source_snapshot'/path;target.parent.mkdir(parents=True,exist_ok=True)
            shutil.copyfile(ROOT/path,target)
        write(out/'setup.json',{'sources':sources,'protocol':protocol,'parent':parent,
            'native_completion_sha256':digest(args.native_run/'completion.json'),
            'training_gate_sha256':digest(args.training_gate/'completion.json')})
        torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
        repo,_=activate_author_repo()
        def new_model():
            return build_author_predictor(SimpleNamespace(method='sam2',device='cpu',
                checkpoint=Path(parent['checkpoint']),model_config=parent['model_config']),repo)
        videos={**load_manifest(ROOT/parent['manifest'],'dev'),**load_manifest(ROOT/parent['manifest'],'test')}
        native_rows={};native_summaries={}
        for name,path in native['entries'].items():
            folder=Path(path);complete=json.loads((folder/'completion.json').read_text())
            if digest(folder/'frames.json')!=complete['frames_sha256']:
                raise RuntimeError('Native trajectory changed')
            native_rows[name]=json.loads((folder/'frames.json').read_text())
            native_summaries[name]=summarize(native_rows[name])
        if args.gate_only:
            artifacts={}
            for name in parent['gate_sequences']:
                model=new_model()
                plain,_=evaluate(model,videos[name],name,native_rows[name],out,check_native=True)
                restore_core(model,load(args.training_gate/'decoder_core.pt',training_gate['weights_sha256']))
                adapted,_=evaluate(model,videos[name],name,native_rows[name],out)
                if [r['attention_layers'] for r in plain]!=[r['attention_layers'] for r in adapted]:
                    raise RuntimeError('Adaptation changed spatial/pointer schedules')
                for label,rows in [('native',plain),('diagnostic',adapted)]:
                    filename=f'{name}_{label}.json';write(out/filename,rows);artifacts[filename]=digest(out/filename)
                del model;gc.collect()
            write(out/'completion.json',{'passed':True,'engineering_only':True,'sources':sources,
                'artifacts':artifacts,'training_gate_sha256':digest(args.training_gate/'completion.json'),
                'note':'No efficacy or cross-validation claim; diagnostic weights excluded from CV.'})
            return
        if set(native_rows)!=set(SEQUENCES) or sum(map(len,native_rows.values()))!=2012:
            raise RuntimeError('Full native reference coverage differs')
        for key in ['WANDB_CACHE_DIR','WANDB_CONFIG_DIR','WANDB_DATA_DIR','TMPDIR']:
            folder=out/key.lower();folder.mkdir();os.environ[key]=str(folder)
        run=start_wandb('ade-seg','adenoid-hypertrophy',out.name,args.wandb_mode,None,
            protocol,out,['ordinary-decoder','exploratory-cv'])
        settings=parent['ordinary_baseline'];verify_folds(settings['folds'])
        results={'native':native_summaries,'ordinary':{}}
        for fold,validation in enumerate(settings['folds'],1):
            fold_dir=out/f'fold{fold}';fold_dir.mkdir()
            torch.manual_seed(settings['seed']+fold)
            model=new_model();selected=train_decoder_core(model)
            if sum(p.numel() for n,p in selected)!=settings['trainable_parameters']:
                raise RuntimeError('Trainable scope differs')
            frozen=lambda:{n:p for n,p in model.named_parameters() if not p.requires_grad}
            frozen_sha=fingerprint(frozen())['sha256']
            optimizer=torch.optim.AdamW([p for n,p in selected],lr=settings['learning_rate'],weight_decay=.01)
            updates=schedule(parent,fold)
            if len(updates)!=settings['updates_per_fold']:
                raise RuntimeError('Training schedule length differs')
            logs=[]
            for epoch,name,seed in updates:
                folder=Path(native['entries'][name]);receipt=json.loads((folder/'receipt.json').read_text())
                bank=load_bank(folder/'bank.pt',receipt,setup['checkpoint_sha256'],setup['manifest_sha256'],setup['model_config'])
                clip=sample_clip(videos[name],settings['clip_length'],seed)
                samples,record=capture_cached_block(model,videos[name],[f.frame_index for f in clip],out,bank)
                optimizer.zero_grad(set_to_none=True)
                with NativeStageCapture(model) as observer:
                    loss,_=decoder_clip_loss(model,samples)
                if not torch.isfinite(loss) or [r['layer'] for r in observer.layers]!=[0,1,2,3]*len(samples):
                    raise RuntimeError('Nonfinite loss or incomplete training reads')
                loss.backward()
                gradients=[p.grad for n,p in selected if p.grad is not None]
                if not gradients or any(not torch.isfinite(g).all() for g in gradients):
                    raise RuntimeError('Invalid decoder gradients')
                norm=torch.nn.utils.clip_grad_norm_([p for n,p in selected],1.0)
                if not torch.isfinite(norm):
                    raise RuntimeError('Invalid decoder gradient norm')
                optimizer.step()
                if any(not torch.isfinite(p).all() for n,p in selected) or fingerprint(frozen())['sha256']!=frozen_sha:
                    raise RuntimeError('Nonfinite update or frozen-weight mutation')
                logs.append({'epoch':epoch,'sequence':name,'seed':seed,'capture':record,
                    'loss':float(loss.detach()),'gradient_norm':float(norm),'layers':observer.layers})
                write(fold_dir/'training.json',logs)
                write(out/'progress.json',{'phase':'training','fold':fold,'update':len(logs),'total':len(updates)})
                run.log({f'fold{fold}/loss':logs[-1]['loss'],f'fold{fold}/gradient_norm':float(norm)})
                del loss,samples,bank,gradients;gc.collect()
            weights={n:p.detach().clone() for n,p in selected};save_tensor(weights,fold_dir/'decoder_core.pt')
            receipt={'weights_sha256':digest(fold_dir/'decoder_core.pt'),'training_sha256':digest(fold_dir/'training.json'),
                'updates':len(logs),'validation':validation,'training':sorted({n for e,n,s in updates}),
                'frozen_sha256':frozen_sha,'selection':'Final epoch, anchored before validation'}
            write(fold_dir/'training_completion.json',receipt)
            for name in validation:
                rows,summary=evaluate(model,videos[name],name,native_rows[name],out)
                results['ordinary'][name]=summary
                path=fold_dir/f'{name}_frames.json';write(path,rows)
                write(fold_dir/f'{name}_completion.json',{'frames_sha256':digest(path),'summary':summary,
                    'weights_sha256':receipt['weights_sha256'],'training_receipt_sha256':digest(fold_dir/'training_completion.json')})
            write(fold_dir/'completion.json',{'complete':True,**receipt})
            del model,optimizer,weights,selected;gc.collect()
        result={**results,'summary':{mode:{**summarize_arm(values),
            'gt_empty_targets':sum(r['gt_empty_targets'] for r in values.values())} for mode,values in results.items()}}
        delta=paired_delta(results,'ordinary','native',settings['seed']);result['ordinary_vs_native']=delta
        result['meaningful_signal_screen']=(delta['mean_delta']>=.01 and delta['wins']>=14 and
            delta['ci95'][0]>0 and delta['wilcoxon']['p']<.05 and
            result['summary']['ordinary']['gt_empty_false_alarm_count']<=result['summary']['native']['gt_empty_false_alarm_count'])
        if any(digest(ROOT/n)!=h for n,h in sources.items()):
            raise RuntimeError('Source changed during CV')
        write(out/'results.json',result);(out/'results.md').write_text(result_table(result))
        write(out/'completion.json',{'complete':True,'sources':sources,'results_sha256':digest(out/'results.json')})
        run.finish();run=None
    except BaseException as error:
        write(out/'failure.json',{'error':repr(error)})
        if run is not None:
            run.finish(exit_code=1)
        raise


if __name__=='__main__':
    main()
