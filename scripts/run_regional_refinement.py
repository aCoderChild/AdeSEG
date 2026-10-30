"""Frozen, Drive-only UACANet-derived SAM2 refinement verification and pilot."""
import argparse
import gc
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest, stage_frames
from evaluation.segmentation import calculate_scores
from evaluation.davis2017.metrics import db_eval_boundary
from modeling.regional_refinement import RegionalRefinement, refinement_loss, UACA_REVISION
from scripts.refinement_capture import RefinementCapture
from scripts import run_author_baseline as author
from scripts.bank_integrity import fingerprint
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import validate_output, prompt_index
from training.tracking import start_wandb, finalize_tracking

PROTOCOL='protocols/regional_refinement_pilot_v1.json'
SOURCES=[PROTOCOL,'modeling/regional_refinement.py','scripts/refinement_capture.py',
         'scripts/run_regional_refinement.py','tests/test_regional_refinement.py',
         'scripts/run_author_baseline.py','scripts/bank_integrity.py','modeling/region_definitions.py',
         'modeling/region_tokens.py','modeling/region_bias.py','evaluation/segmentation.py',
         'evaluation/davis2017/metrics.py','scripts/false_alarm_statistics.py','training/tracking.py']
ARMS=('generic','regional','shuffled')
CORE=('prediction_sha256','low_res_sha256','bank_sha256','cache_sha256')


def write(path,value):
    path.write_text(json.dumps(value,indent=2)+'\n')


def load(path):
    return torch.load(path,map_location='cpu',weights_only=True)


def batch(paths,arm):
    samples=[load(p) for p in paths]
    return (torch.stack([s['query'] for s in samples]),torch.stack([s['contexts'][arm] for s in samples]),
            torch.stack([s['valid'][arm] for s in samples]),torch.stack([s['native'] for s in samples]),
            torch.tensor([s['object_score'] for s in samples])),torch.stack([s['target'] for s in samples])


def check_sources(hashes):
    if any(author.digest(ROOT/n)!=sha for n,sha in hashes.items()):raise RuntimeError('Frozen source changed')


def capture(args,name,frames,selected,gate=False):
    folder=args.output_dir/'cache'/name;folder.mkdir(parents=True)
    reference=args.reference_dir/f'full_{name}_C0.zip'
    with zipfile.ZipFile(reference) as z:
        archived={r['frame']:r for r in map(json.loads,z.read('integrity.jsonl').decode().splitlines())}
    prompt=prompt_index(frames,{'initialization':'first_labeled_frame'});model=author.build_author_predictor(args,args.repo)
    parameter_hashes=author.parameter_fingerprints(model)
    rows=[];index=[];layer_calls=[]
    handles=[l.cross_attn_image.register_forward_hook(lambda m,a,o,i=i:layer_calls.append(i)) for i,l in enumerate(model.memory_attention.layers)]
    torch.manual_seed(20261031);head=RegionalRefinement().eval()
    probe=RegionalRefinement().eval()
    with torch.no_grad():probe.head.conv_out4.conv.weight.fill_(0.001)
    with tempfile.TemporaryDirectory(prefix='frames_',dir=args.output_dir) as temp, torch.inference_mode():
        stage_frames(frames,Path(temp))
        state=model.init_state(video_path=temp,offload_video_to_cpu=True,offload_state_to_cpu=False)
        model.add_new_mask(state,frame_idx=prompt,obj_id=1,mask=torch.from_numpy(load_label_mask(frames[prompt].mask_path,[1])==1))
        with RefinementCapture(model,name) as observer:
            iterator=model.propagate_in_video(state,start_frame_idx=prompt)
            for frame in range(prompt,len(frames)):
                layer_calls.clear();actual,ids,logits=next(iterator)
                if actual!=frame or ids!=[1]:raise RuntimeError('Chronology/object mismatch')
                outputs=state['output_dict_per_obj'][0]
                stored=outputs['cond_frame_outputs'].get(frame,outputs['non_cond_frame_outputs'].get(frame))
                core={'prediction_sha256':author.tensor_hash(logits),'low_res_sha256':author.tensor_hash(stored['pred_masks']),
                      'bank_sha256':fingerprint(state['output_dict_per_obj'])['sha256'],
                      'cache_sha256':fingerprint(state['cached_features'])['sha256']}
                if any(core[k]!=archived[frame][k] for k in CORE):raise RuntimeError(f'Native reference mismatch {name}/{frame}')
                if frame==prompt:continue
                if layer_calls!=list(range(4)):raise RuntimeError('Missing four memory layers')
                sample=observer.sample
                if sample is None or sample['query'].shape!=(256,64,64):raise RuntimeError('Missing actual decoder input')
                sample.update(native=stored['pred_masks'][0].float().clone(),object_score=float(stored['object_score_logits'].item()))
                truth=load_label_mask(frames[frame].mask_path,[1])==1
                sample['target']=F.interpolate(torch.from_numpy(truth.astype(np.float32))[None,None],size=(256,256),mode='nearest')[0]
                if gate:
                    for arm in ARMS:
                        inputs=(sample['query'][None],sample['contexts'][arm][None],sample['valid'][arm][None],sample['native'][None],torch.tensor([sample['object_score']]))
                        zero=head(*inputs)
                        if not torch.equal(zero,stored['pred_masks']):raise RuntimeError('Zero head lowres mismatch')
                        _,original=model._get_orig_video_res_output(state,zero)
                        if not torch.equal(original,logits):raise RuntimeError('Zero head full-resolution mismatch')
                        if not torch.isfinite(probe(*inputs)).all():raise RuntimeError('Nonfinite probe')
                    if fingerprint(state['output_dict_per_obj'])['sha256']!=core['bank_sha256'] or fingerprint(state['cached_features'])['sha256']!=core['cache_sha256']:
                        raise RuntimeError('Refinement changed native state')
                predicted=logits[0,0].numpy()>0
                scores={**calculate_scores(predicted,truth),'boundary_f':float(db_eval_boundary(predicted,truth))}
                rows.append({'frame':frame,'present':bool(truth.any()),'scores':scores,'absent_fp':bool(predicted.any()) if not truth.any() else None,
                             'object_score':sample['object_score'],'prediction_sha256':core['prediction_sha256']})
                with (folder/'integrity.jsonl').open('a') as s:s.write(json.dumps({'frame':frame,**core,'memory_frames':sample['memory_frames'],'spatial_keys':sample['spatial_keys'],'pointer_keys':sample['pointer_keys'],'layers':layer_calls})+'\n')
                if frame in selected:
                    path=folder/f'{frame:06d}.pt';torch.save(sample,path)
                    index.append({'sequence':name,'frame':frame,'path':str(path),'sha256':author.digest(path),
                                  'mask':str(frames[frame].mask_path),'object_score':sample['object_score']})
                write(args.output_dir/'progress.json',{'phase':'capture_gate' if gate else 'capture','sequence':name,'frame':frame,'last':len(frames)-1})
                if frame%10==0:print(f'[capture] {name}/{frame}',flush=True)
            try:next(iterator);raise RuntimeError('Extra native frame')
            except StopIteration:pass
        model.reset_state(state)
    for h in handles:h.remove()
    author.assert_parameter_integrity(model,parameter_hashes,folder,'refinement')
    write(folder/'native_frames.json',rows)
    write(folder/'capture.json',{'complete':True,'gate':gate,'targets':len(rows),'saved_samples':len(index),'reference_sha256':author.digest(reference)})
    del model,head,probe;gc.collect()
    return index,rows


def overfit(args,index):
    selected=[r for r in index if r['object_score']>0][:2]
    if len(selected)!=2:raise RuntimeError('Need two enabled training samples')
    torch.manual_seed(20261031);head=RegionalRefinement().train()
    inputs,target=batch([r['path'] for r in selected],'regional')
    optimizer=torch.optim.Adam(head.parameters(),lr=.001)
    losses=[];gradient_seen=False
    for step in range(500):
        optimizer.zero_grad();prediction=head(*inputs);loss=refinement_loss(prediction,target)
        if not torch.isfinite(loss):raise RuntimeError('Nonfinite overfit loss')
        loss.backward()
        if step>0:
            gradients=[p.grad for n,p in head.named_parameters() if ('conv_query' in n or 'conv_key' in n or 'conv_value' in n) and p.grad is not None]
            gradient_seen |= any(bool(g.abs().sum()>0) for g in gradients)
        optimizer.step();losses.append(float(loss.detach()))
        if step%50==0:print(f'[overfit] step={step} loss={losses[-1]:.6f}',flush=True)
    passed=gradient_seen and losses[-1]<=losses[0]*.5
    write(args.output_dir/'overfit.json',{'passed':passed,'gradient_seen':gradient_seen,'samples':selected,'losses':losses,'scope':'training-only debugging, not efficacy'})
    with torch.no_grad():
        fitted=head(*inputs)
        for i in range(2):
            panels=[target[i,0],inputs[3][i,0]>0,fitted[i,0]>0]
            image=np.concatenate([p.float().numpy()*255 for p in panels],axis=1).astype(np.uint8)
            Image.fromarray(image).save(args.output_dir/f'overfit_{i}_gt_native_fitted.png')
    if not passed:raise RuntimeError('Frozen tiny-overfit gate failed; do not launch pilot')


def evaluate(args,head,arm,index,native):
    head.eval();rows={};latencies=[]
    with torch.no_grad():
        for row in index:
            inputs,_=batch([row['path']],arm)
            start=time.perf_counter();output=head(*inputs);latencies.append(time.perf_counter()-start)
            truth=load_label_mask(Path(row['mask']),[1])==1
            predicted=F.interpolate(output,size=truth.shape,mode='bilinear',align_corners=False)[0,0].numpy()>0
            score={**calculate_scores(predicted,truth),'boundary_f':float(db_eval_boundary(predicted,truth))}
            rows.setdefault(row['sequence'],[]).append({'frame':row['frame'],'present':bool(truth.any()),'scores':score,'absent_fp':bool(predicted.any()) if not truth.any() else None})
    summaries={}
    for name,rs in rows.items():
        present=[r for r in rs if r['present']]
        summaries[name]={**propagation_summary(rs),'boundary_f':float(np.mean([r['scores']['boundary_f'] for r in present])),
                         'iou':float(np.mean([r['scores']['iou'] for r in present]))}
    return {'sequences':summaries,'mean_dice':float(np.mean([s['present_dice'] for s in summaries.values()])),
            'false_alarms':sum(s['gt_empty_false_alarm_count'] for s in summaries.values()),
            'head_seconds_per_frame':float(np.mean(latencies)),'parameters':sum(p.numel() for p in head.parameters()),'frames':rows}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('checkpoint','reference-dir','output-dir'):
        parser.add_argument('--'+name,type=Path,required=True)
    parser.add_argument('--manifest',type=Path,default=ROOT/'data/polypgen_sequence.jsonl')
    args=parser.parse_args();args.output_dir=validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True,exist_ok=False)
    for k in ('TMPDIR','WANDB_CACHE_DIR','WANDB_DATA_DIR'):
        os.environ[k]=str(args.output_dir/k.lower());Path(os.environ[k]).mkdir()
    args.method,args.device='sam2','cpu';args.repo,revision=author.source_identity('sam2');sys.path.insert(0,str(args.repo))
    torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    protocol=json.loads((ROOT/PROTOCOL).read_text())
    subprocess.run(['git','ls-files','--error-unmatch',*SOURCES],cwd=ROOT,check=True,capture_output=True)
    subprocess.run(['git','diff','--exit-code','HEAD','--',*SOURCES],cwd=ROOT,check=True,capture_output=True)
    hashes={n:author.digest(ROOT/n) for n in SOURCES}
    for n in ('external/UACANet/lib/modules/attention_module.py','external/UACANet/lib/modules/layers.py',
              'external/UACANet/LICENSE','external/sam2/training/loss_fns.py'):
        hashes[n]=author.digest(ROOT/n)
    reference_setup=json.loads((args.reference_dir/'setup.json').read_text())
    for key,path in [('checkpoint_sha256',args.checkpoint),('manifest_sha256',args.manifest)]:
        if reference_setup['E2_setup'][key]!=author.digest(path):raise RuntimeError('Native checkpoint/manifest changed')
    source31=args.reference_dir.parent/'31_v2_writeback_control_cpu_v1/run'
    if not json.loads((source31/'validation.json').read_text())['passed']:raise RuntimeError('Source31 incomplete')
    videos={**load_manifest(args.manifest,'dev'),**load_manifest(args.manifest,'test')}
    train=protocol['split']['train_sequences'];validation=protocol['split']['validation_sequences']
    input_receipts={name:[{'frame':f.frame_index,'image_sha256':author.digest(f.image_path),'mask_sha256':author.digest(f.mask_path)} for f in videos[name]] for name in train+validation}
    train_hashes={r['image_sha256'] for n in train for r in input_receipts[n]}
    overlap={r['image_sha256'] for n in validation for r in input_receipts[n]} & train_hashes
    for name, receipts in input_receipts.items():
        old_path=args.reference_dir.parent/'16_per_token_attention_polypgen_cpu_v1/full'/name/'inputs.json'
        old=json.loads(old_path.read_bytes())
        if [{k:r[k] for k in ('frame','image_sha256','mask_sha256')} for r in old] != receipts:
            raise RuntimeError('Original native image/GT receipts changed')
    write(args.output_dir/'inputs.json',input_receipts)
    if overlap:raise RuntimeError(f'{len(overlap)} exact image duplicates across splits')
    setup={'protocol':protocol,'sources':hashes,'author_revision':revision,'uaca_revision':UACA_REVISION,
           'checkpoint_sha256':author.digest(args.checkpoint),'manifest_sha256':author.digest(args.manifest),
           'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),'exact_image_overlap':0}
    write(args.output_dir/'setup.json',setup)
    run=start_wandb('ade-seg','adenoid-hypertrophy','32-protected regional refinement','online','https://forge.coreweave.com/api/wandb',setup,args.output_dir,['exploratory','refinement','uacanet'])
    try:
        indexes=[];native={}
        for name in train+validation:
            check_sources(hashes)
            if shutil.disk_usage(args.output_dir).free<4*1024**3:raise RuntimeError('Less than4GiB free')
            frames=videos[name];prompt=prompt_index(frames,{'initialization':'first_labeled_frame'});all_targets=list(range(prompt+1,len(frames)))
            selected=all_targets if name in validation else [all_targets[i] for i in np.linspace(0,len(all_targets)-1,min(16,len(all_targets)),dtype=int)]
            idx,rows=capture(args,name,frames,selected,gate=name in ('seq2','seq3'))
            indexes+=idx;native[name]=rows;write(args.output_dir/'cache_index.json',indexes)
            if name=='seq3':
                write(args.output_dir/'engineering_gate.json',{'passed':True,'sequences':['seq2','seq3'],'native_exact':True,'zero_head_exact':True,'nonzero_read_only':True,'all4layers':True})
                overfit(args,indexes)
        training=[r for r in indexes if r['sequence'] in train and r['object_score']>0]
        testing=[r for r in indexes if r['sequence'] in validation]
        if not training or not testing:raise RuntimeError('Empty split')
        if any(author.digest(r['path'])!=r['sha256'] for r in indexes):
            raise RuntimeError('Captured cache hash mismatch before training')
        write(args.output_dir/'training_samples.json',training)
        # Train once per fixed seed/arm; use final checkpoint only.
        results=[]
        for seed in protocol['training']['seeds']:
            for arm in ARMS:
                check_sources(hashes);torch.manual_seed(seed);rng=np.random.default_rng(seed)
                head=RegionalRefinement().train();optimizer=torch.optim.Adam(head.parameters(),lr=.001)
                losses=[]
                for step in range(protocol['training']['steps']):
                    selected=rng.choice(len(training),size=4,replace=True)
                    inputs,target=batch([training[i]['path'] for i in selected],arm)
                    optimizer.zero_grad();loss=refinement_loss(head(*inputs),target)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite pilot loss')
                    loss.backward();optimizer.step();losses.append(float(loss.detach()))
                    if step%50==0:
                        print(f'[train] {arm} seed={seed} step={step} loss={losses[-1]:.6f}',flush=True)
                        write(args.output_dir/'progress.json',{'phase':'training','arm':arm,'seed':seed,'step':step})
                destination=args.output_dir/f'{arm}_{seed}';destination.mkdir()
                torch.save(head.state_dict(),destination/'head.pt');write(destination/'losses.json',losses)
                evaluation=evaluate(args,head,arm,testing,native)
                write(destination/'results.json',evaluation)
                results.append({'arm':arm,'seed':seed,**{k:v for k,v in evaluation.items() if k!='frames'}})
                write(args.output_dir/'partial_results.json',results)
        baseline={n:propagation_summary(native[n]) for n in validation}
        native_mean=float(np.mean([v['present_dice'] for v in baseline.values()]))
        write(args.output_dir/'results.json',{'completed':True,'exploratory':True,'native':baseline,'native_mean_dice':native_mean,'conditions':results})
        lines=['# Protected regional refinement pilot','','Exploratory validation; final checkpoints, no tuning. Native presence decision and memory retained.','',
               '| Condition | Seed | Mean present Dice | False alarms |','|---|---:|---:|---:|',
               f"| Native | — | {native_mean:.6f} | {sum(v['gt_empty_false_alarm_count'] for v in baseline.values())} |"]
        for r in results:lines.append(f"| {r['arm']} | {r['seed']} | {r['mean_dice']:.6f} | {r['false_alarms']} |")
        lines+=['','| Sequence | Native | Generic mean | Regional mean | Shuffled mean |','|---|---:|---:|---:|---:|']
        for n in validation:
            values=[np.mean([r['sequences'][n]['present_dice'] for r in results if r['arm']==a]) for a in ARMS]
            lines.append(f"| {n} | {baseline[n]['present_dice']:.6f} | "+' | '.join(f'{v:.6f}' for v in values)+' |')
        (args.output_dir/'results.md').write_text('\n'.join(lines)+'\n')
        finalize_tracking(run,args.output_dir,{'completed':True},['results.json','results.md','setup.json','engineering_gate.json','overfit.json'])
        run.finish()
    except BaseException as error:
        write(args.output_dir/'failure.json',{'error':repr(error)});run.finish(exit_code=1);raise

if __name__ == '__main__':
    main()
