"""Sequence-grouped development CV using existing native capture and correction."""
import argparse
import json
from pathlib import Path
import shutil
import subprocess
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from modeling.current_frame_refinement import CurrentFrameRefinement
from modeling.regional_refinement import refinement_loss
from scripts.run_regional_refinement import batch, capture, evaluate, write, check_sources, load, save_tensor
from scripts.run_author_baseline import source_identity
from scripts.memory_token_io import digest, validate_output, prompt_index
from scripts.false_alarm_statistics import propagation_summary
from evaluation.statistics import sequence_bootstrap_interval, paired_wilcoxon

PROTOCOL = 'protocols/current_frame_cv_v1.json'
COMPATIBILITY = 'protocols/current_frame_repair_compatibility_v1.json'
FILES = [PROTOCOL, COMPATIBILITY, 'scripts/run_current_frame_cv.py', 'tests/test_current_frame_cv.py',
         'modeling/current_frame_refinement.py', 'evaluation/statistics.py']


def check_compatible_sources(hashes):
    """Permit only the exact reviewed old-to-new repair hashes; never wildcard."""
    amendment=json.loads((ROOT/COMPATIBILITY).read_text())
    for name, expected in hashes.items():
        actual=digest(ROOT/name)
        if actual==expected:
            continue
        permitted=amendment['repairs'].get(name)
        if permitted!={'before':expected,'after':actual}:
            raise RuntimeError(f'Unreviewed source mismatch: {name}')


def validate_folds(folds):
    names = [n for fold in folds for n in fold]
    expected = {f'seq{i}' for i in range(1,24) if i not in (1,7)}
    if len(folds)!=3 or any(len(f)!=7 for f in folds) or len(names)!=len(set(names)) or set(names)!=expected:
        raise ValueError('Folds must partition all21 eligible sequences exactly once')
    return sorted(expected, key=lambda n:int(n[3:]))


def sampled_targets(targets):
    if not targets:
        return []
    return [targets[i] for i in np.linspace(0,len(targets)-1,min(16,len(targets)),dtype=int)]


def cleanup_new_tensors(index, folder):
    for row in index:
        path = Path(row['path']).resolve()
        if not path.is_relative_to(folder.resolve()) or path.suffix!='.pt':
            raise RuntimeError('Refusing to remove a non-temporary tensor')
    for row in index:
        Path(row['path']).unlink()
    write(folder/'temporary_tensors_removed.json', {'count':len(index),'reason':'metrics and hashes retained; bounded Drive storage'})


def verified_sequence_result(folder, sequence, targets):
    """Accept a completed sequence only with matching checkpoint and frame set."""
    result_path=folder/f'{sequence}_results.json'
    receipt=json.loads((folder/f'{sequence}_complete.json').read_text())
    if receipt['checkpoint_sha256']!=digest(folder/'head.pt') or receipt['results_sha256']!=digest(result_path):
        raise RuntimeError(f'Resume result integrity mismatch: {sequence}')
    scores=json.loads(result_path.read_text())
    if set(scores['frames'])!={sequence} or set(scores['sequences'])!={sequence}:
        raise RuntimeError(f'Resume sequence mismatch: {sequence}')
    if [r['frame'] for r in scores['frames'][sequence]]!=targets:
        raise RuntimeError(f'Resume frame coverage mismatch: {sequence}')
    return scores


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    for n in ('source','baseline','reference-dir','checkpoint','output-dir'):
        parser.add_argument('--'+n,type=Path,required=True)
    parser.add_argument('--gate',action='store_true')
    parser.add_argument('--resume',action='store_true',help='Resume only a run with matching source hashes and integrity receipts')
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    if args.resume and args.gate:
        parser.error('Gate runs must start fresh')
    if args.resume and not (args.output_dir/'setup.json').is_file():
        parser.error('Resume requires an existing setup receipt')
    args.output_dir.mkdir(parents=True,exist_ok=args.resume)
    out = args.output_dir
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        protocol = json.loads((ROOT/PROTOCOL).read_text())
        folds = protocol['validation_folds']
        names = validate_folds(folds)
        prior = json.loads((args.source/'setup.json').read_text())
        check_compatible_sources(prior['sources'])
        if not json.loads((args.baseline/'completion.json').read_text())['complete']:
            raise RuntimeError('Source34 incomplete')
        baseline_setup = json.loads((args.baseline/'setup.json').read_text())
        check_compatible_sources(baseline_setup['source_hashes'])
        subprocess.run(['git','diff','--exit-code','HEAD','--',*FILES],cwd=ROOT,check=True)
        subprocess.run(['git','ls-files','--error-unmatch',*FILES],cwd=ROOT,check=True,capture_output=True)
        hashes = {f:digest(ROOT/f) for f in FILES}
        if args.resume:
            previous=json.loads((out/'setup.json').read_text())
            if previous['hashes']!=hashes or previous['protocol']!=protocol or previous['source32_setup_sha256']!=digest(args.source/'setup.json'):
                raise RuntimeError('Resume source/protocol mismatch; do not mix old and repaired runs')
            if (out/'completion.json').exists():
                raise RuntimeError('Run is already complete')
        manifest = ROOT/'data/polypgen_sequence.jsonl'
        if digest(args.checkpoint)!=prior['checkpoint_sha256'] or digest(manifest)!=prior['manifest_sha256']:
            raise RuntimeError('Checkpoint/manifest mismatch')
        videos = {**load_manifest(manifest,'dev'),**load_manifest(manifest,'test')}
        inputs = json.loads((args.source/'inputs.json').read_text())
        seen = {}
        for n in names:
            for frame, receipt in zip(videos[n], inputs[n], strict=True):
                if frame.frame_index!=receipt['frame'] or digest(frame.image_path)!=receipt['image_sha256'] or digest(frame.mask_path)!=receipt['mask_sha256']:
                    raise RuntimeError('Original input changed')
                owner = seen.setdefault(receipt['image_sha256'],n)
                if owner!=n:
                    raise RuntimeError(f'Exact image shared by {owner} and {n}; folds must be amended')
            write(out/'progress.json',{'phase':'input_integrity','sequence':n})
        write(out/'setup.json',{'protocol':protocol,'hashes':hashes,'commit':subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'source32_setup_sha256':digest(args.source/'setup.json'),'patient_independence_verified':False,'exact_cross_sequence_overlap':0})
        args.method,args.device = 'sam2','cpu'
        args.repo,_ = source_identity('sam2')
        sys.path.insert(0,str(args.repo))
        if args.gate:
            n='seq2'
            prompt=prompt_index(videos[n],{'initialization':'first_labeled_frame'})
            idx, native=capture(args,n,videos[n],list(range(prompt+1,len(videos[n]))),gate=True)
            write(out/'new_cache_index.json',idx)
            torch.manual_seed(20261031)
            head=CurrentFrameRefinement().eval()
            evaluated=evaluate(args,head,'regional',idx,None)
            for a,b in zip(evaluated['frames'][n],native,strict=True):
                if a['frame']!=b['frame'] or any(abs(a['scores'][k]-b['scores'][k])>1e-9 for k in ('dice','iou','boundary_f')):
                    raise RuntimeError('Current-only zero-head full-sequence replay failed')
            write(out/'gate_results.json',evaluated)
            cleanup_new_tensors(idx,out/'cache'/n)
            write(out/'completion.json',{'complete':True,'gate':True,'hashes':hashes,'targets':len(idx)})
            return
        gate=json.loads((out.parent/'gate/completion.json').read_text())
        if not gate['complete'] or not gate['gate'] or gate['hashes']!=hashes:
            raise RuntimeError('Matching real-data gate required')
        index=json.loads((args.source/'cache_index.json').read_text())
        train_pool=[]
        native_rows={}
        for n in names:
            native_rows[n]=json.loads((args.source/'cache'/n/'native_frames.json').read_text())
            chosen=set(sampled_targets([r['frame'] for r in native_rows[n]]))
            available={r['frame']:r for r in index if r['sequence']==n}
            if not chosen.issubset(available):
                raise RuntimeError('Original training sample cache missing')
            for f in sorted(chosen):
                r=available[f]
                if digest(r['path'])!=r['sha256']:
                    raise RuntimeError('Training cache mismatch')
                if r['object_score']>0:train_pool.append(r)
        all_sequences={}
        for fi, valnames in enumerate(folds,1):
            check_sources(hashes)
            folder=out/f'fold{fi}';folder.mkdir(exist_ok=args.resume)
            train=[r for r in train_pool if r['sequence'] not in valnames]
            training_receipt=folder/'training_complete.json'
            torch.manual_seed(20261031)
            head=CurrentFrameRefinement().train()
            if args.resume and training_receipt.exists():
                receipt=json.loads(training_receipt.read_text())
                if json.loads((folder/'training_samples.json').read_text())!=train:
                    raise RuntimeError('Resume training samples changed')
                if digest(folder/'losses.json')!=receipt['losses_sha256']:
                    raise RuntimeError('Resume losses changed')
                head.load_state_dict(load(folder/'head.pt',receipt['checkpoint_sha256']),strict=True)
            else:
                # No final receipt: retrain deterministically from initialization.
                if args.resume and list(folder.glob('*_results.json')):
                    raise RuntimeError('Saved scores without a verified training receipt')
                write(folder/'training_samples.json',train)
                optimizer=torch.optim.Adam(head.parameters(),lr=.001)
                rng=np.random.default_rng(20261031)
                losses=[]
                for step in range(500):
                    selected=rng.choice(len(train),size=4,replace=True)
                    chosen=[train[i] for i in selected]
                    x,y=batch([r['path'] for r in chosen],'regional',[r['sha256'] for r in chosen])
                    optimizer.zero_grad();loss=refinement_loss(head(*x),y)
                    if not torch.isfinite(loss):raise RuntimeError('Nonfinite CV loss')
                    loss.backward();optimizer.step();losses.append(float(loss.detach()))
                    if step%10==0:write(out/'progress.json',{'phase':'training','fold':fi,'step':step+1,'total':500})
                save_tensor(head.state_dict(),folder/'head.pt');write(folder/'losses.json',losses)
                write(training_receipt,{'checkpoint_sha256':digest(folder/'head.pt'),'losses_sha256':digest(folder/'losses.json')})
            for n in valnames:
                result_path=folder/f'{n}_results.json'
                receipt_path=folder/f'{n}_complete.json'
                if args.resume and receipt_path.exists():
                    scores=verified_sequence_result(folder,n,[r['frame'] for r in native_rows[n]])
                    all_sequences[n]=scores['sequences'][n]
                    continue
                existing=sorted([r for r in index if r['sequence']==n],key=lambda r:r['frame'])
                targets=[r['frame'] for r in native_rows[n]]
                fresh=[r['frame'] for r in existing]!=targets
                if fresh:
                    if shutil.disk_usage(out).free < 4*1024**3+len(targets)*6*1024**2:
                        raise RuntimeError('Insufficient Drive-backed disk space for sequence capture')
                    cache_index=folder/f'{n}_cache_index.json'
                    if args.resume and cache_index.exists():
                        existing=json.loads(cache_index.read_text())
                        if [r['frame'] for r in existing]!=targets or any(r['sequence']!=n for r in existing):
                            raise RuntimeError(f'Resume cache coverage mismatch: {n}')
                    else:
                        if (out/'cache'/n).exists():
                            raise RuntimeError(f'Incomplete capture without index: {n}; inspect before recapture')
                        existing,_=capture(args,n,videos[n],targets,gate=False)
                        write(cache_index,existing)
                for r in existing:
                    if digest(r['path'])!=r['sha256']:raise RuntimeError('Validation cache mismatch')
                write(out/'progress.json',{'phase':'validation','fold':fi,'sequence':n})
                scores=evaluate(args,head,'regional',existing,None)
                write(result_path,scores)
                write(receipt_path,{'results_sha256':digest(result_path),'checkpoint_sha256':digest(folder/'head.pt')})
                all_sequences[n]=scores['sequences'][n]
                write(out/'partial_sequences.json',all_sequences)
                if fresh:cleanup_new_tensors(existing,out/'cache'/n)
            write(folder/'completion.json',{'complete':True,'validation_sequences':valnames,'checkpoint_sha256':digest(folder/'head.pt')})
            write(folder/'summary.json', {
                'native_mean_dice':float(np.mean([propagation_summary(native_rows[n])['present_dice'] for n in valnames])),
                'refined_mean_dice':float(np.mean([all_sequences[n]['present_dice'] for n in valnames])),
                'refined_false_alarms':sum(all_sequences[n]['gt_empty_false_alarm_count'] for n in valnames)})
        native={n:propagation_summary(native_rows[n]) for n in names}
        for n in names:
            for metric in ('boundary_f','iou'):
                native[n][metric]=float(np.mean([r['scores'][metric] for r in native_rows[n] if r['present']]))
        left={n:all_sequences[n]['present_dice'] for n in names};right={n:native[n]['present_dice'] for n in names}
        delta={n:left[n]-right[n] for n in names}
        result={'native':native,'refined':all_sequences,'deltas':delta,'mean_native':float(np.mean(list(right.values()))),
                'mean_refined':float(np.mean(list(left.values()))),'delta_ci95':sequence_bootstrap_interval(delta,seed=20261031),
                'exploratory_wilcoxon':paired_wilcoxon(left,right),'patient_independence_verified':False}
        write(out/'results.json',result)
        lines=['# Sequence-grouped current-frame refinement CV','','Exploratory development; patient independence unverified. Every eligible sequence evaluated out of fold once.', '',
               '| Sequence | Native Dice | Refined Dice | Delta |','|---|---:|---:|---:|']
        for n in names:lines.append(f'| {n} | {right[n]:.6f} | {left[n]:.6f} | {delta[n]:+.6f} |')
        lines += ['',f"Mean native: {result['mean_native']:.6f}; refined: {result['mean_refined']:.6f}.",
                  f"Exploratory sequence-bootstrap delta interval: {result['delta_ci95']}.",
                  'No external confirmation, new regional method, or additional seeds were launched. Interpret gains together with all sequence losses, false alarms and return outcomes in results.json.']
        (out/'results.md').write_text('\n'.join(lines)+'\n')
        check_compatible_sources(prior['sources']);check_sources(hashes)
        write(out/'completion.json',{'complete':True,'folds':3,'sequences':len(all_sequences),'seed':20261031})
        write(out/'progress.json',{'phase':'complete'})
    except BaseException as e:
        write(out/(f'resume_failure_{time.time_ns()}.json' if args.resume else 'failure.json'),{'error':repr(e)})
        raise


if __name__=='__main__':main()
