"""Matched source32 current-frame-only ablation; all artifacts stay on Drive."""
import argparse
import json
from pathlib import Path
import subprocess
import sys

import numpy as np
import torch
import torch.nn.functional as F

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from evaluation.segmentation import calculate_scores
from evaluation.davis2017.metrics import db_eval_boundary
from modeling.current_frame_refinement import CurrentFrameRefinement
from modeling.regional_refinement import refinement_loss
from scripts.run_regional_refinement import batch, check_sources, evaluate, write
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, validate_output

FILES = ['modeling/current_frame_refinement.py', 'scripts/run_current_frame_refinement.py',
         'tests/test_current_frame_refinement.py', 'protocols/current_frame_refinement_v1.json']
SEED = 20261031


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--source', type=Path, required=True)
    parser.add_argument('--diagnostics', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    parser.add_argument('--gate', action='store_true')
    args = parser.parse_args()
    args.output_dir = validate_output(args.output_dir)
    args.output_dir.mkdir(parents=True, exist_ok=False)
    out = args.output_dir
    try:
        torch.set_num_threads(2)
        torch.use_deterministic_algorithms(True)
        prior = json.loads((args.source/'setup.json').read_text())
        check_sources(prior['sources'])
        complete = json.loads((args.diagnostics/'completion.json').read_text())
        if not complete['complete'] or not complete['original_replay_passed']:
            raise RuntimeError('Source33 diagnostic incomplete')
        subprocess.run(['git','diff','--exit-code','HEAD','--',*FILES],cwd=ROOT,check=True)
        subprocess.run(['git','ls-files','--error-unmatch',*FILES],cwd=ROOT,check=True,capture_output=True)
        own_hashes = {f: digest(ROOT/f) for f in FILES}
        references = {a: json.loads((args.source/f'{a}_{SEED}/results.json').read_text())
                      for a in ('generic','regional','shuffled')}
        reference_hashes = {a: digest(args.source/f'{a}_{SEED}/results.json') for a in references}
        training = json.loads((args.source/'training_samples.json').read_text())
        index = json.loads((args.source/'cache_index.json').read_text())
        names = prior['protocol']['split']['validation_sequences']
        testing = [r for r in index if r['sequence'] in names]
        if len(training) != 181 or len(testing) != 372:
            raise RuntimeError('Unexpected source32 cohort')
        if args.gate:
            training = training[:4]
            testing = [next(r for r in testing if r['sequence']==n) for n in ('seq18','seq23')]
        else:
            gate = json.loads((out.parent/'gate/completion.json').read_text())
            if not gate.get('gate') or not gate.get('complete') or gate['source_hashes'] != own_hashes:
                raise RuntimeError('Matching gate required before training')
        write(out/'setup.json', {'commit': subprocess.check_output(['git','rev-parse','HEAD'],cwd=ROOT,text=True).strip(),
              'protocol': json.loads((ROOT/FILES[-1]).read_text()), 'source_hashes': own_hashes,
              'source32_setup_sha256': digest(args.source/'setup.json'),
              'source33_completion_sha256': digest(args.diagnostics/'completion.json'),
              'reference_report_hashes': reference_hashes, 'gate': args.gate})
        for i, row in enumerate(training+testing):
            if digest(row['path']) != row['sha256']:
                raise RuntimeError('Cache SHA mismatch: '+row['path'])
            write(out/'progress.json', {'phase':'verify_cache','done':i+1,'total':len(training)+len(testing)})
        torch.manual_seed(SEED)
        head = CurrentFrameRefinement()
        optimizer = torch.optim.Adam(head.parameters(), lr=.001)
        if args.gate:
            head.eval()
            with torch.no_grad():
                for row in testing:
                    inputs, _ = batch([row['path']], 'regional')
                    logits = head(*inputs)
                    if not torch.equal(logits, inputs[3]):
                        raise RuntimeError('Zero residual mismatch')
                    truth = load_label_mask(Path(row['mask']), [1])==1
                    predicted = F.interpolate(logits,size=truth.shape,mode='bilinear',align_corners=False)[0,0].numpy()>0
                    scores = calculate_scores(predicted, truth)
                    scores['boundary_f'] = float(db_eval_boundary(predicted,truth))
                    native = json.loads((args.source/'cache'/row['sequence']/'native_frames.json').read_text())
                    reference = next(r for r in native if r['frame']==row['frame'])
                    if any(abs(scores[k]-reference['scores'][k])>1e-9 for k in ('dice','iou','boundary_f')):
                        raise RuntimeError('Zero-head native metric replay mismatch')
            head.train()
            inputs, target = batch([r['path'] for r in training], 'regional')
            for _ in range(2):
                optimizer.zero_grad()
                loss = refinement_loss(head(*inputs),target)
                loss.backward()
                optimizer.step()
            if not torch.isfinite(loss) or not head.project.conv.weight.grad.abs().sum()>0:
                raise RuntimeError('Real-data gradient gate failed')
            write(out/'completion.json', {'complete':True,'gate':True,'source_hashes':own_hashes,
                  'zero_native_replay':True,'real_gradient_passed':True})
            write(out/'progress.json', {'phase':'complete'})
            return
        rng = np.random.default_rng(SEED)
        head.train()
        losses = []
        batches = []
        for step in range(500):
            selected = rng.choice(len(training),size=4,replace=True)
            batches.append(selected.tolist())
            inputs,target = batch([training[i]['path'] for i in selected], 'regional')
            optimizer.zero_grad()
            loss = refinement_loss(head(*inputs),target)
            if not torch.isfinite(loss):
                raise RuntimeError('Nonfinite training loss')
            loss.backward()
            optimizer.step()
            losses.append(float(loss.detach()))
            if step%10==0:
                write(out/'progress.json', {'phase':'training','step':step+1,'total':500})
                write(out/'losses.json',losses)
        torch.save(head.state_dict(),out/'head.pt')
        write(out/'losses.json',losses)
        write(out/'batch_indices.json',batches)
        frames, sequences, timings = {}, {}, []
        for name in names:
            check_sources(own_hashes)
            rows = [r for r in testing if r['sequence']==name]
            write(out/'progress.json', {'phase':'validation','sequence':name})
            result = evaluate(args,head,'regional',rows,None)
            frames.update(result['frames'])
            sequences.update(result['sequences'])
            timings.append((len(rows),result['head_seconds_per_frame']))
            write(out/f'{name}.json', result)
        result = {'sequences':sequences,'frames':frames,'mean_dice':float(np.mean([v['present_dice'] for v in sequences.values()])),
                  'false_alarms':sum(v['gt_empty_false_alarm_count'] for v in sequences.values()),
                  'parameters':sum(p.numel() for p in head.parameters()),
                  'head_seconds_per_frame':sum(n*t for n,t in timings)/sum(n for n,t in timings)}
        write(out/'results.json',result)
        native = {n:propagation_summary(json.loads((args.source/'cache'/n/'native_frames.json').read_text())) for n in names}
        all_results = {'native':{'sequences':native,'mean_dice':float(np.mean([v['present_dice'] for v in native.values()])),
                                'false_alarms':sum(v['gt_empty_false_alarm_count'] for v in native.values()),'parameters':None},
                       'current_only':result,**references}
        lines = ['# Current-frame refinement ablation','','Single seed; exploratory PolypGen results. Query retains native SAM2 memory conditioning.', '',
                 '| Head | Mean present Dice | False alarms | Head parameters |','|---|---:|---:|---:|']
        for arm,r in all_results.items():
            lines.append(f"| {arm} | {r['mean_dice']:.6f} | {r['false_alarms']} | {r['parameters']} |")
        lines += ['', '| Sequence | Native | Current only | Generic | Regional | Shuffled |','|---|---:|---:|---:|---:|---:|']
        for name in names:
            lines.append('| '+name+' | '+' | '.join(f"{r['sequences'][name]['present_dice']:.6f}" for r in all_results.values())+' |')
        delta = references['regional']['mean_dice']-result['mean_dice']
        lines += ['', '## Interpretation', '',
                  'Earlier pooled-context refinement was inconsistent, and swapping contexts helped some sequences while hurting others. This ablation removes the additional memory branch while retaining the current-frame correction path.',
                  f'Regional minus current-only mean Dice: {delta:+.6f}. This is a single-seed numerical comparison with fewer parameters in the current-only head, not significance or generalization evidence.',
                  'Inspect all sequence deltas and false alarms before deciding whether regional pooling earns its extra complexity. No new configuration or extra seed is automatically launched.']
        (out/'results.md').write_text('\n'.join(lines)+'\n')
        check_sources(prior['sources'])
        check_sources(own_hashes)
        if any(digest(args.source/f'{a}_{SEED}/results.json')!=sha for a,sha in reference_hashes.items()):
            raise RuntimeError('Reference report changed')
        write(out/'completion.json', {'complete':True,'gate':False,'seed':SEED,'steps':500,
              'train_samples':len(training),'validation_samples':len(testing),'checkpoint_sha256':digest(out/'head.pt')})
        write(out/'progress.json', {'phase':'complete'})
    except BaseException as error:
        write(out/'failure.json', {'error':repr(error)})
        raise


if __name__ == '__main__':
    main()
