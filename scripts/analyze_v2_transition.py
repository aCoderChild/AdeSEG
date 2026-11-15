"""Post-hoc inspection of saved seq13 transition rollouts; no inference."""
import argparse
import csv
import json
import os
from io import BytesIO
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.io import load_label_mask
from evaluation.segmentation import calculate_scores
from scripts.false_alarm_statistics import propagation_summary
from scripts.memory_token_io import digest, validate_output

CORE = ('prediction_sha256', 'low_res_sha256', 'bank_sha256', 'cache_sha256')


def first_difference(rows, reference, key):
    return next((f for f in sorted(rows) if rows[f][key] != reference[f][key]), None)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--run', type=Path, required=True)
    parser.add_argument('--inputs', type=Path, required=True)
    parser.add_argument('--output-dir', type=Path, required=True)
    args = parser.parse_args()
    out = validate_output(args.output_dir)
    out.mkdir(parents=True, exist_ok=True)
    os.environ['MPLCONFIGDIR'] = str(out/'mplconfig')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    assert json.loads((args.run/'validation.json').read_text())['passed']
    assert json.loads((args.run/'delivery.json').read_text())['completed']
    inputs = json.loads(args.inputs.read_text())
    archives = {'native': args.run/'gate_seq13_sham.zip'}
    for schedule in ('transition_pulse', 'stop_before_transition'):
        for condition in ('regional', 'random_20261025', 'random_20261026', 'random_20261027'):
            archives[f'{schedule}/{condition}'] = args.run/f'{schedule}_{condition}_first.zip'
    data, integrity, masks, summaries, provenance = {}, {}, {}, {}, {}
    truths = {i: load_label_mask(Path(inputs[i]['mask']), [1]) == 1 for i in range(8,250)}
    checked = 0
    for name, path in archives.items():
        with zipfile.ZipFile(path) as z:
            assert z.testzip() is None
            data[name] = {r['frame']: r for r in map(json.loads,z.read('frames.jsonl').decode().splitlines())}
            integrity[name] = {r['frame']: r for r in map(json.loads,z.read('integrity.jsonl').decode().splitlines())}
            summary = json.loads(z.read('summary.json'))
            assert list(data[name]) == list(range(8,250))
            assert list(integrity[name]) == list(range(7,250))
            masks[name] = {}
            for f, row in data[name].items():
                mask = np.asarray(Image.open(BytesIO(z.read(f'frame_{f:06d}.png')))) > 0
                assert mask.shape == truths[f].shape
                assert bool(mask.any()) == row['prediction_nonempty']
                assert row['present'] == bool(truths[f].any())
                assert row['prediction_sha256'] == integrity[name][f]['prediction_sha256']
                scores = calculate_scores(mask, truths[f])
                assert abs(scores['dice']-row['scores']['dice']) < 1e-12
                assert abs(scores['iou']-row['scores']['iou']) < 1e-12
                masks[name][f] = mask
                checked += 1
            computed = propagation_summary(list(data[name].values()))
            for key in ('present_dice','targets','present_targets','gt_empty_false_alarm_count'):
                assert computed[key] == summary['whole'][key]
            summaries[name] = summary['whole']
        provenance[name] = {'path': str(path), 'sha256': digest(path)}
        if name != 'native':
            with zipfile.ZipFile(Path(str(path).replace('_first.zip','_repeat.zip'))) as z:
                repeat = {r['frame']:r for r in map(json.loads,z.read('integrity.jsonl').decode().splitlines())}
                assert repeat == integrity[name]
                assert z.read('frames.jsonl').decode().splitlines() == [json.dumps(r) for r in data[name].values()]
    events, table = {}, []
    for name, rows in data.items():
        for f,row in rows.items():
            diff = int(np.count_nonzero(masks[name][f] != masks['native'][f]))
            assert diff == row['different_pixels_from_native']
            receipt = integrity[name][f]
            pulse = receipt['pulse']
            removed = 0 if pulse is None else pulse['layers'][0]['raw_spatial_keys']-pulse['layers'][0]['selected_spatial_keys']
            table.append({'condition': name, 'frame': f, 'present': row['present'], 'object_score': row['object_score'],
                          'dice': row['scores']['dice'], 'boundary_f':row['scores']['boundary_f'],
                          'predicted_pixels':int(masks[name][f].sum()), 'different_pixels_from_native':diff,
                          'removed_tokens':removed, **{k+'_differs':receipt[k]!=integrity['native'][f][k] for k in CORE}})
        events[name] = {k:first_difference(integrity[name],integrity['native'],k) for k in CORE}
        events[name].update(first_object_score_difference=first_difference(rows,data['native'],'object_score'),
            first_binary_difference=next((f for f,r in rows.items() if r['different_pixels_from_native']),None),
            nonempty_after_50=sum(r['prediction_nonempty'] for f,r in rows.items() if f>=50),
            present_after_50=sum(r['present'] for f,r in rows.items() if f>=50),
            missed_present_after_50=sum(r['present'] and not r['prediction_nonempty'] for f,r in rows.items() if f>=50),
            returns=summaries[name]['returns'])
    with (out/'frames.csv').open('w') as s:
        writer=csv.DictWriter(s, fieldnames=list(table[0])); writer.writeheader(); writer.writerows(table)
    left, right = 'transition_pulse/regional', 'stop_before_transition/regional'
    convergence = {k: max((f for f in integrity[left] if integrity[left][f][k] != integrity[right][f][k]), default=None) for k in CORE}
    convergence['object_score'] = max((f for f in data[left] if data[left][f]['object_score'] != data[right][f]['object_score']), default=None)
    result={'scope':'Post-hoc seq13 only; hashes establish divergence, not the unique causal memory component.',
            'last_difference_between_regional_schedules':convergence,
            'verified_masks':checked, 'fresh_repeats_exact':True, 'events':events,'summaries':summaries,
            'sources':provenance,'script_sha256':digest(Path(__file__)), 'inputs_sha256':digest(args.inputs)}
    (out/'analysis.json').write_text(json.dumps(result,indent=2)+'\n')
    focus=['native','transition_pulse/regional','stop_before_transition/regional']
    fig,axes=plt.subplots(3,1,figsize=(12,8),sharex=True)
    for name in data:
        rows=data[name]; f=list(rows); random='random' in name
        style={'color':'gray','alpha':.25,'linewidth':.7} if random else {'linewidth':1.4}
        for ax,key in zip(axes[:2],('object_score','dice')):
            values=[r['object_score'] if key=='object_score' else r['scores']['dice'] if r['present'] else np.nan for r in rows.values()]
            ax.plot(f,values,label=name,**style)
        axes[2].plot(f,[int(masks[name][i].sum()) for i in f],**style)
    for ax in axes:
        for f in (50,66): ax.axvline(f,color='black',linestyle=':',linewidth=.6)
        ax.grid(alpha=.2)
    axes[0].axhline(0,color='black',linewidth=.6); axes[0].set_ylabel('Object-score logit')
    axes[1].set_ylabel('Present-frame Dice'); axes[2].set_ylabel('Predicted pixels'); axes[2].set_xlabel('Manifest frame (not source-video frame)')
    axes[0].legend(fontsize=6,ncol=2); fig.tight_layout(); fig.savefig(out/'trajectories.png',dpi=150); plt.close(fig)
    selected=[49,50,66,67,194,249]
    fig,axes=plt.subplots(len(selected),4,figsize=(12,15))
    for row,f in enumerate(selected):
        assert digest(inputs[f]['image']) == inputs[f]['image_sha256']
        assert digest(inputs[f]['mask']) == inputs[f]['mask_sha256']
        image=np.array(Image.open(inputs[f]['image']).convert('RGB'))
        for col,name in enumerate(['GT']+focus):
            mask=truths[f] if name=='GT' else masks[name][f]
            overlay=image.copy(); overlay[mask]=(overlay[mask]*.55+np.array([0,230,100])*.45).astype('uint8')
            axes[row,col].imshow(overlay); axes[row,col].axis('off')
            axes[row,col].set_title(f'{f} {name}\n{int(mask.sum())} mask pixels',fontsize=8)
    fig.suptitle('Original-resolution mask overlays (green); not token-grid overlays',fontsize=12)
    fig.tight_layout(rect=(0,0,1,.97)); fig.savefig(out/'overlays.png',dpi=130); plt.close(fig)
    lines=['# Seq13 transition: saved-trajectory analysis','',
           'Post hoc and exploratory. No inference or model changes. Frame indices are manifest indices.', '',
           f'Validation: {checked} PNG masks checked against saved nonempty flags, Dice/IoU, native pixel differences and prediction receipts; all eight fresh-repeat records agree. Summary metrics recomputed.', '',
           '| Condition | First logit hash difference | First binary difference | First bank difference | First cache difference | Dice | Missed present frames from50 |',
           '|---|---:|---:|---:|---:|---:|---:|']
    for name in data:
        e=events[name]; lines.append(f"| {name} | {e['prediction_sha256']} | {e['first_binary_difference']} | {e['bank_sha256']} | {e['cache_sha256']} | {summaries[name]['present_dice']:.6f} | {e['missed_present_after_50']} / {e['present_after_50']} |")
    lines += ['', '## Transition scores', '', '| Frame | GT present | Native | Pulse at50 | Stop after49 |', '|---|---|---:|---:|---:|']
    for f in [10,49,50,51,56,66,67,194,249]:
        lines.append(f"| {f} | {data['native'][f]['present']} | " + ' | '.join(f"{data[n][f]['object_score']:.8f}" for n in focus)+' |')
    lines += ['', 'Full frame-level data: `frames.csv`; provenance/events/returns: `analysis.json`; figures: `trajectories.png`, `overlays.png`.',
              '', '## Interpretation', '',
              'At frame50 the regional pulse removes 1,400 of 28,672 spatial keys (4.88%). Native object score is +0.01122089; the pulse yields -0.01649044. SAM2 hard-suppresses masks when its object score is non-positive (`external/sam2/sam2/modeling/sam2_base.py:360`). This initially removes an annotated-absent false alarm, but both regional schedules then produce empty masks for every frame50–249, missing all 163 present frames and all six later return episodes. All six random controls remain nonempty and obtain positive overlap on those returns. Positive overlap is not adequate segmentation: native Dice at66 is only about 0.000023.',
              '', 'Restoring native reading at50 after pruning through49 also fails: its first differences occur at10, and score at50 is -0.02769204 without an intervention on that frame. The frame50-only pulse has an exact native prefix. Repetition is therefore unnecessary in this case, and stopping earlier does not undo the altered state.',
              '', 'Both regional schedules have identical suppressed prediction and low-resolution mask hashes from50 onward; their object scores become exactly equal from65 onward. Complete bank hashes remain different through249 because they include historical outputs. Cached-image-feature hashes match native throughout. These facts are consistent with history-dependent loss of recovery under normal writeback, but do not isolate mask-memory vectors, object pointers or their relative causal roles.',
              '', 'The random arms also change hashes and pixels without collapse: divergence alone is not failure. The overlays show a large native false-positive mask at49/50 and poor overlap at66/67, followed by substantially better native overlap at194/249. This is a false-alarm/recovery trade-off in one selected case, not proof that boundary vectors contain unique information or that regional utilization cannot help.',
              '', 'Bank hashes cover the complete stored output dictionary: a changed hash does not isolate memory vectors from masks, scores or pointers. Cache hashes track cached features; equal cache hashes do not imply equal memory-conditioned decoder inputs. No saved tensor-level mediation analysis is claimed.']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')
    print(json.dumps({'verified_masks':checked,'events':events},indent=2))

if __name__ == '__main__':
    main()
