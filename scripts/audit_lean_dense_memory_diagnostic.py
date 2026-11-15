#!/usr/bin/env python3
"""Independently rescore saved fixed-100% Lean memory-retention predictions."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import numpy as np
from PIL import Image

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask  # noqa: E402
from datasets.video import load_manifest  # noqa: E402
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def read(path): return json.loads(path.read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--part',type=Path,action='append',required=True)
    for name in ('merged','historical','verification','output'):
        p.add_argument('--'+name,type=Path,required=True)
    a=p.parse_args()
    protocol=read(ROOT/'protocols/lean_auto_dense_memory_diagnostic_v1.json')
    merge_done=read(a.merged/'completion.json')
    if not merge_done['passed'] or digest(a.merged/'results.json') != merge_done['results_sha256'] or digest(a.historical/'comparison.json') != merge_done['historical_comparison_sha256']:
        raise ValueError('Reference results changed')
    old_all=read(a.merged/'results.json')['sequences']; locations=read(a.merged/'locations.json')
    hist={x['sequence']:x for x in read(a.historical/'comparison.json')['sequences']}
    videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}
    verification=read(a.verification)
    if not verification['passed'] or verification['source_sha256'] != digest(ROOT/'scripts/verify_lean_dense_engineering.py'):
        raise ValueError('Engineering verification changed')
    results={}
    for part in a.part:
        setup=read(part/'setup.json'); done=read(part/'completion.json')
        if not done['passed'] or done['sequences'] != setup['sequences'] or setup['verification_sha256'] != digest(a.verification):
            raise ValueError('Partition incomplete')
        for name in setup['sequences']:
            if name in results: raise ValueError('Overlapping partitions')
            run=part/name; record=read(run/'records.json'); receipt=read(run/'completion.json'); meta=read(run/'setup.json')
            if not receipt['passed'] or digest(run/'records.json') != receipt['records_sha256'] or meta['mode'] != 'ablation' or meta['strength'] != 0.5 or meta['keep_ratio'] != 1.0 or meta['engineering']:
                raise ValueError(f'{name}: bad receipt')
            if meta['gate_identity']['protocol_sha256'] != digest(ROOT/'protocols/lean_auto_dense_memory_diagnostic_v1.json'):
                raise ValueError('Protocol differs')
            if (digest(run/'frames.json') != record['frames_sha256'] or record['events'] <= 0
                or len(read(run/'events.json')) != record['events'] or digest(run/'memory_reads.json') != record['memory_reads_sha256']):
                raise ValueError('Unexpected memory changes or frame record')
            selections=read(run/'selections.json')
            if selections != record['selections'] or any(x['kept'] != x['candidates'] for x in selections):
                raise ValueError('Token selection receipt differs')
            for event in selections:
                expected = hashlib.sha256(np.arange(event['candidates'], dtype=np.int64).tobytes()).hexdigest()
                if event['indices_sha256'] != expected:
                    raise ValueError('Dense retained-index set does not cover every spatial candidate')
            original=Path(locations[name]); old_receipt=read(original/'completion.json'); old_audit=read(original/'independent_audit.json')
            if not old_receipt['passed'] or not old_audit['passed'] or digest(original/'records.json') != old_receipt['records_sha256'] or old_audit['records_sha256'] != old_receipt['records_sha256'] or record['original_records_sha256'] != old_receipt['records_sha256']:
                raise ValueError('Source run differs')
            old_record=read(original/'records.json')[0]; old_folder=original/name
            if digest(old_folder/'frames.json') != old_record['frames_sha256']:
                raise ValueError('Old frames changed')
            if digest(old_folder/'selections.json') != old_record['selections_sha256']:
                raise ValueError('Old token selection receipt changed')
            old_selections=read(old_folder/'selections.json')
            selection_key=lambda x:(x['frame_index'],x['source_frame_index'])
            old_by_frame={selection_key(x):x for x in old_selections}
            new_by_frame={selection_key(x):x for x in selections}
            if len(old_by_frame) != len(old_selections) or len(new_by_frame) != len(selections):
                raise ValueError('Ambiguous token selection event')
            common_selection_keys=old_by_frame.keys() & new_by_frame.keys()
            same_selections=sum(old_by_frame[k]['indices_sha256'] == new_by_frame[k]['indices_sha256'] for k in common_selection_keys)
            memory_reads=read(run/'memory_reads.json')
            if not memory_reads or any(x['pointer_tokens'] < 0 or x['pointer_tokens'] > x['memory_tokens'] for x in memory_reads):
                raise ValueError('Invalid memory-attention shape receipt')
            old_rows=read(old_folder/'frames.json'); new_rows=read(run/'frames.json'); frames=videos[name]
            if not (len(new_rows)==len(old_rows)==len(frames)==record['frames']): raise ValueError('Coverage mismatch')
            values=[]
            for i,(row,old,frame) in enumerate(zip(new_rows,old_rows,frames)):
                path=run/f'{i:06d}.png'
                if row['frame'] != i or digest(path) != row['mask_sha256'] or row['mask_sha256'] != record['mask_sha256'][i] or digest(frame.mask_path) != old['label_sha256'] or digest(frame.image_path) != old['image_sha256']:
                    raise ValueError(f'{name}/{i}: inputs or masks changed')
                pred=np.asarray(Image.open(path))>0; truth=load_label_mask(frame.mask_path,[1])==1
                if pred.shape != truth.shape: raise ValueError('Mask shape differs')
                denom=int(pred.sum()+truth.sum()); present=bool(truth.any())
                dice=2*int((pred & truth).sum())/denom if denom else 1.
                alarm=bool(pred.any()) if not present else None
                if row['present'] != present or old['present'] != present or not np.isclose(row['dice'],dice,atol=1e-12,rtol=0) or row['false_alarm'] != alarm:
                    raise ValueError(f'{name}/{i}: scoring differs')
                values.append((present,dice,alarm))
            targets=[json.loads(line) for line in (a.historical/'full'/name/'frames.jsonl').read_text().splitlines()]
            if len(targets) != hist[name]['targets'] or any(values[x['frame']][0] != x['present'] for x in targets):
                raise ValueError('Historical targets differ')
            present=[v[1] for v in values if v[0]]; empty=[v[2] for v in values if not v[0]]
            hp=[values[x['frame']][1] for x in targets if x['present']]
            he=[values[x['frame']][2] for x in targets if not x['present']]
            results[name]={'frames':len(values),'all_present':len(present),'all_empty':len(empty),
                'ablation_all_dice':float(np.mean(present)),'combined_all_dice':old_all[name]['all_present_dice'],
                'ablation_all_false_alarms':sum(empty),'combined_all_false_alarms':old_all[name]['all_false_alarms'],
                'historical_present':len(hp),'historical_empty':len(he),'ablation_historical_dice':float(np.mean(hp)),
                'combined_historical_dice':old_all[name]['historical_present_dice'],
                'ablation_historical_false_alarms':sum(he),'combined_historical_false_alarms':old_all[name]['historical_false_alarms'],
                'native_gt_prompt_dice':hist[name]['native_dice'],'lean_gt_prompt_dice':hist[name]['lean_dice'],
                'peak_attention_tokens':max(x['memory_tokens'] for x in memory_reads),
                'selection_events':len(selections), 'matched_selection_events':len(common_selection_keys),
                'identical_selection_events':same_selections}
    if set(results) != set(protocol['cohort']): raise ValueError('Frozen cohort incomplete')
    for row in results.values():
        row['delta_ablation_minus_combined']=row['ablation_historical_dice']-row['combined_historical_dice']
    out=validate_output(a.output); out.mkdir(parents=True,exist_ok=False)
    summary={'scope':'Retrospective selected videos. Same automatic prompts and frame schedule; fixed 100% versus 25% Lean TAMP spatial retention. Other Lean mechanisms remain active, and compute is not matched. Historical native/Lean used GT initial masks.',
             'sequences':results, 'mean_ablation_historical_dice':float(np.mean([r['ablation_historical_dice'] for r in results.values()])),
             'mean_combined_historical_dice':float(np.mean([r['combined_historical_dice'] for r in results.values()])),
             'ablation_all_false_alarms':sum(r['ablation_all_false_alarms'] for r in results.values()),
             'combined_all_false_alarms':sum(r['combined_all_false_alarms'] for r in results.values()),
             'all_empty':sum(r['all_empty'] for r in results.values())}
    (out/'results.json').write_text(json.dumps(summary,indent=2)+'\n')
    lines=['# Fixed-prompt Lean spatial-retention diagnostic','',summary['scope'],'',
           '| Sequence | Native GT prompt | Lean GT prompt | Combined automatic | 100% retention automatic | Difference | False alarms combined / ablation (all empty frames) |',
           '| --- | ---: | ---: | ---: | ---: | ---: | ---: |']
    for name,r in results.items():
        lines.append(f"| {name} | {r['native_gt_prompt_dice']:.3f} | {r['lean_gt_prompt_dice']:.3f} | {r['combined_historical_dice']:.3f} | {r['ablation_historical_dice']:.3f} | {r['delta_ablation_minus_combined']:+.3f} | {r['combined_all_false_alarms']} / {r['ablation_all_false_alarms']} ({r['all_empty']}) |")
    lines+=['',f"Across these selected sequences, equal-sequence historical-target Dice is {summary['mean_combined_historical_dice']:.4f} combined and {summary['mean_ablation_historical_dice']:.4f} with 100% retention. These are exploratory sequence-selected outcomes; no causal claim about unrelated components or generalization follows.",
            'No labels were used in either model input. Same automatic prompt is replayed exactly; the retained-token budget changes. The GT-prompt historical columns have different initialization and are descriptive only. False alarms count any predicted foreground on a labeled-empty frame.','']
    (out/'report.md').write_text('\n'.join(lines))
    (out/'completion.json').write_text(json.dumps({'passed':True,'results_sha256':digest(out/'results.json'),'audit_sha256':digest(Path(__file__))},indent=2)+'\n')
    print(json.dumps({k:v for k,v in summary.items() if k != 'sequences'},indent=2))


if __name__=='__main__': main()
