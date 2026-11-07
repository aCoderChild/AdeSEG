"""Run172: instance-aware, fixed-native-history Run171 replication on DAVIS.

Native propagation and output competition remain author SAM2 operations. Full
indexed prediction exports use the pinned official evaluator. Counterfactual
exports are diagnostics on native histories, not autoregressive efficacy runs.
"""
import argparse
from contextlib import nullcontext
from datetime import datetime, timezone
import gc
import gzip
import json
import os
from pathlib import Path
import subprocess
import sys
import tempfile
import time
import zipfile

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.vos_benchmarks import author_mask_io, inventory
from evaluation.segmentation import calculate_scores
from evaluation.statistics import paired_wilcoxon, holm_adjust, sequence_bootstrap_interval
from modeling.cfbi_training import capture_training_read
from modeling.complementary_memory.integration import _tensor_sha256
from modeling.instance_token_contribution import InstanceTokenContributionRead
from modeling.token_contribution import ARMS
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_complementary_memory_replay import check_exact, check_layout
from scripts.run_regional_token_selection import replay, EXTRA_SOURCES
from scripts.run_token_contribution import SOURCES as RUN171_SOURCES
from scripts import run_complementary_memory_pilot as base
from scripts.run_davis_soft_mask_aware_tome import (
    build_model, add_first_frame_prompts, run_official_evaluator, tensor_digest,
    EVALUATOR_REVISION, CONFIG,
)
from scripts.run_davis_region_constrained_tome import verify_prompt_inputs

PROTOCOL = 'protocols/davis_token_contribution_v1.json'
PRIMARY = ('native', 'rejected', 'retained', 'random')
SPLITS = ('rejected_a', 'rejected_b')
OWN_SOURCES = (PROTOCOL, 'modeling/instance_token_contribution.py',
               'scripts/run_davis_token_contribution.py', 'tests/test_instance_token_contribution.py')
EXTRA = ('datasets/vos_benchmarks.py', 'scripts/run_davis_soft_mask_aware_tome.py',
         'scripts/run_davis_region_constrained_tome.py', 'external/sam2/tools/vos_inference.py',
         'external/davis2017-evaluation/davis2017/evaluation.py',
         'external/davis2017-evaluation/davis2017/davis.py',
         'external/davis2017-evaluation/davis2017/metrics.py',
         'external/davis2017-evaluation/davis2017/results.py',
         'external/davis2017-evaluation/davis2017/utils.py', 'external/MOSE-api/README.md')


def status(out, phase, **fields):
    base.atomic_json(out/'run_status.json', {'phase':phase, 'pid':os.getpid(),
        'updated_at':datetime.now(timezone.utc).isoformat(), **fields})


def stored(state, object_id, frame):
    bank = state['output_dict_per_obj'][state['obj_id_to_idx'][object_id]]
    return bank['cond_frame_outputs'].get(frame, bank['non_cond_frame_outputs'].get(frame))


def raw_signatures(state, frame, ids):
    return {str(oid): {k:_tensor_sha256(stored(state,oid,frame)[k])
            for k in ('pred_masks','obj_ptr','object_score_logits','maskmem_features')}
            for oid in ids}


def bank_signature(state):
    return fingerprint({'banks':state['output_dict_per_obj'], 'cache':state['cached_features'],
        'id_to_index':state['obj_id_to_idx'], 'index_to_id':state['obj_idx_to_id']})['sha256']


def verify_trace(actual, native, owner):
    check_layout(actual,native)
    if actual['mask_sources'] != native['mask_sources']:
        raise RuntimeError('Stored-mask lineage changed during replay')
    if actual['object_id'] != owner or any(x['object_id'] != owner for x in actual['mask_sources']):
        raise RuntimeError('Cross-instance provenance in diagnostic read')
    for layer in actual['selection_layers']:
        if layer['object_id'] != owner:
            raise RuntimeError('Layer used another instance selector cache')


def merged_prediction(model, state, logits_by_id, ids):
    low = torch.cat([logits_by_id[oid] for oid in ids], dim=0)
    full = model._get_orig_video_res_output(state,low)[1]
    masks = {oid:(full[i,0] > 0).cpu().numpy() for i,oid in enumerate(ids)}
    indexed = author_mask_io()['put_per_obj_mask'](masks,state['video_height'],state['video_width'])
    return full,indexed,masks


def score_object(prediction, labels, object_id):
    valid = labels != 255
    truth = labels == object_id
    # Supplementary Dice ignores void. Official J/F are computed separately by
    # the unchanged semi-supervised evaluator (which maps void GT to background).
    dice = calculate_scores(prediction & valid,truth & valid)['dice']
    present = bool(truth.any())
    return {'dice':dice, 'present':present,
            'false_alarm':bool((prediction & valid).any()) if not present else None}


def video_supplement(rows, ids, plan):
    per_object = {}
    for oid in ids:
        relevant = [r for r in rows if r['object_id']==oid and r['official_scored']]
        present = [r for r in relevant if r['present']]
        sampled = [r for r in relevant if r['frame'] in plan[str(oid)]['present']]
        empty = [r for r in relevant if r['frame'] in plan[str(oid)]['absent']]
        # Split-arm engineering fixtures intentionally contain a shorter target
        # subset. Full-cohort completeness is checked separately below.
        if any(not r['present'] for r in sampled) or any(r['present'] for r in empty):
            raise RuntimeError('Frozen present/absent target labels changed')
        per_object[str(oid)] = {'official_frames':len(relevant), 'present_frames':len(present),
            'full_present_dice':float(np.mean([r['dice'] for r in present])) if present else None,
            'sampled_present_dice':float(np.mean([r['dice'] for r in sampled])) if sampled else None,
            'sampled_present_frames':len(sampled), 'sampled_empty_frames':len(empty),
            'sampled_false_alarms':sum(r['false_alarm'] for r in empty)}
    means = {key:float(np.mean([o[key] for o in per_object.values() if o[key] is not None]))
             if any(o[key] is not None for o in per_object.values()) else None
             for key in ('full_present_dice','sampled_present_dice')}
    return {'per_object':per_object, **means}


def validate_coverage(rows, ids, plan, frame_count):
    expected={(frame,oid) for frame in range(frame_count) for oid in ids}
    for arm in PRIMARY:
        actual=[(r['frame'],r['object_id']) for r in rows[arm]]
        if len(set(actual))!=len(actual) or set(actual)!=expected:
            raise RuntimeError('Incomplete or duplicated full DAVIS arm coverage')
    sampled=set().union(*(set(p['present']+p['absent']) for p in plan.values()))
    for arm in SPLITS:
        actual=[(r['frame'],r['object_id']) for r in rows[arm]]
        if len(set(actual))!=len(actual) or set(actual)!={(f,o) for f in sampled for o in ids}:
            raise RuntimeError('Incomplete split-interaction target union')
    native={(r['frame'],r['object_id']):r for r in rows['native']}
    for arm in ARMS:
        for row in rows[arm]:
            original=native[(row['frame'],row['object_id'])]
            if any(row[key]!=original[key] for key in ('present','sampled','official_scored','object_ids')):
                raise RuntimeError('Unmatched diagnostic denominators')


def summarize(results, official):
    names = list(results)
    contrasts,pvalues = {},{}
    for endpoint in ('official_jf','sampled_present_dice'):
        for comparator in ('retained','random'):
            label = f'{endpoint}:rejected_minus_{comparator}'
            values = {}
            left,right = {},{}
            for name in names:
                get = lambda arm: (official[arm]['per_video'][name]['J&F-Mean'] if endpoint=='official_jf'
                                   else results[name][arm]['sampled_present_dice'])
                l,r = get('rejected'),get(comparator)
                if l is None or r is None:
                    continue
                left[name],right[name] = l,r
                values[name] = l-r
            contrasts[label] = {'n_videos':len(values), 'video_deltas':values,
                'mean':float(np.mean(list(values.values()))),
                'median':float(np.median(list(values.values()))),
                'bootstrap_95_ci':sequence_bootstrap_interval(values,seed=20261107),
                'wins':sum(v>0 for v in values.values()), 'ties':sum(v==0 for v in values.values()),
                'losses':sum(v<0 for v in values.values())}
            contrasts[label]['leave_one_video_out_means']={
                n:float(np.mean([v for k,v in values.items() if k!=n])) for n in values if len(values)>1}
            pvalues[label] = paired_wilcoxon(left,right)['p']
    means = {}
    for arm in PRIMARY:
        means[arm] = {'official_global':official[arm]['global'],
            'equal_video_jf':float(np.mean([official[arm]['per_video'][n]['J&F-Mean'] for n in names])),
            **{key:float(np.mean([results[n][arm][key] for n in names if results[n][arm][key] is not None]))
               for key in ('sampled_present_dice','full_present_dice')}}
    interaction = {}
    for name in names:
        rows = [results[name][a]['sampled_present_dice'] for a in ('native','rejected_a','rejected_b','rejected')]
        if None not in rows:
            interaction[name] = rows[1]+rows[2]-rows[3]-rows[0]
    signal=len(names)==30 and all(c['bootstrap_95_ci'][0]>0 and c['wins']>c['n_videos']/2 for c in contrasts.values())
    return {'scope':'Full-coverage fixed-native-history counterfactual exports; not autoregressive efficacy',
        'video_count':len(names),'means':means,'primary_contrasts':contrasts,
        'ranking_discrimination_signal':signal,
        'holm_pvalues_four_contrasts':holm_adjust(pvalues), 'supplementary':results,
        'official':official,'sampled_interaction_video_deltas':interaction,
        'sampled_interaction_mean':float(np.mean(list(interaction.values()))) if interaction else None,
        'sampled_interaction_bootstrap_95_ci':sequence_bootstrap_interval(interaction,seed=20261107) if interaction else None}


def verify_saved(out, root, core, results):
    """Re-score persisted indexed PNGs and audit receipts, independent of logits."""
    counts={'videos':0,'object_records':0,'indexed_pngs':0,'four_layer_replays':0}
    for record in core['design']['inventory']['videos']:
        name=record['video'];directory=out/'full/records'/name/'diagnostic'
        receipt=json.loads((directory/'completion.json').read_text())
        if receipt['parameter_sha256']!=core['native_reference']['videos'][name]['parameter_digest']:
            raise RuntimeError('Persisted model parameter digest differs from native reference')
        rows={};images={};truth={}
        for arm in ARMS:
            path=directory/f'{arm}.jsonl.gz'
            if digest(path)!=receipt['record_sha256'][arm]:
                raise RuntimeError('Compressed records changed after video completion')
            with gzip.open(path,'rt') as stream:rows[arm]=[json.loads(line) for line in stream]
            for row in rows[arm]:
                key=(arm,row['frame'])
                if key not in images:
                    png=out/'full'/row['indexed_png']
                    if digest(png)!=row['indexed_png_sha256']:
                        raise RuntimeError('Indexed PNG checksum differs from its record')
                    labels=author_mask_io()['load_ann_png'](str(png))[0]
                    if set(np.unique(labels))-set([0,*record['object_ids']]):
                        raise RuntimeError('Indexed export has unknown or merged object IDs')
                    images[key]=labels;counts['indexed_pngs']+=1
                if row['frame'] not in truth:
                    truth[row['frame']]=author_mask_io()['load_ann_png'](
                        str(root/'Annotations/480p'/name/f"{row['stem']}.png"))[0]
                score=score_object(images[key]==row['object_id'],truth[row['frame']],row['object_id'])
                if any(score[k]!=row[k] for k in ('dice','present','false_alarm')):
                    raise RuntimeError('Saved indexed-mask metrics do not reproduce records')
                counts['object_records']+=1
            if video_supplement(rows[arm],record['object_ids'],core['design']['sample_plan'][name])!=results[name][arm]:
                raise RuntimeError('Saved records do not reproduce the video summary')
        validate_coverage(rows,record['object_ids'],core['design']['sample_plan'][name],len(record['frames']))
        native={(r['frame'],r['object_id']):r for r in rows['native']}
        for arm in ARMS:
            for row in rows[arm]:
                trace=row['read']
                if trace is None:continue
                verify_trace(trace,native[(row['frame'],row['object_id'])]['read'],row['object_id'])
                for layer in trace['selection_layers']:
                    removed=0 if arm=='native' else 128 if arm in SPLITS else 256
                    expected=[4096 if i in (0,len(layer['counts'])-1) else 4096-removed
                              for i in range(len(layer['counts']))]
                    if layer['counts']!=expected:
                        raise RuntimeError('Persisted key budget/protected-frame invariant failed')
                counts['four_layer_replays']+=1
        counts['videos']+=1
    source_guard(core)
    return {'passed':True,**counts,'scope':'Saved PNG checksums/metrics, full arm coverage, persistent IDs, four-layer lineage/budgets and native parameter digests rechecked.'}


def report(out, result, core):
    old=json.loads(Path(core['protocol']['polypgen_reference']['results']).read_text())
    lines=['# Run172: DAVIS token-group contribution replication','',
        '30 videos, 61 objects. Fixed native histories; no treatment writeback. Exploratory transfer, not a deployed pruning method or untouched confirmation.','',
        '| Removal | Official global J | F | J&F | Equal-video J&F | Sampled present Dice | Full present Dice |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for arm,m in result['means'].items():
        g=m['official_global']
        lines.append(f"| {arm} | {g['J-Mean']:.6f} | {g['F-Mean']:.6f} | {g['J&F-Mean']:.6f} | {m['equal_video_jf']:.6f} | {m['sampled_present_dice']:.6f} | {m['full_present_dice']:.6f} |")
    lines+=['','Official globals average objects; inference/statistics use video means. First and last frames are excluded by the unchanged official evaluator. Sampled Dice uses the frozen four-present-target plan per object.','',
        '| Prespecified contrast | Mean | Video-bootstrap 95% CI | Holm p (four tests) | Wins / ties / losses |',
        '|---|---:|---|---:|---|']
    for key,c in result['primary_contrasts'].items():
        lo,hi=c['bootstrap_95_ci']
        lines.append(f"| {key} | {c['mean']:+.6f} | [{lo:+.6f}, {hi:+.6f}] | {result['holm_pvalues_four_contrasts'][key]:.6f} | {c['wins']} / {c['ties']} / {c['losses']} |")
    lines+=['','| Video | Native J&F | Remove rejected | Remove retained | Remove random | Native sampled Dice | Rejected sampled Dice |',
        '|---|---:|---:|---:|---:|---:|---:|']
    for name in result['supplementary']:
        values=[result['official'][a]['per_video'][name]['J&F-Mean'] for a in PRIMARY]
        dice=[result['supplementary'][name][a]['sampled_present_dice'] for a in ('native','rejected')]
        lines.append('| '+name+' | '+' | '.join(f'{v:.6f}' for v in (*values,*dice))+' |')
    lines+=['','| Sampled Dice contrast | PolypGen Run171 | DAVIS Run172 | Same mean direction |',
        '|---|---:|---:|---|']
    for comparator in ('retained','random'):
        a=old['primary_contrasts'][f'rejected_minus_{comparator}']['mean']
        b=result['primary_contrasts'][f'sampled_present_dice:rejected_minus_{comparator}']['mean']
        lines.append(f"| rejected minus {comparator} | {a:+.6f} | {b:+.6f} | {bool(np.sign(a)==np.sign(b))} |")
    lines+=['',f"Frozen ranking criterion: {'passed' if result['ranking_discrimination_signal'] else 'not passed'}. Positive contrasts mean rejected-enriched deletion was safer. Effect direction and uncertainty, rather than equal absolute dataset scores, test transfer.",
        '','Group effects include changed attention normalization, overlap competition and redundant context. They do not establish individual-token usefulness or semantic purity. No budget/seed retuning or automatic efficacy follow-up.',
        '','Full per-video leave-one-out influence, split-interaction effects and per-object denominators are in results.json; record and PNG checks are in completion_review.json.']
    (out/'analysis.md').write_text('\n'.join(lines)+'\n')


def first_prompts(model, state, record, root, order=None):
    images = sorted((root/'JPEGImages/480p'/record['video']).glob('*.jpg'),key=lambda p:int(p.stem))
    if order is None:
        return add_first_frame_prompts(model,state,record['video'],root,record['object_ids'])
    labels,palette = author_mask_io()['load_ann_png'](str(root/'Annotations/480p'/record['video']/f'{images[0].stem}.png'))
    for oid in order:
        model.add_new_mask(state,frame_idx=0,obj_id=oid,mask=torch.from_numpy((labels==oid).copy()))
    return images,palette


def check_inputs(core, name, root):
    for row in core['design']['inputs'][name]:
        stem = Path(row['image']).stem
        if (digest(root/'JPEGImages/480p'/name/row['image']) != row['image_sha256']
                or digest(root/'Annotations/480p'/name/f'{stem}.png') != row['mask_sha256']):
            raise RuntimeError(f'Input changed: {name}/{stem}')


def source_guard(core):
    for rel,sha in core['sources'].items():
        if digest(ROOT/rel) != sha:
            raise RuntimeError(f'Source changed during DAVIS replication: {rel}')


@torch.inference_mode()
def run_video(record, root, checkpoint, out, core, *, gate=False, reference=None,
              instrument=True, order=None, replays=True):
    name = record['video']
    stage = out/'gate' if gate else out/'full'
    tag = ('reference' if not instrument else 'diagnostic') if order is None else 'order_'+'_'.join(map(str,order))
    directory = stage/'records'/name/tag
    directory.mkdir(parents=True,exist_ok=False)
    model = build_model(checkpoint)
    weights = parameter_digest(model)
    if model.non_overlap_masks_for_mem_enc:
        raise RuntimeError('Native memory writing unexpectedly couples instance masks')
    state = model.init_state(video_path=str(root/'JPEGImages/480p'/name),
        offload_video_to_cpu=True,offload_state_to_cpu=False,async_loading_frames=False)
    images,palette = first_prompts(model,state,record,root,order)
    ids = list(state['obj_ids'])
    expected_ids = record['object_ids'] if order is None else order
    if ids != expected_ids:
        raise RuntimeError('Persistent instance registration/order changed')
    prompt_record = {**record,'object_ids':ids}
    prompt_receipts = verify_prompt_inputs(model,state,prompt_record,root,'train' if gate else 'val')
    controller = InstanceTokenContributionRead(model,state,name) if instrument else None
    count = min(8,len(images)) if gate else len(images)
    plan = core['design']['sample_plan'][name]
    union = set().union(*(set(x['present']+x['absent']) for x in plan.values()))
    if gate:
        union = {3,6,7}
    archive = None if gate else core['native_reference']['videos'][name]
    records=[]; rows={a:[] for a in ARMS}; handles={}
    if replays:
        for arm in ARMS:
            path = directory/f'{arm}.jsonl.gz'
            handles[arm] = gzip.open(path,'wt')
    started=time.monotonic()
    try:
        iterator=iter(model.propagate_in_video(state,start_frame_idx=0,max_frame_num_to_track=count-1))
        for expected in range(count):
            if controller is not None:
                controller.arm='collect'
            with controller if controller is not None else nullcontext():
                frame,output_ids,original=next(iterator)
            if frame!=expected or list(output_ids)!=ids:
                raise RuntimeError('Native frame or instance order changed')
            signature={'frame':frame,'object_ids':ids,'full_logits_sha256':tensor_digest(original),
                       'raw':raw_signatures(state,frame,ids)}
            if reference is not None:
                prior=reference['records'][frame]
                if any(signature['raw'][str(oid)]!=prior['raw'][str(oid)] for oid in ids):
                    raise RuntimeError(f'Native per-object tensor parity failed: {name}/{frame}')
                if ids==reference['object_ids'] and signature['full_logits_sha256']!=prior['full_logits_sha256']:
                    raise RuntimeError('Native multi-object final logits changed under hooks')
            if archive is not None and signature['full_logits_sha256']!=archive['frames'][frame]['logits_sha256']:
                raise RuntimeError(f'Archived native DAVIS logits mismatch: {name}/{frame}')
            records.append(signature)
            official_scored=0<frame<count-1
            do_replay=replays and frame>0 and (official_scored if not gate else True)
            active_arms=(*PRIMARY,*SPLITS) if frame in union and do_replay else PRIMARY
            collected={};results={a:{} for a in active_arms};traces={a:{} for a in active_arms}
            if controller is not None and frame>0:
                if len(controller.calls)!=len(ids) or {c['object_id'] for c in controller.calls}!=set(ids):
                    raise RuntimeError('Native collector lost an instance or duplicated a read')
                collected={c['object_id']:c for c in controller.calls}
            if do_replay:
                before=bank_signature(state);cache_before=controller.cache_signature()
                for oid in ids:
                    sample=capture_training_read(model,state,frame,object_id=oid)
                    sample['object_id']=oid
                    native_trace=None
                    for arm in active_arms:
                        controller.arm=arm
                        with controller.replaying(oid):
                            result,trace=replay(model,sample,controller)
                        if arm=='native':
                            check_exact(result,stored(state,oid,frame));native_trace=trace
                            verify_trace(trace,collected[oid],oid)
                        else:
                            verify_trace(trace,native_trace,oid)
                        if frame==3 and arm=='random':
                            with controller.replaying(oid):
                                repeat,repeat_trace=replay(model,sample,controller)
                            check_exact(result,repeat)
                            if trace!=repeat_trace:
                                raise RuntimeError('Instance replay control is nondeterministic')
                        results[arm][oid]=result;traces[arm][oid]=trace
                    for layer in range(4):
                        plans={a:traces[a][oid]['selection_layers'][layer] for a in active_arms}
                        if not (plans['rejected']['counts']==plans['retained']['counts']==plans['random']['counts']):
                            raise RuntimeError('Per-instance key budgets differ between main conditions')
                    if gate and frame==7 and oid==ids[0]:
                        controller.only_owner=oid
                        asymmetric={}
                        try:
                            for other in ids:
                                other_sample=capture_training_read(model,state,frame,object_id=other)
                                controller.arm='rejected'
                                with controller.replaying(other):
                                    changed,trace=replay(model,other_sample,controller)
                                if other!=oid:
                                    check_exact(changed,stored(state,other,frame))
                                    if any(s['spatial_keys_after']!=s['spatial_keys_before'] for s in trace['selection_layers']):
                                        raise RuntimeError('Asymmetric intervention pruned the other object')
                                asymmetric[other]=changed['pred_masks']
                            _,joint,_=merged_prediction(model,state,asymmetric,ids)
                            _,native_joint,_=merged_prediction(model,state,{k:stored(state,k,frame)['pred_masks'] for k in ids},ids)
                            base.atomic_json(directory/'asymmetric_isolation.json',{'passed':True,'target_object_id':oid,
                                'raw_other_outputs_unchanged':True,'merged_other_pixels_changed':{
                                    str(k):int(np.count_nonzero((joint==k)!=(native_joint==k))) for k in ids if k!=oid},
                                'note':'Native argmax overlap competition may change visible other-object pixels without bank contamination.'})
                        finally:
                            controller.only_owner=None
                if bank_signature(state)!=before or controller.cache_signature()!=cache_before:
                    raise RuntimeError('Diagnostic replay modified a bank, cache or selector state')
            elif replays:
                for arm in active_arms:
                    results[arm]={oid:stored(state,oid,frame) for oid in ids}
            if replays:
                labels,_=author_mask_io()['load_ann_png'](str(root/'Annotations/480p'/name/f'{images[frame].stem}.png'))
                for arm in active_arms:
                    full,indexed,masks=merged_prediction(model,state,{oid:results[arm][oid]['pred_masks'] for oid in ids},ids)
                    if arm=='native' and tensor_digest(full)!=signature['full_logits_sha256']:
                        raise RuntimeError('Native joint replay differs after official overlap resolution')
                    output_dir=stage/'predictions'/arm
                    author_mask_io()['save_masks_to_dir'](str(output_dir),name,images[frame].stem,masks,
                        state['video_height'],state['video_width'],False,palette or author_mask_io()['DAVIS_PALETTE'])
                    path=output_dir/name/f'{images[frame].stem}.png'
                    png_sha=digest(path)
                    if archive is not None and arm=='native':
                        old_path=Path(core['native_reference']['prediction_dir'])/name/f'{images[frame].stem}.png'
                        if not np.array_equal(indexed,author_mask_io()['load_ann_png'](str(old_path))[0]):
                            raise RuntimeError('Native indexed pixels differ from archived same-setup baseline')
                    for oid in ids:
                        score=score_object(masks[oid],labels,oid)
                        trace=traces[arm].get(oid)
                        row={'frame':frame,'stem':images[frame].stem,'object_id':oid,'object_ids':ids,
                            'official_scored':official_scored,'sampled':frame in plan[str(oid)]['present']+plan[str(oid)]['absent'],
                            **score,'object_score_logit':float(results[arm][oid]['object_score_logits'].item()),
                            'raw_lowres_sha256':_tensor_sha256(results[arm][oid]['pred_masks']),
                            'indexed_png_sha256':png_sha,'indexed_png':str(path.relative_to(stage)), 'read':trace}
                        handles[arm].write(json.dumps(row,allow_nan=False)+'\n');handles[arm].flush()
                        rows[arm].append(row)
            if controller is not None:
                controller.calls.clear();controller.expire(frame)
            status(out,'engineering' if gate else 'cohort',video=name,frame=frame,total_frames=count,
                   object_ids=ids,gate_variant=tag)
            if frame%10==0 or frame==count-1:
                print(f"[{'gate' if gate else 'full'}:{tag}] {name} {frame}/{count-1} ids={ids}",flush=True)
        iterator.close()
    finally:
        for handle in handles.values():handle.close()
    if parameter_digest(model)!=weights:
        raise RuntimeError('Foundation model parameters changed')
    caches={str(oid):controller.cache_signature(oid) for oid in ids} if controller else {}
    result={'complete':True,'video':name,'object_ids':ids,'frames':count,'records':records,
        'parameter_sha256':weights,'prompt_inputs':prompt_receipts,'selector_cache_sha256':caches,
        'wall_seconds':time.monotonic()-started}
    if replays:
        if not gate:
            validate_coverage(rows,ids,plan,len(images))
        result['supplementary']={a:video_supplement(rows[a],ids,plan) for a in ARMS}
        result['record_sha256']={a:digest(directory/f'{a}.jsonl.gz') for a in ARMS}
    base.atomic_json(directory/'completion.json',result)
    model.reset_state(state)
    del model,state,controller,rows
    gc.collect()
    return result


def engineering(root, checkpoint, out, core):
    record=core['design']['engineering_inventory']
    reference=run_video(record,root,checkpoint,out,core,gate=True,instrument=False,replays=False)
    joint=run_video(record,root,checkpoint,out,core,gate=True,reference=reference)
    reverse=run_video(record,root,checkpoint,out,core,gate=True,reference=joint,
                      order=list(reversed(record['object_ids'])),replays=False)
    for oid in record['object_ids']:
        alone=run_video(record,root,checkpoint,out,core,gate=True,reference=joint,order=[oid],replays=False)
        if alone['selector_cache_sha256'][str(oid)]!=joint['selector_cache_sha256'][str(oid)]:
            raise RuntimeError('Another instance changed selector masks, usage or prompt anchors')
    if reverse['selector_cache_sha256']!=joint['selector_cache_sha256']:
        raise RuntimeError('Instance registration order changed the per-ID selector cache')
    with tempfile.TemporaryDirectory(dir=out/'gate',prefix='official_clip_fixture_') as temp:
        fixture=Path(temp)
        for folder,extension in (('JPEGImages/480p','.jpg'),('Annotations/480p','.png')):
            target=fixture/folder/record['video'];target.mkdir(parents=True)
            for file in record['frames'][:8]:
                name=Path(file).stem+extension
                (target/name).symlink_to(root/folder/record['video']/name)
        split=fixture/'ImageSets/2017';split.mkdir(parents=True)
        (split/'train.txt').write_text(record['video']+'\n')
        small={**record,'frames':record['frames'][:8]}
        metrics={a:run_official_evaluator(out/'gate/predictions'/a,fixture,[small],split='train') for a in PRIMARY}
        base.atomic_json(out/'gate/official_clip_metrics.json',{'scope':'8-frame training engineering fixture only; not a DAVIS benchmark score','conditions':metrics})
    source_guard(core)
    receipt={'passed':True,'objects':record['object_ids'],'frames':8,'variants':['native','collector/replay','reverse','each alone','asymmetric intervention'],
        'setup_sha256':digest(out/'setup.json'),'indexed_output_competition':'native non-overlap handling retained; raw instance isolation checked'}
    base.atomic_json(out/'engineering_pass.json',receipt)
    return receipt


def native_reference(path, design, checkpoint_sha):
    setup=json.loads((path/'setup.json').read_text())
    if (setup['checkpoint_sha256']!=checkpoint_sha or setup['model_config']!=CONFIG
            or not setup['indexed_mask_non_overlap_outputs'] or setup['evaluator_revision']!=EVALUATOR_REVISION):
        raise RuntimeError('Archived native baseline has a different setup')
    if setup['sam2_revision']!=design['inventory']['author_revision']:
        raise RuntimeError('Archived native author revision differs')
    for rel in ('external/sam2/sam2/modeling/sam2_base.py',
                'external/sam2/sam2/sam2_video_predictor.py','external/sam2/tools/vos_inference.py'):
        if digest(ROOT/rel)!=setup['sources'][rel]:
            raise RuntimeError('Archived native predictor source differs')
    videos={};prediction_dir=path/'conditions/native/predictions'
    for record in design['inventory']['videos']:
        name=record['video'];receipt=json.loads((path/'conditions/native'/f'{name}.complete.json').read_text())
        if (not receipt['parameters_unchanged'] or receipt['frame_count']!=len(record['frames'])
                or receipt['object_ids']!=record['object_ids'] or receipt['setup_sha256']!=digest(path/'setup.json')):
            raise RuntimeError('Archived native coverage/provenance differs')
        if ([r['frame'] for r in receipt['frames']]!=list(range(len(record['frames'])))
                or any(r['object_ids']!=record['object_ids'] for r in receipt['frames'])):
            raise RuntimeError('Archived native frame/object order is incomplete')
        for row in receipt['frames']:
            if digest(prediction_dir/row['prediction_png'])!=row['prediction_png_sha256']:
                raise RuntimeError('Archived native PNG checksum failed')
        videos[name]=receipt
    return {'path':str(path),'setup_sha256':digest(path/'setup.json'),
            'prediction_dir':str(prediction_dir),'videos':videos}


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    for name in ('dataset-root','checkpoint','output-dir','design-preflight','native-reference'):
        parser.add_argument('--'+name,type=Path,required=True)
    args=parser.parse_args();out=validate_output(args.output_dir);out.mkdir(parents=True,exist_ok=False)
    os.environ['TMPDIR']=str(out);torch.set_num_threads(2);torch.use_deterministic_algorithms(True)
    started=time.monotonic()
    try:
        status(out,'preflight')
        for rel in OWN_SOURCES:
            if subprocess.check_output(['git','show',f'HEAD:{rel}'],cwd=ROOT)!=(ROOT/rel).read_bytes():
                raise RuntimeError(f'Commit the unchanged protocol/integration before inference: {rel}')
        protocol=json.loads((ROOT/PROTOCOL).read_text());design=json.loads(args.design_preflight.read_text())
        if digest(args.design_preflight)!=protocol['design_preflight_sha256'] or digest(args.checkpoint)!=protocol['checkpoint_sha256']:
            raise RuntimeError('Frozen data design or checkpoint changed')
        if inventory(args.dataset_root,'davis2017','val')!=design['inventory']:
            raise RuntimeError('DAVIS inventory changed after target freeze')
        base.activate_author_repo()
        evaluator=ROOT/'external/davis2017-evaluation'
        if subprocess.check_output(['git','-C',str(evaluator),'rev-parse','HEAD'],text=True).strip()!=EVALUATOR_REVISION:
            raise RuntimeError('Official evaluator revision changed')
        if subprocess.check_output(['git','-C',str(evaluator),'status','--porcelain','--untracked-files=no']):
            raise RuntimeError('Official evaluator has tracked modifications')
        old_setup=json.loads(Path(protocol['polypgen_reference']['setup']).read_text())
        if (digest(Path(protocol['polypgen_reference']['setup']))!=protocol['polypgen_reference']['setup_sha256']
                or digest(Path(protocol['polypgen_reference']['results']))!=protocol['polypgen_reference']['results_sha256']):
            raise RuntimeError('PolypGen reference artifact changed')
        for rel in protocol['unchanged_selector_sources']:
            if digest(ROOT/rel)!=old_setup['sources'][rel]:
                raise RuntimeError('Run171 selector implementation changed')
        paths=sorted(set(base.SOURCE_REL)|set(EXTRA_SOURCES)|set(RUN171_SOURCES)|set(OWN_SOURCES)|set(EXTRA))
        core={'protocol':protocol,'design':design,'sources':{p:digest(ROOT/p) for p in paths},
            'checkpoint':str(args.checkpoint),'torch':torch.__version__,
            'commit':subprocess.check_output(['git','rev-parse','HEAD'],text=True).strip(),
            'native_reference':native_reference(args.native_reference,design,protocol['checkpoint_sha256'])}
        base.atomic_json(out/'setup.json',core)
        with zipfile.ZipFile(out/'source_snapshot.zip','w',zipfile.ZIP_DEFLATED) as archive:
            for rel in paths:archive.write(ROOT/rel,arcname=rel)
        check_inputs(core,'bmx-bumps',args.dataset_root)
        engineering(args.dataset_root,args.checkpoint,out,core)
        results={};official={a:{'per_video':{}} for a in PRIMARY}
        for record in design['inventory']['videos']:
            if not json.loads((out/'engineering_pass.json').read_text())['passed']:
                raise RuntimeError('Cohort requires a passing instance-aware engineering gate')
            source_guard(core);check_inputs(core,record['video'],args.dataset_root)
            completed=run_video(record,args.dataset_root,args.checkpoint,out,core)
            results[record['video']]=completed['supplementary']
            for arm in PRIMARY:
                metrics=run_official_evaluator(out/'full/predictions'/arm,args.dataset_root,[record])
                official[arm]['per_video'].update(metrics['per_video'])
            check_inputs(core,record['video'],args.dataset_root)
            base.atomic_json(out/'progress.json',{'completed_videos':list(results),'expected_videos':30})
            base.atomic_json(out/'partial_results.json',{'scope':'incomplete cohort; no decision','supplementary':results,'official':official})
        for arm in PRIMARY:
            official[arm]=run_official_evaluator(out/'full/predictions'/arm,args.dataset_root,design['inventory']['videos'])
        result=summarize(results,official)
        source_guard(core)
        base.atomic_json(out/'results.json',result)
        base.atomic_json(out/'completion_review.json',verify_saved(out,args.dataset_root,core,results))
        report(out,result,core)
        base.atomic_json(out/'completion.json',{'complete':True,'videos':30,
            'results_sha256':digest(out/'results.json'),'setup_sha256':digest(out/'setup.json'),
            'elapsed_seconds':time.monotonic()-started,'scope':result['scope']})
        status(out,'complete',completed_videos=30)
    except BaseException as exc:
        base.atomic_json(out/'failure.json',{'error':repr(exc),'no_bypass':True})
        status(out,'failed',error=repr(exc));raise


if __name__=='__main__':
    main()
