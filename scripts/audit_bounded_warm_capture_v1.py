#!/usr/bin/env python3
"""Audit separately versioned bounded seq6/seq11 warm full captures, no labels."""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image
import torch

ROOT = Path(__file__).resolve().parents[1]; sys.path.insert(0, str(ROOT))
from scripts.audit_crossfit_warm_video_bounded_base_v1 import checked_run
from scripts.certify_crossfit_warm_gate_v2 import verify_certificate
from scripts.crossfit_observer_capture_shared_v1 import DATA, OUT, PER_READ, phase_artifact, validate_plan
from scripts.memory_token_io import digest
from scripts.run_bounded_warm_capture_v1 import sources
from scripts.run_crossfit_warm_video_v2 import warm_plan
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.verify_bounded_loader_gate_v2 import verify_gate
from scripts.warm_crossfit_video_contract_v1 import read_bound, require


def compare_tree(left, right, path='root'):
    """Compare corresponding tensor trees with documented numerical tolerance."""
    if isinstance(left, torch.Tensor) and isinstance(right, torch.Tensor):
        require(left.shape == right.shape and left.dtype == right.dtype and
                torch.isfinite(left).all() and torch.isfinite(right).all(), f'Tensor geometry/finiteness differs: {path}')
        if left.is_floating_point():
            a,b=left.float(),right.float(); diff=(a-b).abs()
            maximum=float(diff.max()) if diff.numel() else 0.; mean=float(diff.mean()) if diff.numel() else 0.
            return [{'path':path,'max_abs':maximum,'mean_abs':mean,
                     'within_tolerance':bool(torch.allclose(a,b,rtol=1e-5,atol=1e-6))}]
        return [{'path':path,'max_abs':0.,'mean_abs':0.,'within_tolerance':bool(torch.equal(left,right))}]
    if isinstance(left,dict) and isinstance(right,dict):
        require(set(left)==set(right), f'Tensor tree keys differ: {path}')
        return [record for key in sorted(left) for record in compare_tree(left[key],right[key],f'{path}.{key}')]
    if isinstance(left,(list,tuple)) and type(left) is type(right):
        require(len(left)==len(right), f'Tensor tree length differs: {path}')
        return [record for i,(a,b) in enumerate(zip(left,right,strict=True)) for record in compare_tree(a,b,f'{path}[{i}]')]
    return [{'path':path,'max_abs':0.,'mean_abs':0.,'within_tolerance':type(left) is type(right) and left==right}]


def compare_observations(left, right, frames):
    results=[]
    for frame in frames:
        a=next(x for x in left['observations'] if x['frame']==frame)
        b=next(x for x in right['observations'] if x['frame']==frame)
        for kind in ('feature','snapshot'):
            loaded=[]
            for row in (a,b):
                path=Path(row[kind+'_file']).resolve(strict=True); sha=row[kind+'_sha256']
                require(digest(path)==sha, 'Captured observation changed')
                loaded.append(torch.load(path,map_location='cpu',weights_only=True))
                require(digest(path)==sha, 'Captured observation changed while loading')
            detail=compare_tree(*loaded)
            results.append({'frame':frame,'kind':kind,'max_abs':max((x['max_abs'] for x in detail),default=0.),
                            'all_within_tolerance':all(x['within_tolerance'] for x in detail),
                            'mismatches':[x['path'] for x in detail if not x['within_tolerance']]})
    return results


def compare_masks(left, left_folder, right, right_folder):
    require([r['frame'] for r in left['rows']]==[r['frame'] for r in right['rows']], 'Mask frame coverage differs')
    result=[]
    for a,b in zip(left['rows'],right['rows'],strict=True):
        pa=Path(left_folder)/f"{a['frame']:06d}.png"; pb=Path(right_folder)/f"{b['frame']:06d}.png"
        require(digest(pa)==a['mask_sha256'] and digest(pb)==b['mask_sha256'], 'Mask changed')
        with Image.open(pa) as x,Image.open(pb) as y:
            x=np.asarray(x.convert('L'))>0;y=np.asarray(y.convert('L'))>0
        require(x.shape==y.shape and digest(pa)==a['mask_sha256'] and digest(pb)==b['mask_sha256'],
                'Mask shape or bytes changed')
        total=int(x.sum()+y.sum()); difference=int(np.count_nonzero(x!=y))
        agreement=2*int(np.count_nonzero(x&y))/total if total else 1.
        result.append({'frame':a['frame'],'differing_pixels':difference,'mask_dice':agreement,
                       'left_pixels':int(x.sum()),'right_pixels':int(y.sum())})
    return result


def compare_prompt_artifacts(left, left_folder, right, right_folder):
    """Check the committed prompt and report early/prototype/point-mask differences."""
    tensors={}
    for name, field in (('prompt_early.pt','prompt_early_sha256'),('prototypes.pt','prototypes_sha256')):
        values=[]
        for record,folder in ((left,left_folder),(right,right_folder)):
            path=Path(folder)/name;sha=record[field];require(digest(path)==sha,'Prompt tensor changed')
            values.append(torch.load(path,map_location='cpu',weights_only=True))
            require(digest(path)==sha,'Prompt tensor changed during load')
        detail=compare_tree(*values)
        tensors[name]={'max_abs':max((x['max_abs'] for x in detail),default=0.),
                       'all_within_tolerance':all(x['within_tolerance'] for x in detail),
                       'mismatches':[x['path'] for x in detail if not x['within_tolerance']]}
    masks={}
    for name, field in (('conditioning_mask.png','conditioning_sha256'),('generated_mask.png','generated_sha256')):
        arrays=[]
        for record,folder in ((left,left_folder),(right,right_folder)):
            path=Path(folder)/name;sha=record[field];require(digest(path)==sha,'Prompt mask changed')
            with Image.open(path) as image:arrays.append(np.asarray(image.convert('L'))>0)
            require(digest(path)==sha,'Prompt mask changed during decode')
        require(arrays[0].shape==arrays[1].shape,'Prompt mask shape differs')
        masks[name]={'differing_pixels':int(np.count_nonzero(arrays[0]!=arrays[1])),
                     'left_pixels':int(arrays[0].sum()),'right_pixels':int(arrays[1].sum())}
    return {'tensors':tensors,'masks':masks}


def audit(root, sequence, certificate_path, short_gate_path):
    from scripts.run_bounded_warm_capture_study_v1 import phases, verify_seq6_proof
    root=Path(root).resolve(strict=True); require(root.is_relative_to(OUT.resolve(strict=True)), 'Invalid full capture root')
    require(sequence in ('seq6','seq11'), 'Unsupported bounded full sequence')
    manifest_path=root/'run_manifest.json';manifest_sha=digest(manifest_path);manifest=read_bound(manifest_path,manifest_sha)
    warm=warm_plan(); index=validate_plan(); certificate, cert_sha=verify_certificate(certificate_path)
    gate, gate_sha=verify_gate(short_gate_path); require(gate['certificate_sha256']==cert_sha,'Short gate certificate differs')
    src=sources(); receipts={}; gib=1024**3
    expected_phases=[f'{sequence}_reference',f'{sequence}_capture',f'{sequence}_audit']
    read_count=sum(x['sequence']==sequence for x in index['fresh_reads'])
    require(manifest.get('sequence')==sequence and manifest.get('certificate_path')==str(Path(certificate_path).resolve()) and
            manifest.get('certificate_sha256')==cert_sha and manifest.get('short_gate_path')==str(Path(short_gate_path).resolve()) and
            manifest.get('short_gate_sha256')==gate_sha and manifest.get('source_sha256')==src and
            manifest.get('phases')==expected_phases and type(manifest.get('cache_frames')) is int and
            1<=manifest['cache_frames']<=4 and type(manifest.get('min_disk_bytes')) is int and
            manifest['min_disk_bytes']>=26*gib and type(manifest.get('min_available_ram_bytes')) is int and
            manifest['min_available_ram_bytes']>=4*gib and type(manifest.get('max_process_rss_bytes')) is int and
            0<manifest['max_process_rss_bytes']<=6*gib and read_count>0 and
            manifest.get('planned_capture_reads')==read_count and
            manifest.get('planned_payload_bytes')==read_count*PER_READ,'Bounded full capture manifest differs')
    snapshot_path=root/'source_snapshot/snapshot_manifest.json'
    snapshot=read_bound(snapshot_path,manifest['source_snapshot_manifest_sha256'])
    snapshot_map=snapshot.get('source_sha256',{})
    author_files=('external/Lean-SAM2-token-window/sam2/sam2_video_predictor_swr_wrapper.py',
                  'external/Lean-SAM2-token-window/sam2/sam2_video_predictor_topk_insurance.py',
                  'external/Lean-SAM2-token-window/sam2/utils/misc.py')
    require(set(snapshot_map)==set(src)|set(author_files) and
            all(snapshot_map.get(name)==sha for name,sha in src.items()) and
            all(snapshot_map[name]==digest(ROOT/name) for name in author_files),'Full capture source snapshot differs')
    for name,sha in snapshot_map.items():
        path=(root/'source_snapshot'/name).resolve(strict=True)
        require(path.is_relative_to((root/'source_snapshot').resolve()) and digest(path)==sha,
                'Full capture source snapshot changed')
    phase_rows=[json.loads(x) for x in (root/'phases.jsonl').read_text().splitlines() if x.strip()]
    names=[x.get('phase') for x in phase_rows]
    require(names in (expected_phases[:2],expected_phases), 'Bounded full capture phase order differs')
    if len(names)==3: phase_artifact(root,expected_phases[2],root/sequence/'audit/audit.json')
    for row in phase_rows:
        monitor=row.get('resource_monitor',{})
        require(type(monitor.get('minimum_sampled_disk_free_bytes')) is int and
                monitor['minimum_sampled_disk_free_bytes']>=manifest['min_disk_bytes'] and
                type(monitor.get('minimum_sampled_system_available_ram_bytes')) is int and
                monitor['minimum_sampled_system_available_ram_bytes']>=manifest['min_available_ram_bytes'] and
                type(monitor.get('peak_sampled_process_rss_bytes')) is int and
                0<=monitor['peak_sampled_process_rss_bytes']<=manifest['max_process_rss_bytes'] and
                type(monitor.get('minimum_sampled_payload_margin_bytes')) is int and
                monitor['minimum_sampled_payload_margin_bytes']>=0 and
                monitor.get('planned_payload_bytes')==manifest['planned_payload_bytes'] and
                type(monitor.get('written_payload_bytes')) is int and
                0<=monitor['written_payload_bytes']<=manifest['planned_payload_bytes'],
                'Bounded full capture resource evidence differs')
    seq6_proof_sha=None
    if sequence=='seq11':
        _,seq6_proof_sha=verify_seq6_proof(manifest['seq6_proof_path'],cert_sha,gate_sha)
        require(seq6_proof_sha==manifest.get('seq6_proof_sha256'),'Seq6 proof differs')
    for mode in ('reference','capture'):
        folder=root/sequence/mode; side=folder/'bounded_receipt.json'
        original=folder/'receipt.json'; original_sha=phase_artifact(root,f'{sequence}_{mode}',original)
        phase=next(x for x in phase_rows if x['phase']==f'{sequence}_{mode}')
        side_sha=phase.get('bounded_receipt_sha256'); record=read_bound(side,side_sha)
        require(record.get('passed') is True and record.get('sequence')==sequence and record.get('mode')==mode and
            record.get('original_receipt_sha256')==original_sha and record.get('certificate_sha256')==cert_sha and
            record.get('short_gate_sha256')==gate_sha and record.get('source_sha256')==src and
            record.get('label_guard',{}).get('open_attempts')==0 and
            type(record.get('peak_process_rss_bytes')) is int and
            0<record['peak_process_rss_bytes']<=manifest['max_process_rss_bytes'] and
            record.get('cache',{}).get('final_sha256_verified') is True and record['cache'].get('evictions',0)>0,
            'Bounded full-capture receipt differs')
        run, run_sha=checked_run(root,sequence,mode,index,warm,certificate['warm_gate_sha256'],cert_sha)
        require(run_sha==original_sha and run.get('bounded_base_source_sha256')==
                digest(ROOT/'scripts/run_crossfit_warm_video_bounded_base_v1.py') and
                run.get('process_id')==record.get('process_id') and run.get('run_uuid')==record.get('run_uuid'),
                'Bounded base capture phase/source changed')
        receipts[mode]={'run':run,'folder':folder,'side':record,'original_sha256':run_sha,'bounded_sha256':side_sha}
    ref,cap=receipts['reference'],receipts['capture']
    require(cap['run'].get('reference_receipt_sha256')==ref['original_sha256'] and
            ref['side']['process_id'] != cap['side']['process_id'] and ref['side']['run_uuid'] != cap['side']['run_uuid'],
            'Full captures were not separate processes')
    same_trace={k:ref['run'][k]==cap['run'][k] for k in ('positions','selection','anchor_events','token_window_events')}
    prompt=compare_prompt_artifacts(ref['run'],ref['folder'],cap['run'],cap['folder'])
    mask_pairs={'reference_vs_observer':compare_masks(ref['run'],ref['folder'],cap['run'],cap['folder'])}
    historical_observations=[]; historical_trace=None
    if sequence=='seq6':
        old=certificate['seq6_historical_reference']; historical=read_bound(old['reference_path'],old['reference_sha256'])
        mask_pairs['bounded_vs_historical_eager']=compare_masks(ref['run'],ref['folder'],historical,Path(old['reference_path']).parent)
        historical_trace={k:ref['run'][k]==historical[k] for k in ('positions','selection','anchor_events','token_window_events')}
        current=certificate['seq6_gate_capture']; oldcap=read_bound(current['receipt_path'],current['receipt_sha256'])
        phase_artifact(Path(current['receipt_path']).parent.parent.parent,'seq6_capture',current['receipt_path'])
        frame_ids=[r['frame'] for r in cap['run']['observations']]
        require(frame_ids == [r['frame'] for r in oldcap['observations']] and {24,90}.issubset(frame_ids),
                'Seq6 gate overlap grid differs')
        historical_observations=compare_observations(cap['run'],oldcap,frame_ids)
    # Per-frame Dice compares predicted masks only; it is not a ground-truth score.
    minimum_mask_dice=min((x['mask_dice'] for rows in mask_pairs.values() for x in rows),default=1.)
    passing=(all(same_trace.values()) and (historical_trace is None or all(historical_trace.values())) and
             all(x['all_within_tolerance'] for x in prompt['tensors'].values()) and
             prompt['masks']['conditioning_mask.png']['differing_pixels']==0 and
             all(x['all_within_tolerance'] for x in historical_observations) and minimum_mask_dice >= .995)
    require(digest(manifest_path)==manifest_sha and sources()==src and verify_certificate(certificate_path)[1]==cert_sha and
            verify_gate(short_gate_path)[1]==gate_sha,'Bounded full capture sources changed during audit')
    return {'passed': passing, 'scope':'Same-device bounded capture compatibility, no labels; not full state parity or efficacy.',
        'sequence':sequence,'certificate_sha256':cert_sha,'short_gate_sha256':gate_sha,
        'manifest_sha256':manifest_sha,'seq6_proof_sha256':seq6_proof_sha,
        'reference_sha256':ref['original_sha256'],'reference_bounded_sha256':ref['bounded_sha256'],
        'capture_sha256':cap['original_sha256'],'capture_bounded_sha256':cap['bounded_sha256'],
        'source_sha256':src,'same_capture_traces':same_trace,'historical_traces':historical_trace,
        'prompt_comparison':prompt,
        'mask_agreement_minimum_dice':.995,'minimum_observed_mask_dice':minimum_mask_dice,
        'mask_pairs':mask_pairs,'historical_observations':historical_observations,
        'auditor_sha256':digest(Path(__file__))}


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root',type=Path,required=True); p.add_argument('--sequence',choices=('seq6','seq11'),required=True)
    p.add_argument('--certificate',type=Path,required=True); p.add_argument('--short-gate',type=Path,required=True)
    p.add_argument('--output',type=Path,required=True);a=p.parse_args();out=a.output.resolve()
    require(out.is_relative_to(OUT.resolve(strict=True)) and not out.exists(),'Invalid output')
    guard=install_mask_open_guard(DATA);result=audit(a.root,a.sequence,a.certificate,a.short_gate)
    require(guard['open_attempts']==0,'Target label opened');result['label_guard']=dict(guard)
    out.mkdir(parents=True,exist_ok=False);(out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'output':str(out)}))
    require(result['passed'],'Bounded full compatibility failed; diagnostics retained')


if __name__=='__main__':main()
