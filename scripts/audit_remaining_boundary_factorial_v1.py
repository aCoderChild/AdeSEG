#!/usr/bin/env python3
"""Independent CPU verification of frozen extended-video scores.

No target mask is opened until all 8 prediction sets, both short gates,
complete launcher logs and the primary score receipt have been checked.
Boundary F and segmentation metrics use the independent SciPy-based auditor.
"""
import argparse
import json
import math
from pathlib import Path
import sys

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import validate_output
from scripts.run_samurai_automatic_priority import MANIFEST
from scripts.run_remaining_boundary_factorial_v1 import ARMS, SEQUENCES
from scripts.audit_boundary_video_factorial_v1 import (sha, require, label, boundary_target,
                                                       metrics, summarize, paired, compare)


def read(path): return json.loads(Path(path).read_text())


def telemetry_stats(records, start, frame_count):
    later=[x for x in records if not x['conditioning'] and x['frame']>start]
    require([x['frame'] for x in later] == list(range(start+1,frame_count)), 'Telemetry coverage differs')
    for x in later:
        require(type(x.get('native_object_present')) is bool and type(x.get('native_exact_suppressed')) is bool
                and type(x.get('object_score_logit')) in (int,float) and math.isfinite(x['object_score_logit'])
                and x['native_object_present'] == (x['object_score_logit']>0), 'Invalid presence telemetry')
        for key in ('native','refined','output','memory_high_res'):
            values=x.get(key,{})
            require(type(values.get('numel')) is int and values['numel']>0 and
                type(values.get('positive_pixels')) is int and 0<=values['positive_pixels']<=values['numel'] and
                all(type(values.get(k)) in (int,float) and math.isfinite(values[k]) for k in ('min','max')) and
                values['min']<=values['max'] and isinstance(values.get('sha256'),str) and len(values['sha256'])==64,
                'Invalid logit telemetry')
    return {'frames':len(later),'native_absence_frames':sum(not x['native_object_present'] for x in later),
        'native_exact_suppressed_frames':sum(x['native_exact_suppressed'] for x in later),
        'native_absence_refiner_positive_frames':sum(not x['native_object_present'] and x['refined']['positive_pixels']>0 for x in later),
        'native_absence_output_positive_frames':sum(not x['native_object_present'] and x['output']['positive_pixels']>0 for x in later),
        'native_absence_memory_positive_frames':sum(not x['native_object_present'] and x['memory_high_res']['positive_pixels']>0 for x in later)}


def precheck(run):
    """Check source, gates, coverage, masks and routing before any labels."""
    run=Path(run).resolve(strict=True)
    from scripts.resume_source_group_capture_v1 import BASE
    require(run.is_relative_to(BASE.resolve(strict=True)), 'Run outside experiment root')
    launch_path=run/'run_manifest.json'; map_path=run/'run_map.json'
    launch,mapping=read(launch_path),read(map_path)
    score_path=run/'scores/results.json'; report=run/'scores/report.md'; saved=read(score_path)
    require(saved.get('passed') is True and saved.get('scorer_sha256') == sha(ROOT/'scripts/score_remaining_boundary_factorial_v1.py')
            and saved.get('report_sha256') == sha(report) and saved.get('run_map_sha256') == sha(map_path)
            and saved.get('launch_manifest_sha256') == sha(launch_path), 'Primary scorer or report differs')
    from scripts.run_remaining_boundary_factorial_v1 import study_sources
    source_paths=study_sources()
    require(launch.get('source_sha256') == {k:sha(v) for k,v in source_paths.items()}, 'Launch sources changed')
    require(launch.get('run_map') == mapping and launch.get('sequences') == list(SEQUENCES) and
            launch.get('arms') == list(ARMS) and launch.get('mps_fallback') == '0' and
            launch.get('mps_fast_math') == '0' and launch.get('python_hash_seed') == '0', 'Launch config differs')
    phases=[json.loads(x) for x in (run/'phases.jsonl').read_text().splitlines() if x.strip()]
    expected=[f'{s}_{a}' for s in SEQUENCES for a in ARMS]+['score']
    require([p.get('phase') for p in phases] == expected and all(p.get('passed') is True and
            p.get('log_sha256') == sha(run/(p['phase']+'.log')) for p in phases), 'Phase/log sequence differs')
    expected_inputs=('preflight','gate','gate_audit','extension_gate','extension_audit')
    require(set(launch.get('inputs',{})) == set(expected_inputs), 'Launch inputs incomplete')
    inputs={k:Path(launch['inputs'][k]['path']).resolve(strict=True) for k in expected_inputs}
    from scripts.preflight_remaining_boundary_video_v1 import load_extended_preflight
    from scripts.run_remaining_boundary_factorial_v1 import verify_original_gate,verify_extended_gate,CHECKPOINT,DETECTOR,DETECTOR_SHA256,STUDY,input_digest
    require(all(input_digest(k,p) == launch['inputs'][k]['sha256'] for k,p in inputs.items()) and
            sha(inputs['preflight']/'completion.json') == launch['preflight_completion_sha256'],'Input gate/preflight changed')
    load_extended_preflight(inputs['preflight'],CHECKPOINT,DETECTOR); verify_original_gate(inputs['gate'],inputs['gate_audit'])
    verify_extended_gate(inputs['preflight'],inputs['extension_gate'],inputs['extension_audit'],inputs['gate'])
    pre=read(inputs['preflight']/'manifest.json'); require(pre.get('selection') == list(SEQUENCES),'Frozen selection differs')
    original_identity=read(inputs['gate'])['runtime_identity']
    require(original_identity == read(inputs['extension_gate'])['runtime_identity'], 'Engineering runtime identities differ')
    require(saved.get('preflight_sha256') == sha(inputs['preflight']/'manifest.json') and
            saved.get('engineering_gate_sha256') == sha(inputs['gate']) and
            saved.get('extension_gate_sha256') == sha(inputs['extension_gate']), 'Scored gate binding differs')
    require(launch.get('checkpoint_sha256') == sha(CHECKPOINT) and launch.get('detector_sha256') ==
            DETECTOR_SHA256 == sha(DETECTOR),'Model/detector identity differs')
    videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')}
    require(set(mapping) == set(SEQUENCES) and set(saved.get('rows',{})) == set(SEQUENCES),'Sequence coverage differs')
    seen=set(); receipts={}; runtime_identity=None; count=0
    for seq in SEQUENCES:
        require(set(mapping[seq]) == set(ARMS),'Arm coverage differs')
        frames=videos[seq]; preseq=pre['sequences'][seq]
        require(preseq['frame_count'] == len(frames) == len(saved['rows'][seq]),'Frame coverage differs')
        expected_images=[]
        for i,f in enumerate(frames):
            entry=preseq['frames'][i]; value=sha(f.image_path)
            require(entry == {'frame':i,'source_frame_index':f.frame_index,'image_path':str(f.image_path.resolve()),
                              'image_sha256':value},'Source image identity differs')
            expected_images.append({'frame':i,'source_frame_index':f.frame_index,'sha256':value})
        receipts[seq]={}
        for arm in ARMS:
            folder=Path(mapping[seq][arm]).resolve(strict=True)
            require(folder not in seen and folder.is_relative_to(run),'Aliased/outside output'); seen.add(folder)
            rec_path=folder/'completion.json'; rec=read(rec_path)
            expected_rec={'passed':True,'full_video':True,'limit':0,'sequence':seq,'arm':arm,
                'runner_source_sha256':sha(source_paths['runner']), 'prompt_frame':preseq['first_detection_index'],
                'prompt_sha256':preseq['prompt_metadata_sha256'],'prompt_mask_sha256':preseq['prompt_mask_sha256'],
                'view':preseq['view'],'checkpoint_sha256':sha(CHECKPOINT),'detector_sha256':DETECTOR_SHA256,
                'candidate_checkpoint_sha256':sha(STUDY/'runs/region_boundary/checkpoint.pt'),
                'control_checkpoint_sha256':sha(STUDY/'runs/mask_only_boundary/checkpoint.pt'),
                'engineering_gate_sha256':sha(inputs['gate']),'extended_preflight_sha256':sha(inputs['preflight']/'manifest.json'),
                'extension_gate_sha256':sha(inputs['extension_gate']),
                'extension_audit_sha256':sha(inputs['extension_audit'])}
            require(all(rec.get(k) == v for k,v in expected_rec.items()) and
                    rec.get('label_guard',{}).get('open_attempts') == 0 and
                    rec.get('input_images') == expected_images and
                    saved['run_receipt_sha256'][seq][arm] == sha(rec_path),'Run identity differs')
            start=rec['prompt_frame']
            require([r['frame'] for r in rec['masks']] == list(range(len(frames))) and
                    [r['frame'] for r in rec['post_prompt_state']] == list(range(start,len(frames))), 'Coverage differs')
            for k,n in (('runtime_verification_sha256','verification/runtime_verification.json'),
                ('memory_events_sha256','memory_events.json'),('positions_sha256','positions.json'),
                ('selections_sha256','selections.json'),('telemetry_sha256','telemetry.json')):
                require(rec[k] == sha(folder/n),f'Frozen {n} differs')
            runtime=read(folder/'verification/runtime_verification.json')
            require(runtime.get('passed') and runtime.get('candidate_checkpoint_sha256') == rec['candidate_checkpoint_sha256'] and
                runtime.get('device') == 'mps float32' and runtime.get('fallback') == 'disabled' and
                runtime.get('fast_math') == 'disabled' and runtime.get('memory_budget') == 16384 and
                runtime.get('insurance_reserve') == 3072 and runtime.get('baseline_replays') == 2,'Runtime identity differs')
            require(runtime.get('source_sha256') and all(sha(ROOT/k) == v for k,v in runtime['source_sha256'].items()),
                    'Live runtime source changed')
            require(runtime == original_identity, 'Runtime does not match both engineering gates')
            if runtime_identity is None: runtime_identity=runtime
            require(runtime == runtime_identity,'Runtime varies across runs')
            for name in ('model','refiner'):
                require(sha(folder/f'verification/{name}_tensor_fingerprints.json') ==
                        runtime[name+'_tensor_fingerprints_sha256'],'Model fingerprints differ')
            for row in rec['masks']:
                i=row['frame']; path=folder/f'{i:06d}.png'
                require(sha(path) == row['sha256'],'Prediction hash differs')
                with Image.open(path) as mask, Image.open(frames[i].image_path) as image:
                    arr=np.asarray(mask.convert('L'))
                    require(mask.format == 'PNG' and mask.size == image.size and np.isin(arr,(0,255)).all(),
                            'Prediction format/geometry differs')
                    if i<start: require(not arr.any(),'Pre-prompt output is not empty')
                count+=1
            receipts[seq][arm]=rec
        a,b=receipts[seq]['A'],receipts[seq]['B']
        for i,row in enumerate(saved['rows'][seq]):
            require(row.get('frame') == i and row.get('source_frame_index') == frames[i].frame_index and
                    row.get('post_prompt') is (i > a['prompt_frame']) and
                    row.get('initialization_frame') is (i == a['prompt_frame']), 'Scored frame mapping differs')
        for arm in 'BCD':
            cur=receipts[seq][arm]
            for key in ('mask_sha256','pred_masks_sha256','maskmem_features_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                require(cur['post_prompt_state'][0].get(key) is not None and
                        cur['post_prompt_state'][0][key] == a['post_prompt_state'][0][key],'Conditioning changed')
            for key in ('pred_masks_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                require(cur['post_prompt_state'][1].get(key) is not None and
                        cur['post_prompt_state'][1][key] == a['post_prompt_state'][1][key],'First decoder state changed')
        first={arm:receipts[seq][arm]['post_prompt_state'][1] for arm in ARMS}
        require(first['A']['output_logits_sha256']==first['C']['output_logits_sha256'] and
                first['B']['output_logits_sha256']==first['D']['output_logits_sha256'] and
                first['C']['maskmem_features_sha256']==first['D']['maskmem_features_sha256'],'Routes differ')
        for x,y in zip(b['post_prompt_state'],a['post_prompt_state'],strict=True):
            for key in ('pred_masks_sha256','maskmem_features_sha256','obj_ptr_sha256','object_score_logits_sha256'):
                require(x.get(key) is not None and x[key] == y.get(key),'B changed baseline state')
        for name in ('memory_events.json','positions.json','selections.json'):
            require(read(Path(mapping[seq]['A'])/name)==read(Path(mapping[seq]['B'])/name),'B changed memory routing')
    return run,mapping,videos,pre,receipts,saved,count


def audit(run):
    run,mapping,videos,pre,receipts,saved,count=precheck(run)
    errors,recomputed,labels_checked=[],{},0
    for seq in SEQUENCES:
        meta=pre['sequences'][seq]; metadata_path=Path(meta['archive_frames_path'])
        require(sha(metadata_path)==meta['archive_frames_sha256'],'Archived label metadata changed')
        metadata=read(metadata_path); require(len(metadata)==len(videos[seq]),'Label coverage differs')
        rows=[]; start=receipts[seq]['A']['prompt_frame']
        for i,frame in enumerate(videos[seq]):
            require(sha(frame.mask_path)==metadata[i]['label_sha256'],'Label hash differs')
            gt=label(frame.mask_path); labels_checked+=1; target=boundary_target(gt)
            row={'post_prompt':i>start}
            for arm in ARMS:
                with Image.open(Path(mapping[seq][arm])/f'{i:06d}.png') as im:
                    pred=np.asarray(im.convert('L'))>0
                row[arm]=metrics(pred,gt,target)
                compare(errors,[seq,i,arm],row[arm],saved['rows'][seq][i][arm])
            rows.append(row)
        recomputed[seq]={}
        for arm in ARMS:
            recomputed[seq][arm]={}
            for name,post in (('post_prompt',True),('all',False)):
                value=summarize(rows,arm,post)
                compare(errors,[seq,arm,name],value,saved['summaries'][seq][arm][name])
                recomputed[seq][arm][name]=value
        for name,arm,base in (('B_minus_A','B','A'),('C_minus_A','C','A'),('D_minus_B','D','B'),('D_minus_A','D','A')):
            compare(errors,[seq,name],paired(rows,arm,base),saved['paired_descriptive'][seq][name])
        compare(errors,[seq,'preprompt'],{'frames':start,'present_misses':sum(r['A']['present'] for r in rows[:start])},saved['preprompt'][seq])
        for arm in ARMS:
            records=read(Path(mapping[seq][arm])/'telemetry.json')
            stats=telemetry_stats(records,start,len(videos[seq]))
            compare(errors,[seq,arm,'telemetry'],stats,saved['suppression_telemetry'][seq][arm])
    return {'passed':not errors,'scope':'CPU artifact/metric audit; independent SciPy metric implementation; gate validation reuses reviewed gate helpers. Previously inspected patients, not blinded.',
        'run':str(run),'score_sha256':sha(run/'scores/results.json'),'run_manifest_sha256':sha(run/'run_manifest.json'),
        'run_map_sha256':sha(run/'run_map.json'),'audit_source_sha256':sha(Path(__file__)),
        'label_files_checked':labels_checked,'prediction_files_checked':count,'error_count':len(errors),
        'errors':errors,'summaries':recomputed}


def main():
    p=argparse.ArgumentParser(description=__doc__); p.add_argument('--run',type=Path,required=True); p.add_argument('--output',type=Path,required=True)
    args=p.parse_args(); result=audit(args.run); out=validate_output(args.output); out.mkdir(parents=True,exist_ok=False)
    (out/'audit.json').write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'errors':result['error_count'],'output':str(out)}))
    if not result['passed']: raise SystemExit(1)


if __name__=='__main__': main()
