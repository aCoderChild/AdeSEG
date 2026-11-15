#!/usr/bin/env python3
"""CPU-only, label-free input/duplicate preflight for exploratory captures.

This does not run a model, open mask labels, or establish patient independence.
The old seq6/13-only runner cannot consume this manifest without a new version.
"""
import argparse,hashlib,json,re,shutil,sys
from collections import defaultdict
from pathlib import Path
from PIL import Image
import torch
import numpy as np
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output
from scripts.train_region_mixed_loss_v1 import mixed_manifest
from scripts.train_multilevel_refiner_seq4_empty_v2 import read_capture

CONFIG=ROOT/'protocols/source_group_capture_v1.json'
SOURCES=('scripts/preflight_expanded_train_capture_v1.py','scripts/run_expanded_train_capture_v1.py',
         'scripts/audit_expanded_train_capture_v1.py','modeling/lean_mps_decoder_observer.py',
         'modeling/lean_indexed_spatial_rope_mps.py','modeling/lean_mps_blocking_pointer_position.py',
         'modeling/prompt_feature_refinement.py','datasets/video.py','datasets/io.py',
         'protocols/SOURCE_GROUP_CAPTURE_FEASIBILITY_20261114.md',
         'scripts/train_region_mixed_loss_v1.py','scripts/train_multilevel_refiner_seq4_empty_v2.py','scripts/preflight_mixed_refiner_v1.py',
         'scripts/audit_lean_window_mps_full.py','scripts/audit_lean_window_mps_replay.py','scripts/verify_lean_window_mps_replay.py')

def verify_config(c):
    expected={'seq2','seq3','seq4','seq5','seq12','seq13','seq14','seq9','seq15'}
    if c.get('format')!='source_group_capture_exploratory_v1' or set(c.get('sequences',{}))!=expected:raise ValueError('Config cohort differs')
    if set(c.get('protected_sequences',[]))!={'seq8','seq11','seq20','seq23'} or set(c.get('duplicate_screen_sequences',[]))!={'seq8','seq11','seq16','seq17','seq18','seq19','seq20','seq21','seq22','seq23'}:raise ValueError('Protected screen differs')
    roles={name:s['split'] for name,s in c['sequences'].items()}
    if {x for x,v in roles.items() if v=='train'}!={'seq2','seq3','seq4','seq5','seq12','seq13','seq14'} or {x for x,v in roles.items() if v=='validation'}!={'seq9','seq15'}:raise ValueError('Sequence roles differ')
    expected_groups={'seq2':'prefix0','seq3':'prefix1','seq4':'prefix1','seq5':'seq5_unknown','seq12':'seq12_unknown','seq13':'seq13_unknown','seq14':'seq14_unknown','seq9':'prefix7','seq15':'seq15_unknown'}
    groups=defaultdict(set)
    for name,s in c['sequences'].items():
        if s['split'] not in ('train','validation') or not s['targets'] or len(s['targets'])!=len(set(s['targets'])):
            raise ValueError('Invalid split or targets')
        if min(s['targets'])<=s['prompt_frame'] or max(s['targets'])>=s['frames'] or s['targets']!=sorted(s['targets']):raise ValueError('Invalid capture indices')
        count=min(25,s['frames']-s['prompt_frame']-1)
        expected_targets=np.rint(np.linspace(s['prompt_frame']+1,s['frames']-1,count)).astype(int).tolist()
        if s['targets']!=expected_targets or s['source_group_hypothesis']!=expected_groups[name]:raise ValueError('Sampling rule or source hypothesis differs')
        groups[s['source_group_hypothesis']].add(s['split'])
    if any(len(parts)>1 for parts in groups.values()):raise ValueError('Suspected source group spans splits')
    if any(x in c['sequences'] for x in ('seq6','seq7','seq8','seq10','seq11','seq20','seq23')):raise ValueError('Protected or excluded sequence present')
    if sum(len(x['targets']) for x in c['sequences'].values() if x['split']=='train')!=164 or sum(len(x['targets']) for x in c['sequences'].values() if x['split']=='validation')!=50:raise ValueError('Unexpected sample counts')


def validate_feature_pipeline(ref):
    canonical=hashlib.sha256(json.dumps(ref['pipeline'],sort_keys=True,separators=(',',':')).encode()).hexdigest()
    if ref['feature_pipeline_sha256']!=canonical:raise ValueError('Reference feature pipeline identity differs')


def ensure_distinct_paths(a,b):
    if a.resolve()==b.resolve() or (a.stat().st_dev,a.stat().st_ino)==(b.stat().st_dev,b.stat().st_ino):
        raise ValueError('Independent replay paths alias')


def validate_replay_identity(receipt,expected,expected_sha,actual_sha,device):
    if expected_sha!=actual_sha or not receipt.get('passed') or receipt.get('device')!=device or any(receipt.get(k)!=v for k,v in expected.items()):
        raise ValueError('Replay receipt identity differs')


def read_frame_source_projection(path):
    """Project only frame identities; archive also contains unused label-derived metrics."""
    raw=Path(path).read_bytes();objects=re.findall(rb'\{[^{}]*\}',raw);result=[]
    for obj in objects:
        fields={}
        for key,pattern in (('index',rb'"index"\s*:\s*(\d+)'),('source_frame_index',rb'"source_frame_index"\s*:\s*(\d+)'),('image_sha256',rb'"image_sha256"\s*:\s*"([0-9a-f]{64})"')):
            hits=re.findall(pattern,obj)
            if len(hits)!=1:raise ValueError('Ambiguous frame identity projection')
            fields[key]=hits[0].decode() if key=='image_sha256' else int(hits[0])
        result.append(fields)
    if not result:raise ValueError('Empty frame identity projection')
    return result


def decoded_sha(path):
    with Image.open(path) as im:
        rgb=im.convert('RGB');h=hashlib.sha256()
        h.update(str(rgb.size).encode());h.update(rgb.tobytes());return h.hexdigest()


def main():
    torch.set_num_threads(2)
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('merged','reference-capture','mps-short-audit','mps-full-audit','short-cpu-receipt','short-mps-a-receipt','short-mps-b-receipt','full-mps-a-receipt','full-mps-b-receipt','checkpoint','existing-preflight','output'):p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();c=json.loads(CONFIG.read_text());verify_config(c)
    out=validate_output(a.output)
    if out.exists():raise FileExistsError(out)
    free=shutil.disk_usage(out.parent).free
    if free<c['min_free_bytes']:raise ValueError('Less than predeclared 22 GiB free; no outputs written')
    done=json.loads((a.merged/'completion.json').read_text())
    if not done.get('passed') or done.get('results_sha256')!=digest(a.merged/'results.json'):raise ValueError('Merged reference changed')
    loc=json.loads((a.merged/'locations.json').read_text());refpath=a.reference_capture/'channel_feature_receipt.json';ref=json.loads(refpath.read_text())
    if not ref.get('passed') or ref.get('sequence')!='seq4' or digest(a.checkpoint)!=ref.get('checkpoint_sha256'):raise ValueError('Reference/checkpoint changed')
    validate_feature_pipeline(ref)
    short=json.loads(a.mps_short_audit.read_text());full=json.loads(a.mps_full_audit.read_text())
    if (not short.get('cpu_matches_archived_captures') or not full.get('passed') or len(full.get('rows',[]))!=39 or not full.get('mps_repeat_selection_equal') or not full.get('mps_repeat_positions_equal') or not full.get('mps_repeat_events_equal') or any(x['mps_repeat_pixel_difference'] for x in full['rows'])):raise ValueError('MPS reference engineering receipt differs')
    videos=load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev');test=load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')
    samples={};source_signature=ref['pipeline']['cpu_identity']['source_signature'];paths_by_group=defaultdict(list)
    for seq,spec in c['sequences'].items():
        source=Path(loc[seq]);comp=json.loads((source/'completion.json').read_text());records=json.loads((source/'records.json').read_text())
        if not comp.get('passed') or comp.get('records_sha256')!=digest(source/'records.json') or len(records)!=1 or records[0]['sequence']!=seq:raise ValueError('Archived record changed')
        rec=records[0];folder=source/seq;promptpath=folder/'prompt.json';framespath=folder/'frames.json';prompt=json.loads(promptpath.read_text());metadata=read_frame_source_projection(framespath);frames=videos[seq]
        if (rec['prompt_sha256']!=digest(promptpath) or rec['frames_sha256']!=digest(framespath) or json.loads((source/'setup.json').read_text())['signature']!=source_signature or len(frames)!=spec['frames'] or len(metadata)!=spec['frames'] or rec['frames']!=spec['frames'] or prompt['first_detection_index']!=spec['prompt_frame'] or rec['first_detection_index']!=spec['prompt_frame'] or prompt['points'] is None or prompt['lean_mask_status']!='automatic_mask' or prompt['refined_mask_sha256']!=digest(folder/'prompt_refined.png')):raise ValueError('Archived prompt, initial mask, images or settings changed')
        hashes=[]
        for position,(frame,row) in enumerate(zip(frames,metadata,strict=True)):
            h=digest(frame.image_path)
            if row['index']!=position or row['source_frame_index']!=frame.frame_index or h!=row['image_sha256']:raise ValueError('Image changed')
            hashes.append(h);paths_by_group[spec['split']].append((seq,frame.frame_index,frame.image_path,h))
        samples[seq]={'split':spec['split'],'source_group_hypothesis':spec['source_group_hypothesis'],'source':str(source),'prompt_frame':spec['prompt_frame'],'frames':spec['frames'],'targets':spec['targets'],'archived_completion_sha256':digest(source/'completion.json'),'archived_records_sha256':digest(source/'records.json'),'archived_frames_sha256':digest(framespath),'archived_prompt_sha256':digest(promptpath),'archived_prompt_mask_sha256':digest(folder/'prompt_refined.png'),'image_sha256':hashes}
    expected_prompt=samples['seq4']['archived_prompt_sha256'];ids=full.get('identities',{})
    expected_identity={'sequence':'seq4','prompt_frame':9,'end_frame':47,'capture_receipt_sha256':digest(refpath),'checkpoint_sha256':digest(a.checkpoint),'prompt_sha256':expected_prompt,'runner_sha256':digest(ROOT/'scripts/verify_lean_window_mps_replay.py'),'cpu_rope_sha256':digest(ROOT/'modeling/lean_indexed_spatial_rope.py'),'torch':ref['pipeline']['torch'],'fork_patches':ref['pipeline']['raw_identity']['patch_sha256'],'mps_rope_sha256':digest(ROOT/'modeling/lean_indexed_spatial_rope_mps.py'),'mps_blocking_sha256':digest(ROOT/'modeling/lean_mps_blocking_pointer_position.py')}
    if any(ids.get(k)!=v for k,v in expected_identity.items()) or full.get('auditor_sha256')!=digest(ROOT/'scripts/audit_lean_window_mps_full.py') or short.get('runner_sha256')!=digest(ROOT/'scripts/audit_lean_window_mps_replay.py'):
        raise ValueError('Historical MPS audit source or runtime identities differ')
    short_paths=(a.short_cpu_receipt,a.short_mps_a_receipt,a.short_mps_b_receipt);short_names=('cpu','mps_a','mps_b')
    full_paths=(a.full_mps_a_receipt,a.full_mps_b_receipt)
    ensure_distinct_paths(short_paths[1],short_paths[2]);ensure_distinct_paths(full_paths[0],full_paths[1])
    for name,path in zip(short_names,short_paths,strict=True):
        receipt=json.loads(path.read_text())
        expected={**expected_identity,'end_frame':20}
        validate_replay_identity(receipt,expected,short['input_receipt_sha256'].get(name),digest(path),'cpu' if name=='cpu' else 'mps')
    for i,path in enumerate(full_paths):
        receipt=json.loads(path.read_text())
        validate_replay_identity(receipt,expected_identity,full['mps_input_receipt_sha256'][i],digest(path),'mps')
    for seq in c['duplicate_screen_sequences']:
        frames=videos.get(seq) or test.get(seq)
        if not frames:raise ValueError('Missing protected sequence')
        for frame in frames:paths_by_group['priority'].append((seq,frame.frame_index,frame.image_path,digest(frame.image_path)))
    byte_seen=defaultdict(set);pixel_seen=defaultdict(set)
    for group,entries in paths_by_group.items():
        for seq,index,path,h in entries:
            byte_seen[h].add(group);pixel_seen[decoded_sha(path)].add(group)
    byte_overlap=sum(len(v)>1 for v in byte_seen.values());pixel_overlap=sum(len(v)>1 for v in pixel_seen.values())
    if byte_overlap or pixel_overlap:raise ValueError(f'Cross-group duplicates: bytes={byte_overlap}, decoded={pixel_overlap}')
    prior,prior_sha=mixed_manifest(a.existing_preflight);reuse=[]
    for item in prior['samples']:
        seq=item['sequence'];frame=item['frame']
        if item['split']!='train' or seq not in samples or frame not in samples[seq]['targets']:
            continue
        if item['image_sha256']!=samples[seq]['image_sha256'][frame]:raise ValueError('Reusable image identity differs')
        early,deep,logits,shape=read_capture(item)
        feature=torch.load(item['feature'],map_location='cpu',weights_only=True);snapshot=torch.load(item['snapshot'],map_location='cpu',weights_only=True)
        raw_dtypes=[str(feature['feature'].dtype),str(snapshot['decoder_inputs']['backbone_features'].dtype),str(snapshot['decoder_outputs'][3].dtype)]
        if raw_dtypes!=['torch.float32']*3:raise ValueError('Historical read dtype differs')
        reuse.append({'sequence':seq,'frame':frame,'capture_device':item['capture_device'],'raw_dtypes':raw_dtypes,
                      'feature_path':item['feature'],'feature_sha256':item['feature_sha256'],
                      'snapshot_path':item['snapshot'],'snapshot_sha256':item['snapshot_sha256'],
                      'historically_label_selected':seq=='seq4' and frame in (17,18,21,24),
                      'eligible_for_new_training':False,
                      'eligibility_requirement':'Per-sequence prompt/trajectory/policy and matching-feature verification; preserve mixed-device identity.'})
    if len(reuse)!=22 or sum(x['capture_device']=='cpu' for x in reuse)!=18 or sum(x['capture_device']=='mps' for x in reuse)!=4:raise ValueError('Historical matching-read inventory differs')
    result={'passed':True,'scope':'CPU-only preflight; no validation/test mask file opened and no label-derived metadata fields parsed; archived frames metadata files contain such fields but only input identities are projected. No model inference, patient independence or efficacy claim.','config_sha256':digest(CONFIG),'manifest_sha256':digest(ROOT/'data/polypgen_sequence.jsonl'),'checkpoint_sha256':digest(a.checkpoint),'merged_locations_sha256':digest(a.merged/'locations.json'),'reference_capture_sha256':digest(refpath),'mps_short_audit_sha256':digest(a.mps_short_audit),'mps_full_audit_sha256':digest(a.mps_full_audit),'short_receipt_paths':[str(x.resolve()) for x in short_paths],'short_receipt_sha256':[digest(x) for x in short_paths],'full_receipt_paths':[str(x.resolve()) for x in full_paths],'full_receipt_sha256':[digest(x) for x in full_paths],'source_sha256':{name:digest(ROOT/name) for name in SOURCES},'preflight_sha256':digest(Path(__file__)),'sequences':samples,'historical_mixed_manifest_sha256':prior_sha,'matching_historical_train_reads':reuse,'free_bytes_at_preflight':free,'duplicates':{'groups':{k:len(v) for k,v in paths_by_group.items()},'cross_group_exact_byte':byte_overlap,'cross_group_decoded_rgb':pixel_overlap,'limitation':'No near-duplicate, source-video, patient or detector-training-exposure proof.'},'next_step':'A separately versioned generalized capture runner and independent GPU parity/no-op tests are required; this manifest cannot be used by the old seq6/13-only runner unchanged.'}
    out.mkdir(parents=True,exist_ok=False);manifest=out/'manifest.json';manifest.write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'manifest_sha256':digest(manifest)},indent=2)+'\n')
    print(json.dumps({'passed':True,'targets':{k:len(v['targets']) for k,v in samples.items()},'duplicates':result['duplicates'],'free_gib':round(free/1024**3,2)},indent=2))
if __name__=='__main__':main()
