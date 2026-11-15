#!/usr/bin/env python3
"""Freeze label-free seq6/seq13 development capture inputs and identities."""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output

TARGETS={'seq6':[24,31,39,46,53,61,68,75,83,90],
         'seq13':[8,35,62,88,115,142,169,195,222,249]}


def read(path):return json.loads(Path(path).read_text())


def main():
    p=argparse.ArgumentParser(description=__doc__)
    for k in ('merged','reference-capture','mps-short-audit','mps-full-audit','checkpoint','output'):
        p.add_argument('--'+k,type=Path,required=True)
    a=p.parse_args();merged_done=read(a.merged/'completion.json')
    if not merged_done['passed'] or merged_done['results_sha256']!=digest(a.merged/'results.json'):
        raise ValueError('Merged automatic reference changed')
    locations=read(a.merged/'locations.json');ref=read(a.reference_capture/'channel_feature_receipt.json')
    if not ref['passed'] or ref['sequence']!='seq4' or digest(a.checkpoint)!=ref['checkpoint_sha256']:
        raise ValueError('CPU training capture or checkpoint differs')
    if ref['feature_pipeline_sha256']!=digest_pipeline(ref['pipeline']):
        raise ValueError('Reference feature-pipeline hash differs')
    short=read(a.mps_short_audit);full=read(a.mps_full_audit)
    if not short['cpu_matches_archived_captures'] or not full['passed'] or len(full['rows'])!=39:
        raise ValueError('Previous seq4 CPU/MPS engineering audits incomplete')
    if (not full['mps_repeat_selection_equal'] or not full['mps_repeat_positions_equal'] or not full['mps_repeat_events_equal']
            or any(x['mps_repeat_pixel_difference'] for x in full['rows'])):
        raise ValueError('Prior full-sequence MPS repeats differ')
    expected=ref['pipeline']['cpu_identity']['source_signature']
    videos=load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev')
    samples={}
    for seq,targets in TARGETS.items():
        source=Path(locations[seq]);done=read(source/'completion.json');records=read(source/'records.json')
        if not done['passed'] or done['records_sha256']!=digest(source/'records.json') or len(records)!=1 or records[0]['sequence']!=seq:
            raise ValueError('Archived automatic run changed')
        rec=records[0];folder=source/seq;prompt_file=folder/'prompt.json';frame_file=folder/'frames.json'
        if rec['prompt_sha256']!=digest(prompt_file) or rec['frames_sha256']!=digest(frame_file):raise ValueError('Archived prompt or frame metadata changed')
        if read(source/'setup.json')['signature']!=expected:raise ValueError('Archived automatic prompt configuration differs')
        prompt=read(prompt_file);start=prompt['first_detection_index'];frames=videos[seq];metadata=read(frame_file)
        if (seq not in TARGETS or len(frames)!=len(metadata) or len(frames)!=rec['frames']
                or start!=rec['first_detection_index'] or prompt['points'] is None or prompt['lean_mask_status']!='automatic_mask'
                or len(targets)!=10 or min(targets)<=start or max(targets)!=len(frames)-1):
            raise ValueError('Prompt or prespecified target coverage differs')
        for frame,row in zip(frames,metadata,strict=True):
            if digest(frame.image_path)!=row['image_sha256']:raise ValueError('Source image differs')
        samples[seq]={'source':str(source),'prompt_frame':start,'frames':len(frames),'targets':targets,
                      'source_records_sha256':digest(source/'records.json'),'source_frames_sha256':digest(frame_file),
                      'prompt_sha256':digest(prompt_file),'image_sha256':[x['image_sha256'] for x in metadata]}
    result={'passed':True,'scope':'Label-free development capture plan; patient independence unknown.',
            'sequences':samples,'checkpoint_sha256':digest(a.checkpoint),'cpu_feature_pipeline_sha256':ref['feature_pipeline_sha256'],
            'reference_capture_sha256':digest(a.reference_capture/'channel_feature_receipt.json'),
            'mps_short_audit_sha256':digest(a.mps_short_audit),'mps_full_audit_sha256':digest(a.mps_full_audit),
            'protocol_sha256':digest(ROOT/'protocols/EXPANDED_TRAIN_CAPTURE_SEQ6_SEQ13_V1.md'),
            'data_manifest_sha256':digest(ROOT/'data/polypgen_sequence.jsonl'),'runner_sha256':digest(Path(__file__))}
    out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
    (out/'manifest.json').write_text(json.dumps(result,indent=2)+'\n')
    (out/'completion.json').write_text(json.dumps({'passed':True,'manifest_sha256':digest(out/'manifest.json')},indent=2)+'\n')
    print(json.dumps({'passed':True,'targets':{k:v['targets'] for k,v in samples.items()}},indent=2))


def digest_pipeline(pipeline):
    import hashlib
    return hashlib.sha256(json.dumps(pipeline,sort_keys=True,separators=(',',':')).encode()).hexdigest()


if __name__=='__main__':main()
