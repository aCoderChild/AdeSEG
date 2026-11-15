#!/usr/bin/env python3
"""Audit exact image-byte overlap across exploratory training and all test videos.

This never opens labels. It cannot detect near duplicates, patient overlap or
dataset/checkpoint training exposure.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from scripts.memory_token_io import digest,validate_output


def main():
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--mixed-preflight',required=True,type=Path)
    parser.add_argument('--output',required=True,type=Path)
    args=parser.parse_args()
    manifest_path=args.mixed_preflight/'manifest.json'
    manifest=json.loads(manifest_path.read_text());completion=json.loads((args.mixed_preflight/'completion.json').read_text())
    if not manifest['passed'] or not completion['passed'] or completion['manifest_sha256']!=digest(manifest_path):
        raise ValueError('Mixed manifest changed')
    train_ids={x['sequence'] for x in manifest['samples'] if x['split']=='train'}
    if train_ids!={'seq2','seq3','seq4','seq6','seq13'}:raise ValueError('Unexpected training cohort')
    dev=load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev');test=load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')
    train_hashes={seq:[digest(frame.image_path) for frame in dev[seq]] for seq in sorted(train_ids)}
    test_hashes={seq:[digest(frame.image_path) for frame in frames] for seq,frames in sorted(test.items())}
    collisions=[]
    for train_seq,source in train_hashes.items():
        source_set=set(source)
        for test_seq,target in test_hashes.items():
            for value in sorted(source_set.intersection(target)):
                collisions.append({'train_sequence':train_seq,'test_sequence':test_seq,'image_sha256':value})
    result={'passed':not collisions,'scope':'Exact source-image file bytes only; no labels read, not near-duplicate or patient identity proof.',
            'mixed_manifest_sha256':digest(manifest_path),'data_manifest_sha256':digest(ROOT/'data/polypgen_sequence.jsonl'),
            'runner_sha256':digest(Path(__file__)),'train_sequence_frame_counts':{s:len(x) for s,x in train_hashes.items()},
            'test_sequence_frame_counts':{s:len(x) for s,x in test_hashes.items()},'exact_cross_split_collisions':collisions}
    out=validate_output(args.output)
    if out.exists():raise FileExistsError(out)
    out.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':result['passed'],'train_frames':sum(map(len,train_hashes.values())),
                      'test_frames':sum(map(len,test_hashes.values())),'collisions':len(collisions)}))
    if collisions:raise ValueError('Exact train/test image overlap found')


if __name__=='__main__':main()
