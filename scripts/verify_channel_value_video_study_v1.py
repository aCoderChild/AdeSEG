#!/usr/bin/env python3
"""Verify historical fixed-read provenance once for an asinh video batch.

This CPU-only certificate covers archived training inputs at certificate
creation; video children do not rehash inactive training tensors. It opens no
target labels and does not run model inference or assert efficacy.
"""
import argparse
import json
from pathlib import Path
import sys

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.memory_token_io import digest
from scripts.channel_value_fixed_read_v1 import deps,require_environment_flags
from modeling.lean_channel_value_window_runtime_v1 import (BASE,HASHES,STUDY,
    expected_lean_fingerprints,runtime_sources,verify_study)


def make_certificate(output):
    require_environment_flags()
    out=Path(output).resolve()
    if not out.is_relative_to(BASE.resolve(strict=True)) or out.exists():
        raise ValueError('Certificate output must be new and under Drive experiment root')
    guard=install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl')
    manifest,sha,plan,payload=verify_study()
    _,fingerprint=expected_lean_fingerprints()
    if (sha!=HASHES['manifest'] or payload.get('mode')!='foreground_background_asinh'
            or guard['open_attempts']!=0):raise ValueError('Study or label guard differs')
    proof={'passed':True,'scope':'Full historical capture provenance checked once at certificate creation; '
           'children bind the snapshot and checkpoint but do not rehash inactive historical tensors.',
           'study':str(STUDY),'study_hashes':HASHES,'runtime_sources':runtime_sources(),
           'training_dependencies':deps(),'legacy_model_fingerprint_sha256':fingerprint,
           'training_environment':plan['environment'],'feature_pipeline_sha256':manifest['feature_pipeline_sha256'],
           'label_guard':dict(guard),'verifier_sha256':digest(Path(__file__))}
    out.mkdir(parents=True,exist_ok=False);file=out/'certificate.json'
    file.write_text(json.dumps(proof,indent=2)+'\n')
    return file


def main():
    p=argparse.ArgumentParser(description=__doc__);p.add_argument('--output',type=Path,required=True);a=p.parse_args()
    file=make_certificate(a.output);print(json.dumps({'passed':True,'certificate':str(file),'sha256':digest(file)}))


if __name__=='__main__':main()
