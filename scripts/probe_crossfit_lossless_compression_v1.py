#!/usr/bin/env python3
"""Losslessly compress/decompress six existing capture files in memory only.

This CPU-only resource probe opens no labels, creates no compressed artifact,
and says nothing about actual future-capture compressibility or MPS behavior.
"""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import time
import zlib

ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
from scripts.source_group_label_guard_v1 import install_mask_open_guard
from scripts.run_boundary_video_factorial_v1 import BASE,OUT

MANIFEST=OUT/'source_group_boundary_factorial_v1_bvqxd475/manifest/manifest.json'
SEQUENCES=('seq2','seq4','seq13')


def main():
    parser=argparse.ArgumentParser(description=__doc__);parser.add_argument('--output',type=Path,required=True);a=parser.parse_args()
    out=a.output.resolve()
    if not out.is_relative_to(BASE.resolve()) or out.exists():raise ValueError('Output must be a new Drive experiment folder')
    guard=install_mask_open_guard(ROOT/'data/polypgen_sequence.jsonl');manifest_sha=digest(MANIFEST)
    data=json.loads(MANIFEST.read_text())
    if digest(MANIFEST)!=manifest_sha:raise ValueError('Manifest changed while reading')
    rows=[]
    for seq in SEQUENCES:
        item=next(x for x in data['samples'] if x['sequence']==seq and x['split']=='train')
        for kind in ('feature','snapshot'):
            path=Path(item[kind]).resolve(strict=True)
            if not path.is_relative_to(BASE.resolve()):raise ValueError('Capture outside experiment folder')
            compressor=zlib.compressobj(1);decompressor=zlib.decompressobj();original=hashlib.sha256();recovered=hashlib.sha256()
            size=compressed=0;start=time.monotonic()
            with path.open('rb') as f:
                while chunk:=f.read(1024*1024):
                    size+=len(chunk);original.update(chunk);encoded=compressor.compress(chunk)
                    compressed+=len(encoded);recovered.update(decompressor.decompress(encoded))
            encoded=compressor.flush();compressed+=len(encoded);recovered.update(decompressor.decompress(encoded));recovered.update(decompressor.flush())
            if (original.hexdigest()!=item[kind+'_sha256'] or recovered.hexdigest()!=original.hexdigest()
                    or digest(path)!=original.hexdigest() or not decompressor.eof):raise ValueError('Capture or lossless roundtrip changed')
            rows.append({'sequence':seq,'frame':item['frame'],'kind':kind,'path':str(path),'sha256':original.hexdigest(),
                         'source_bytes':size,'compressed_bytes':compressed,'ratio':compressed/size,
                         'seconds':time.monotonic()-start,'roundtrip_exact':True})
    if guard['open_attempts'] or digest(MANIFEST)!=manifest_sha:raise ValueError('Label access or manifest change')
    result={'passed':True,'scope':'Six existing training-read files; streaming in-memory zlib level 1. No new feature captures.',
            'limitations':'Not a random sample, storage forecast, cloud quota measurement, or validated capture format.',
            'manifest_sha256':manifest_sha,'source_sha256':digest(Path(__file__)),'label_guard':dict(guard),
            'rows':rows,'aggregate_compressed_fraction':sum(x['compressed_bytes'] for x in rows)/sum(x['source_bytes'] for x in rows)}
    out.mkdir(parents=True,exist_ok=False);path=out/'probe.json';path.write_text(json.dumps(result,indent=2)+'\n')
    print(json.dumps({'passed':True,'output':str(out),'aggregate_compressed_fraction':result['aggregate_compressed_fraction']}))


if __name__=='__main__':main()
