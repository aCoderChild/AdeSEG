#!/usr/bin/env python3
"""Independently compare every reused Ultralytics label-cache entry with staged labels."""
import argparse,json,sys
from datetime import datetime, timezone
from pathlib import Path
import numpy as np
from PIL import Image
from ultralytics.data.utils import load_dataset_cache_file
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
p=argparse.ArgumentParser()
for x in ('dataset','dataset-verification','training','output'):p.add_argument('--'+x,type=Path,required=True)
a=p.parse_args();data=json.loads((a.dataset/'dataset.json').read_text());v=json.loads((a.dataset_verification/'verification.json').read_text());setup=json.loads((a.training/'setup.json').read_text())
if not v['passed'] or v['dataset_sha256']!=digest(a.dataset/'dataset.json') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_pilot_dataset.py') or setup['dataset_sha256']!=v['dataset_sha256'] or setup['training_args']['cache'] is not False:raise ValueError('Dataset or training identity changed')
start=datetime.fromisoformat(setup['started_utc']).timestamp();records={};counts={}
for split in ('train','val'):
 cache=a.dataset/'labels'/f'{split}.cache';mtime=cache.stat().st_mtime;before=digest(cache)
 if mtime>start:raise ValueError('Label cache changed after training started')
 actual=load_dataset_cache_file(cache);entries=actual['labels'];expected={str(Path(r['staged_image']).resolve()):r for r in data['included'] if r['split']==split}
 if len(entries)!=len(expected) or len(expected)!=(1377 if split=='train' else 88):raise ValueError('Cache image count differs')
 seen=set();boxes=0
 for e in entries:
  image=Path(e['im_file']).resolve();key=str(image)
  if key not in expected or key in seen:raise ValueError('Cache image identity differs')
  seen.add(key);row=expected[key];label=Path(row['staged_label'])
  if digest(image)!=row['image_sha256'] or digest(label)!=row['label_sha256']:raise ValueError('Staged image or label changed')
  text=label.read_text().splitlines();values=np.array([[float(x) for x in line.split()] for line in text if line.strip()],dtype=np.float32).reshape(-1,5)
  cached=np.concatenate((e['cls'],e['bboxes']),axis=1)
  if values.shape!=cached.shape or not np.array_equal(values,cached) or not e['normalized'] or e['bbox_format']!='xywh' or len(e['segments']) or e['keypoints'] is not None:raise ValueError('Cached annotations differ from staged labels')
  with Image.open(image) as im:shape=(im.height,im.width)
  if tuple(e['shape'])!=shape:raise ValueError('Cached image shape differs')
  boxes+=len(values)
 if seen!=set(expected) or digest(cache)!=before:raise ValueError('Cache coverage or bytes changed')
 records[split]={'cache_sha256':before,'last_modified_utc':datetime.fromtimestamp(mtime,timezone.utc).isoformat(),'training_started_utc':setup['started_utc'],'entries':len(entries),'boxes':boxes};counts[split]=len(entries)
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
(out/'verification.json').write_text(json.dumps({'passed':True,'dataset_sha256':v['dataset_sha256'],'training_setup_sha256':digest(a.training/'setup.json'),'splits':records,'verifier_sha256':digest(Path(__file__)),'scope':'Image cache setting was false; cached annotation arrays were independently checked against every staged label. Cache modification times precede the training start.'},indent=2)+'\n')
print('All cached annotation entries agree exactly with verified staged labels, and cache modification times precede the run.')
