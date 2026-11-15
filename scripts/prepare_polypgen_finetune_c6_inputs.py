#!/usr/bin/env python3
"""Freeze label-free C6 image inputs from the verified detector dataset."""
import argparse,json,sys
from pathlib import Path
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output

def read(p):return json.loads(p.read_text())
p=argparse.ArgumentParser()
for k in ('dataset','dataset-verification','training-verification','output'):p.add_argument('--'+k,type=Path,required=True)
a=p.parse_args();d=read(a.dataset/'dataset.json');v=read(a.dataset_verification/'verification.json');t=read(a.training_verification/'verification.json')
if not v['passed'] or v['dataset_sha256']!=digest(a.dataset/'dataset.json') or v['config_sha256']!=digest(a.dataset/'data.yaml') or v['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_detector_pilot_dataset.py') or not t['passed'] or t['dataset_sha256']!=v['dataset_sha256'] or t['verifier_sha256']!=digest(ROOT/'scripts/verify_polypgen_original_detector_finetune.py'):raise ValueError('Dataset or trained model changed')
rows=sorted([x for x in d['included'] if x['split']=='val'],key=lambda x:x['staged_image'])
if len(rows)!=88 or any(x['center']!=6 for x in rows):raise ValueError('C6 validation scope changed')
inputs=[]
for i,row in enumerate(rows):
 path=Path(row['staged_image'])
 if path.is_symlink() or digest(path)!=row['image_sha256']:raise ValueError('C6 image changed')
 with Image.open(path) as im:w,h=im.size
 if (w,h)!=(row['image_width'],row['image_height']):raise ValueError('Image geometry changed')
 inputs.append({'id':i,'image':str(path),'image_sha256':row['image_sha256'],'width':w,'height':h})
out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
r={'passed':True,'scope':'All fixed C6 validation images; image-only, no target label values.','inputs':inputs,'dataset_sha256':digest(a.dataset/'dataset.json'),'training_verification_sha256':digest(a.training_verification/'verification.json'),'protocol_sha256':digest(ROOT/'protocols/polypgen_original_detector_finetune_v1.json'),'runner_sha256':digest(Path(__file__))}
(out/'inputs.json').write_text(json.dumps(r,indent=2)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'inputs_sha256':digest(out/'inputs.json')},indent=2)+'\n');print('Frozen 88 C6 image-only inputs.')
