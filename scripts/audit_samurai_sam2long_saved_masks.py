#!/usr/bin/env python3
"""Independently recompute Dice and absence from archived author masks."""
import json,sys
from pathlib import Path
import numpy as np
from PIL import Image
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from scripts.memory_token_io import digest,validate_output
base=Path('/Users/maianhpham/Library/CloudStorage/GoogleDrive-phammaianh11102005@gmail.com/My Drive/AdeSEG/outputs/experiment')
out=validate_output(base/'paper_comparator_audit_v1/saved_masks_attempt01');out.mkdir(parents=True,exist_ok=False)
methods={'samurai':base/'12_author_region_attention_polypgen_cpu_v1/samurai_full','sam2long':base/'12_author_region_attention_polypgen_cpu_v1/sam2long_full'};results=[];issues=[]
for name,root in methods.items():
 for seq in [f'seq{i}' for i in range(2,24) if i!=7]:
  folder=root/seq;inputs=json.loads((folder/'inputs.json').read_text());byid={int(x['frame']):x for x in inputs};records=[json.loads(x) for x in (folder/'frames.jsonl').read_text().splitlines()];pcount=ecount=fa=0;total=0.;maxerror=0.
  if len(byid)!=len(inputs):raise ValueError('Duplicate source IDs')
  for row in records:
   i=int(row['source_frame_index']);item=byid[i];label=Path(item['mask']);image=Path(item['image']);mask=folder/f"frame_{row['frame']:06d}.png"
   if digest(label)!=item['mask_sha256'] or digest(image)!=item['image_sha256']:raise ValueError(f'Input hash changed: {name}/{seq}/{i}')
   gt=load_label_mask(label,[1])==1;pred=np.asarray(Image.open(mask))>0
   if gt.shape!=pred.shape:raise ValueError('Mask geometry changed')
   tp=int(np.count_nonzero(pred & gt));fp=int(np.count_nonzero(pred & ~gt));fn=int(np.count_nonzero(~pred & gt));den=2*tp+fp+fn;dice=2*tp/den if den else 1.;err=abs(dice-row['scores']['dice']);maxerror=max(maxerror,err)
   if err>1e-12 or bool(gt.any())!=row['present'] or (bool(pred.any()) if not gt.any() else None)!=row['absent_fp']:issues.append({'method':name,'sequence':seq,'source_frame':i,'dice_error':err,'presence':bool(gt.any()),'recorded_presence':row['present']})
   if gt.any():pcount+=1;total+=dice
   else:ecount+=1;fa+=int(pred.any())
  results.append({'method':name,'sequence':seq,'present':pcount,'empty':ecount,'false_alarms':fa,'mean_present_dice':total/pcount if pcount else None,'max_dice_error':maxerror,'frames_sha256':digest(folder/'frames.jsonl'),'inputs_sha256':digest(folder/'inputs.json')})
  print(name,seq,'checked',len(records),flush=True)
data={'passed':not issues,'results':results,'issues':issues,'runner_sha256':digest(Path(__file__)),'scope':'Saved binary mask rescore; no claim about native model forward equivalence or prospective generalization.'};(out/'results.json').write_text(json.dumps(data,indent=2)+'\n');(out/'completion.json').write_text(json.dumps({'passed':not issues,'results_sha256':digest(out/'results.json')},indent=2)+'\n')
if issues:raise RuntimeError(f'{len(issues)} discrepancies')
