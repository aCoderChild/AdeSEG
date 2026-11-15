#!/usr/bin/env python3
"""Render retrospective seq23 baseline and token-window mask comparisons."""
import argparse,json,os,sys
from pathlib import Path
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest,validate_output
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('baseline','candidate','audit','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();done=read(a.audit/'completion.json');aud=read(a.audit/'results.json')
 if not done['passed'] or digest(a.audit/'results.json')!=done['results_sha256'] or done['source_sha256']!=digest(ROOT/'scripts/audit_lean_token_window.py'):raise ValueError('Audit invalid')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);os.environ['MPLCONFIGDIR']=str(out/'matplotlib_cache')
 import cv2
 import matplotlib
 matplotlib.use('Agg')
 import matplotlib.pyplot as plt
 import numpy as np
 from PIL import Image
 from datasets.video import load_manifest
 from datasets.io import load_label_mask
 frames={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')}['seq23'];baseline=read(a.baseline/'frames.json');candidate=read(a.candidate/'frames.json');indices=[8,35,47,54]
 fig,axes=plt.subplots(len(indices),3,figsize=(13,11),constrained_layout=True);records=[]
 for ax,index in zip(axes,indices):
  frame=frames[index];data=aud['sequences']['seq23']['per_frame'][index]
  if digest(frame.image_path)!=baseline[index]['image_sha256'] or digest(frame.mask_path)!=baseline[index]['label_sha256']:raise ValueError('Input changed')
  image=np.asarray(Image.open(frame.image_path).convert('RGB'));truth=load_label_mask(frame.mask_path,[1])==1;ax[0].imshow(image);ax[0].set_title(f'Frame {index}: image')
  item={'frame':index,'image_sha256':digest(frame.image_path),'label_sha256':digest(frame.mask_path)}
  for col,(name,folder,metadata,scorekey) in enumerate([('Baseline',a.baseline,baseline,'baseline'),('Token window',a.candidate,candidate,'candidate')],start=1):
   path=folder/f'{index:06d}.png'
   if digest(path)!=metadata[index]['mask_sha256']:raise ValueError('Prediction changed')
   mask=np.asarray(Image.open(path))>0;overlay=image.copy()
   cv2.drawContours(overlay,cv2.findContours(truth.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(255,225,0),4)
   cv2.drawContours(overlay,cv2.findContours(mask.astype('uint8'),cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)[0],-1,(0,210,255),3)
   ax[col].imshow(overlay);ax[col].set_title(f"{name}: Dice {data['scores'][scorekey]['dice']:.3f}");item[name]=digest(path)
  for axis in ax:axis.axis('off')
  records.append(item)
 fig.suptitle('Sequence 23: source shown; inset excluded during inference\nLabel in yellow, prediction in cyan (selected examples)',fontsize=14)
 fig.savefig(out/'comparison.png',dpi=130);plt.close(fig)
 receipt={'scope':'Frames selected after observing the results; illustrative only.','frames':records,'audit_sha256':done['results_sha256'],'figure_sha256':digest(out/'comparison.png'),'source_sha256':digest(Path(__file__))}
 (out/'records.json').write_text(json.dumps(receipt,indent=2)+'\n');print('Saved illustrative comparison')
if __name__=='__main__':main()
