#!/usr/bin/env python3
"""Reconstruct label-free source feature scores, checked against captured scores."""
import argparse,json,sys,hashlib
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest
from modeling.seq23_viewport import crop_viewport,VIEWPORT
from modeling.prompt_feature_refinement import reference_prototypes
from modeling.lean_early_contrast_anchor import blend_anchor_with_early_contrast
from modeling.lean_channel_attention import PromptContrastChannelAttention
from modeling.lean_channel_feature_capture import sha
from modeling.lean_channel_training import parameter_hash
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import build_author_predictor,parameter_fingerprints,assert_parameter_integrity
from scripts.run_lean_token_window import FORK,PATCHED

def read(x):return json.loads(x.read_text())
def main():
 p=argparse.ArgumentParser()
 for k in ('checkpoint','capture','merged','pool-verification','dataset','dataset-audit','training','training-audit','preflight','output'):p.add_argument('--'+k,type=Path,required=True)
 a=p.parse_args();cap=read(a.capture/'priority_capture_receipt.json');pool=read(a.pool_verification/'verification.json');data=read(a.dataset/'dataset.json');da=read(a.dataset_audit/'verification.json');tr=read(a.training/'training_receipt.json');ta=read(a.training_audit/'verification.json');pre=read(a.preflight/'manifest.json')
 if not cap['passed'] or cap['sequence'] not in ('seq11','seq23') or cap['viewport']!=(cap['sequence']=='seq23'):raise ValueError('Capture scope')
 for k,path in (('capture_sha256',ROOT/'modeling/lean_region_read_capture.py'),('selector_sha256',ROOT/'modeling/lean_region_read_intervention.py'),('priority_runner_sha256',ROOT/'scripts/capture_lean_region_priority_reads.py')):
  if cap[k]!=digest(path):raise ValueError('Capture implementation changed')
 if not pool['passed'] or pool['verifier_sha256']!=digest(ROOT/'scripts/verify_lean_region_token_pool_gates.py') or digest(a.checkpoint)!=pool['identity']['checkpoint_sha256'] or any(digest(FORK/x)!=pool['identity']['patch_sha256'][x] for x in PATCHED):raise ValueError('Model gate changed')
 if not da['passed'] or da['dataset_sha256']!=digest(a.dataset/'dataset.json') or da['verifier_sha256']!=digest(ROOT/'scripts/audit_lean_channel_training_dataset.py') or not ta['passed'] or ta['verifier_sha256']!=digest(ROOT/'scripts/audit_lean_channel_training_repeat.py') or ta['first_receipt_sha256']!=digest(a.training/'training_receipt.json') or ta['dataset_sha256']!=digest(a.dataset/'dataset.json') or not tr['passed'] or data['preflight_sha256']!=digest(a.preflight/'manifest.json'):raise ValueError('Training provenance changed')
 if data['backbone_sha256']!=digest(a.checkpoint) or cap['sequence'] not in data['groups']['protected_priority']:raise ValueError('Backbone or held-out sequence mismatch')
 samples=[x for x in data['samples'] if x['sequence']=='seq9' and x['group']=='validation']
 if not samples:raise ValueError('No fixed foreign descriptor')
 foreign_path=Path(samples[0]['feature_path'])
 if digest(foreign_path)!=samples[0]['feature_file_sha256']:raise ValueError('Foreign descriptor changed')
 foreign=torch.load(foreign_path,map_location='cpu',weights_only=True);f_fg,f_bg=foreign['foreground'],foreign['background']
 videos={**load_manifest(ROOT/'data/polypgen_sequence.jsonl','dev'),**load_manifest(ROOT/'data/polypgen_sequence.jsonl','test')};seq=cap['sequence'];frames=videos[seq]
 if cap['viewport']:prompt_path=a.capture/'prompt.png';prompt=read(a.capture/'prompt.json')
 else:
  folder=Path(read(a.merged/'locations.json')[seq])/seq;prompt_path=folder/'prompt_refined.png';prompt=read(folder/'prompt.json')
  if digest(prompt_path)!=prompt['refined_mask_sha256']:raise ValueError('Prompt changed')
 if cap['viewport'] and digest(prompt_path)!=digest(Path(cap['reference'])/'prompt.png'):raise ValueError('Cropped prompt changed')
 mask=np.asarray(Image.open(prompt_path))>0
 if cap['viewport']:x0,y0,x1,y1=VIEWPORT;mask=mask[y0:y1,x0:x1]
 first=prompt['first_detection_index'];required=set();sources={}
 for item in cap['snapshots']:
  path=a.capture/'source_snapshots'/f"{item['frame']:06d}.pt"
  if digest(path)!=item['sources_sha256']:raise ValueError('Source snapshot changed')
  snapshot=torch.load(path,map_location='cpu',weights_only=True)
  for source in snapshot['sources']:
   i=source['frame'];required.add(i);sources.setdefault(i,[]).append(source)
 images={}
 for i in required|{first}:
  path=frames[i].image_path;images[str(i)]=digest(path)
  # The archived frame metadata independently fixes the original input bytes.
  meta=pre['sequences'][seq];rows=read(Path(meta['source'])/seq/'frames.json')
  if digest(Path(meta['source'])/seq/'frames.json')!=meta['frames_metadata_sha256'] or images[str(i)]!=rows[i]['image_sha256']:raise ValueError('Input image changed')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False);torch.set_num_threads(2);sys.path.insert(0,str(FORK))
 model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),FORK);before=parameter_fingerprints(model)
 import sam2
 if not Path(sam2.__file__).resolve().is_relative_to(FORK.resolve()):raise ValueError('Unexpected SAM2 package')
 gates={}
 for arm,info in tr['arms'].items():
  path=a.training/f'{arm}.pt'
  if digest(path)!=info['checkpoint_sha256']:raise ValueError('Gate checkpoint changed')
  gates[arm]=PromptContrastChannelAttention(checkpoint=path,expected_backbone_sha256=data['backbone_sha256'],expected_feature_pipeline_sha256=data['feature_pipeline_sha256']).eval().requires_grad_(False)
 params={arm:parameter_hash(gate) for arm,gate in gates.items()}
 mean=torch.tensor([.485,.456,.406])[:,None,None];std=torch.tensor([.229,.224,.225])[:,None,None]
 def encode(i):
  with Image.open(frames[i].image_path) as im:pixels=np.array(im.convert('RGB'),copy=True)
  if cap['viewport']:pixels=crop_viewport(pixels)
  # Reproduce _load_img_as_tensor followed by float32 video-buffer assignment.
  resized=np.array(Image.fromarray(pixels).resize((1024,1024))) / 255.0
  image=torch.from_numpy(resized).permute(2,0,1).float();image-=mean;image/=std
  return model.forward_image(image[None])['backbone_fpn'][0][0].detach().float()
 events=[];maxdiff=0.
 with torch.inference_mode():
  initial=encode(first);fg,bg=reference_prototypes(initial,mask)
  for i in sorted(required):
   feature=encode(i);baseline=blend_anchor_with_early_contrast(torch.zeros(4096),feature,fg,bg,(64,64),1.)
   for source in sources[i]:
    diff=float((baseline[source['token_ids']]-source['contrast']).abs().max());maxdiff=max(maxdiff,diff)
    if diff>1e-6:raise ValueError(f'Reconstructed captured source contrast differs on {seq}:{i}: {diff}')
   scores={}
   for arm,gate in gates.items():
    if arm=='prompt_conditioned':gfg,gbg=fg,bg
    elif arm=='descriptor_independent':gfg,gbg=torch.zeros_like(fg),torch.zeros_like(bg)
    elif arm=='mismatched_prompt':gfg,gbg=f_fg,f_bg
    else:raise ValueError('Unexpected trained arm')
    score,weights=gate(feature,fg,bg,gate_foreground=gfg,gate_background=gbg);scores[arm]=score.cpu()
   path=out/f'{i:06d}.pt';torch.save(scores,path);events.append({'source_frame':i,'file_sha256':digest(path),'score_sha256':{k:sha(v) for k,v in scores.items()}})
 if params!={arm:parameter_hash(gate) for arm,gate in gates.items()}:raise ValueError('Gate parameters changed')
 assert_parameter_integrity(model,before,out,'after')
 receipt={'passed':True,'sequence':seq,'events':events,'max_unweighted_error':maxdiff,'input_sha256':images,'prompt_sha256':digest(prompt_path),'foreign_feature_sha256':digest(foreign_path),'capture_sha256':digest(a.capture/'priority_capture_receipt.json'),'training_receipt_sha256':digest(a.training/'training_receipt.json'),'training_audit_sha256':digest(a.training_audit/'verification.json'),'dataset_sha256':digest(a.dataset/'dataset.json'),'protocol_sha256':digest(ROOT/'protocols/lean_channel_fixed_read_v1.json'),'runner_sha256':digest(Path(__file__))}
 (out/'completion.json').write_text(json.dumps(receipt,indent=2)+'\n');print(json.dumps({'passed':True,'sequence':seq,'source_frames':len(events),'max_unweighted_error':maxdiff}))
if __name__=='__main__':main()
