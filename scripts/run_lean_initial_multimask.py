#!/usr/bin/env python3
"""Compare initial prompts with Lean's native single- versus multi-mask heads."""
import argparse
import hashlib
import json
from pathlib import Path
import sys
import tempfile
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch

ROOT=Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))
from datasets.video import load_manifest,stage_frames
from modeling.early_feature_prompt import early_feature_points
from modeling.lean_initial_multimask import initial_multimask
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import source_identity,build_author_predictor,parameter_fingerprints,assert_parameter_integrity

PROTOCOL=ROOT/'protocols/lean_initial_multimask_v1.json'; MANIFEST=ROOT/'data/polypgen_sequence.jsonl'
def read(path): return json.loads(path.read_text())

def main():
 p=argparse.ArgumentParser(description=__doc__)
 for name in ('checkpoint','merged','indexed-verification','output'): p.add_argument('--'+name,type=Path,required=True)
 p.add_argument('--mode',choices=('gate','cohort'),required=True); p.add_argument('--sequence',action='append',required=True)
 p.add_argument('--gate-a',type=Path); p.add_argument('--gate-b',type=Path)
 a=p.parse_args(); protocol=read(PROTOCOL)
 if len(set(a.sequence))!=len(a.sequence) or not set(a.sequence)<=set(protocol['cohort']) or (a.mode=='gate' and a.sequence!=['seq3']): raise ValueError('Invalid cohort')
 merged=read(a.merged/'completion.json'); verified=read(a.indexed_verification)
 if not merged['passed'] or not verified['passed'] or digest(a.merged/'results.json')!=merged['results_sha256']:
  raise ValueError('Reference incomplete')
 sig=verified['identity']['cpu_identity']['source_signature']
 pinned={'early_module_sha256':ROOT/'modeling/early_feature_prompt.py','prompt_module_sha256':ROOT/'modeling/prompt_feature_refinement.py',
  'memory_module_sha256':ROOT/'modeling/lean_multiscale_anchor.py','lean_runner_sha256':ROOT/'scripts/run_author_baseline.py',
  'runner_sha256':ROOT/'scripts/run_lean_auto_multiscale.py','manifest_sha256':MANIFEST,
  'bypass_sha256':ROOT/'external/Lean-SAM2/checkpoints/bypass_bottleneck_base.pth','lean_config_sha256':ROOT/'external/Lean-SAM2/configs/lean_sam2/lean_sam2.yaml'}
 if (digest(a.checkpoint)!=sig['checkpoint_sha256'] or any(digest(path)!=sig[key] for key,path in pinned.items())
     or torch.__version__!=sig['torch'] or np.__version__!=sig['numpy'] or sys.version!=sig['python']): raise ValueError('Model or runtime differs')
 identity={'runner_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL), 'module_sha256':digest(ROOT/'modeling/lean_initial_multimask.py'),
  'checkpoint_sha256':digest(a.checkpoint),'source_signature':sig,'indexed_verification_sha256':digest(a.indexed_verification),
  'merged_results_sha256':merged['results_sha256']}
 if a.mode=='cohort':
  if not a.gate_a or not a.gate_b: raise ValueError('Two engineering runs required')
  gates=[]
  for gate in (a.gate_a,a.gate_b):
   setup,done=read(gate/'setup.json'),read(gate/'completion.json')
   if setup['mode']!='gate' or setup['identity']!=identity or not done['passed'] or digest(gate/'records.json')!=done['records_sha256']:
    raise ValueError('Invalid engineering run')
   gates.append(read(gate/'records.json'))
  if gates[0]!=gates[1]: raise ValueError('Engineering replays differ')
 videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')}; locations=read(a.merged/'locations.json')
 out=validate_output(a.output); out.mkdir(parents=True,exist_ok=False)
 (out/'setup.json').write_text(json.dumps({'mode':a.mode,'sequences':a.sequence,'identity':identity},indent=2)+'\n')
 torch.set_num_threads(2); repo,revision=source_identity('lean')
 if revision!=sig['revision']: raise ValueError('Author source differs')
 results=[]
 for name in a.sequence:
  source=Path(locations[name]); folder=source/name; olddone=read(source/'completion.json'); oldrecord=read(source/'records.json')[0]; audit=read(source/'independent_audit.json')
  if (not olddone['passed'] or not audit['passed'] or digest(source/'records.json')!=olddone['records_sha256']
      or audit['records_sha256']!=olddone['records_sha256'] or digest(folder/'prompt.json')!=oldrecord['prompt_sha256']
      or digest(folder/'frames.json')!=oldrecord['frames_sha256']): raise ValueError('Original run changed')
  prompt=read(folder/'prompt.json'); oldrows=read(folder/'frames.json'); first=prompt['first_detection_index']; frames=videos[name]
  if first is None or len(frames)!=len(oldrows): raise ValueError('No recorded first detection')
  if digest(frames[first].image_path)!=oldrows[first]['image_sha256']: raise ValueError('Prompt image changed')
  if digest(folder/'prompt_provisional.png')!=prompt['provisional_mask_sha256'] or digest(folder/'prompt_refined.png')!=prompt['refined_mask_sha256']:
   raise ValueError('Saved prompt masks changed')
  target=out/name; target.mkdir(); conditions={}
  for condition in ('baseline','candidate'):
   model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),repo)
   before=parameter_fingerprints(model); events=[]
   with tempfile.TemporaryDirectory(prefix='staged_',dir=target) as tmp:
    stage_frames(frames,Path(tmp)); staged=sorted(Path(tmp).glob('*.jpg'),key=lambda q:int(q.stem))
    if len(staged)!=len(frames) or any(q.resolve()!=f.image_path.resolve() for q,f in zip(staged,frames)): raise ValueError('Frame order differs')
    state=model.init_state(video_path=tmp,offload_video_to_cpu=True,offload_state_to_cpu=False)
    from contextlib import nullcontext
    old_choose,old_head=model._use_multimask,model._forward_sam_heads
    try:
     with torch.inference_mode(), (initial_multimask(model,events) if condition=='candidate' else nullcontext()):
      box=np.asarray(prompt['box'],np.float32)
      _,ids,logits=model.add_new_points_or_box(state,frame_idx=first,obj_id=1,box=box)
      if ids!=[1]: raise ValueError('Object ID changed')
      provisional=logits[0,0].cpu().numpy()>0
      _,backbone,_,_,_=model._get_image_feature(state,first,batch_size=1)
      early=backbone['backbone_fpn'][0].detach().float().cpu().contiguous()
      if hashlib.sha256(early.numpy().tobytes()).hexdigest()!=prompt['early_feature_sha256']: raise ValueError('Feature extraction changed')
      points,status=early_feature_points(backbone,provisional,box,seed=20261114)
      if points is not None:
       _,ids,logits=model.add_new_points_or_box(state,frame_idx=first,obj_id=1,points=np.asarray(points['feature_points'],np.float32),
          labels=np.array([1,0],np.int32),clear_old_points=False)
       if ids!=[1]: raise ValueError('Object ID changed')
      mask=logits[0,0].cpu().numpy()>0
    finally: model.reset_state(state)
    if model._use_multimask!=old_choose or model._forward_sam_heads!=old_head: raise ValueError('Multimask context leaked')
   assert_parameter_integrity(model,before,target,condition)
   for suffix,data in [('provisional',provisional),('refined',mask)]: Image.fromarray(data.astype(np.uint8)*255).save(target/f'{condition}_{suffix}.png')
   if condition=='baseline':
    if (not np.array_equal(provisional,np.asarray(Image.open(folder/'prompt_provisional.png'))>0)
        or not np.array_equal(mask,np.asarray(Image.open(folder/'prompt_refined.png'))>0)
        or status!=prompt['point_status'] or (None if points is None else points['feature_points'])!=prompt['points']):
     raise ValueError(f'{name}: baseline prompt differs')
   else:
    if len(events)!=(2 if points is not None else 1): raise ValueError('Unexpected candidate decoder calls')
   conditions[condition]={'provisional_sha256':digest(target/f'{condition}_provisional.png'),'refined_sha256':digest(target/f'{condition}_refined.png'),
    'point_status':status,'points':None if points is None else points['feature_points'],'decoder_events':events}
   del model
  result={'sequence':name,'first_detection_index':first,'image_sha256':oldrows[first]['image_sha256'],
          'original_records_sha256':olddone['records_sha256'],'baseline':conditions['baseline'],'candidate':conditions['candidate']}
  results.append(result); (target/'record.json').write_text(json.dumps(result,indent=2)+'\n')
  (out/'progress.json').write_text(json.dumps({'completed':[x['sequence'] for x in results]},indent=2)+'\n')
  print(f'Completed {name}',flush=True)
 (out/'records.json').write_text(json.dumps(results,indent=2)+'\n'); (out/'completion.json').write_text(json.dumps({'passed':True,'records_sha256':digest(out/'records.json')},indent=2)+'\n')
if __name__=='__main__':main()
