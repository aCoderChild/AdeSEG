#!/usr/bin/env python3
"""Propagate a verified native-multimask initial prompt with indexed Lean."""
import argparse,json,sys,tempfile
from pathlib import Path
from types import SimpleNamespace
import numpy as np
from PIL import Image
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from datasets.io import load_label_mask
from datasets.video import load_manifest,stage_frames
from modeling.early_feature_prompt import early_feature_points
from modeling.lean_initial_multimask import initial_multimask
from modeling.lean_multiscale_anchor import install_multiscale_anchor
from modeling.lean_indexed_spatial_rope import indexed_spatial_rope
from scripts.memory_token_io import digest,validate_output
from scripts.run_author_baseline import source_identity,build_author_predictor,parameter_fingerprints,assert_parameter_integrity
PROTOCOL=ROOT/'protocols/lean_initial_multimask_video_v1.json';MANIFEST=ROOT/'data/polypgen_sequence.jsonl'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for n in ('checkpoint','merged','indexed-root','indexed-verification','prompt-root','prompt-verification','prompt-audit','output'):p.add_argument('--'+n,type=Path,required=True)
 p.add_argument('--sequence',required=True);p.add_argument('--mode',choices=('gate','candidate'),required=True);p.add_argument('--gate-a',type=Path);p.add_argument('--gate-b',type=Path)
 a=p.parse_args();proto=read(PROTOCOL)
 if a.sequence not in proto['cohort'] or (a.mode=='gate' and a.sequence!='seq3'):raise ValueError('Sequence out of scope')
 indexed=read(a.indexed_verification); prompt_v=read(a.prompt_verification); audit=read(a.prompt_audit/'completion.json'); merged=read(a.merged/'completion.json')
 if (not indexed['passed'] or not prompt_v['passed'] or not audit['passed'] or not merged['passed']
     or digest(a.prompt_audit/'results.json')!=audit['results_sha256'] or digest(a.merged/'results.json')!=merged['results_sha256']
     or indexed['identity']['cpu_identity']['merged_results_sha256']!=merged['results_sha256']
     or prompt_v['identity']['indexed_verification_sha256']!=digest(a.indexed_verification)
     or prompt_v['identity']['module_sha256']!=digest(ROOT/'modeling/lean_initial_multimask.py')
     or indexed['identity']['adapter_sha256']!=digest(ROOT/'modeling/lean_indexed_spatial_rope.py')
     or indexed['identity']['lookup_sha256']!=digest(ROOT/'modeling/lean_memory_position_lookup.py')):raise ValueError('Verified sources or prompt audit changed')
 sig=indexed['identity']['cpu_identity']['source_signature'];pinned={'early_module_sha256':ROOT/'modeling/early_feature_prompt.py','prompt_module_sha256':ROOT/'modeling/prompt_feature_refinement.py','memory_module_sha256':ROOT/'modeling/lean_multiscale_anchor.py','lean_runner_sha256':ROOT/'scripts/run_author_baseline.py','runner_sha256':ROOT/'scripts/run_lean_auto_multiscale.py','manifest_sha256':MANIFEST,'bypass_sha256':ROOT/'external/Lean-SAM2/checkpoints/bypass_bottleneck_base.pth','lean_config_sha256':ROOT/'external/Lean-SAM2/configs/lean_sam2/lean_sam2.yaml'}
 if digest(a.checkpoint)!=sig['checkpoint_sha256'] or any(digest(path)!=sig[key] for key,path in pinned.items()) or torch.__version__!=sig['torch'] or np.__version__!=sig['numpy'] or sys.version!=sig['python']:raise ValueError('Runtime or model changed')
 identity={'runner_sha256':digest(Path(__file__)),'protocol_sha256':digest(PROTOCOL),'indexed_verification_sha256':digest(a.indexed_verification),
  'prompt_verification_sha256':digest(a.prompt_verification),'prompt_audit_sha256':digest(a.prompt_audit/'results.json'), 'source_signature':sig}
 if a.mode=='candidate':
  if not a.gate_a or not a.gate_b:raise ValueError('Two video gate replays required')
  gate_records=[]
  for g in (a.gate_a,a.gate_b):
   st,done=read(g/'setup.json'),read(g/'completion.json')
   if st['identity']!=identity or st['mode']!='gate' or not done['passed'] or digest(g/'records.json')!=done['records_sha256']:raise ValueError('Video gate invalid')
   gate_records.append(read(g/'records.json'))
  if gate_records[0]!=gate_records[1] or gate_records[0]['changed_masks']<=0:raise ValueError('Video gates differ or have no effect')
 refs=[d/a.sequence for d in (a.indexed_root/'part_a_attempt01',a.indexed_root/'part_b_attempt01') if (d/a.sequence/'completion.json').exists()]
 prompt_parts=[d for d in (a.prompt_root/'part_a_attempt01',a.prompt_root/'part_b_attempt01') if (d/a.sequence/'record.json').exists()]
 if len(refs)!=1 or len(prompt_parts)!=1:raise ValueError('Reference or prompt missing')
 refdir=refs[0];ref=read(refdir/'records.json');refdone=read(refdir/'completion.json');refsetup=read(refdir/'setup.json')
 if (not refdone['passed'] or digest(refdir/'records.json')!=refdone['records_sha256'] or refsetup['mode']!='candidate'
     or refsetup['identity']!=indexed['identity'] or refsetup['sequence']!=a.sequence or digest(refdir/'frames.json')!=ref['frames_sha256']):raise ValueError('Indexed reference changed')
 promptpart=prompt_parts[0];psetup=read(promptpart/'setup.json');pdone=read(promptpart/'completion.json');pall=read(promptpart/'records.json');matches=[x for x in pall if x['sequence']==a.sequence]
 if (not pdone['passed'] or digest(promptpart/'records.json')!=pdone['records_sha256'] or psetup['identity']!=prompt_v['identity'] or len(matches)!=1 or matches[0]!=read(promptpart/a.sequence/'record.json')):raise ValueError('Prompt run invalid')
 item=matches[0];pfolder=promptpart/a.sequence;pc=item['candidate'];
 for kind in ('provisional','refined'):
  if digest(pfolder/f'candidate_{kind}.png')!=pc[kind+'_sha256']:raise ValueError('Candidate prompt changed')
 source=Path(read(a.merged/'locations.json')[a.sequence]); folder=source/a.sequence; olddone=read(source/'completion.json');old=read(source/'records.json')[0];oldrows=read(folder/'frames.json')
 if digest(source/'records.json')!=olddone['records_sha256'] or olddone['records_sha256']!=item['original_records_sha256'] or digest(folder/'frames.json')!=old['frames_sha256'] or digest(folder/'prompt.json')!=old['prompt_sha256']:raise ValueError('Original input changed')
 prompt=read(folder/'prompt.json');index=prompt['first_detection_index'];videos={**load_manifest(MANIFEST,'dev'),**load_manifest(MANIFEST,'test')};frames=videos[a.sequence]
 if index!=item['first_detection_index'] or len(frames)!=len(oldrows):raise ValueError('Input order changed')
 for frame,row in zip(frames,oldrows):
  if digest(frame.image_path)!=row['image_sha256']:raise ValueError('Input image changed')
 out=validate_output(a.output);out.mkdir(parents=True,exist_ok=False)
 (out/'setup.json').write_text(json.dumps({'mode':a.mode,'sequence':a.sequence,'identity':identity,'indexed_reference_sha256':refdone['records_sha256'],'prompt_record_sha256':digest(pfolder/'record.json')},indent=2)+'\n')
 torch.set_num_threads(2);repo,revision=source_identity('lean')
 if revision!=sig['revision']:raise ValueError('Lean source differs')
 model=build_author_predictor(SimpleNamespace(method='lean',checkpoint=a.checkpoint,device='cpu'),repo)
 if model.mem_keep_ratio!=.25:raise ValueError('Memory budget changed')
 before=parameter_fingerprints(model);selections=[];events=install_multiscale_anchor(model,.5,selection_events=selections);positions=[];prompt_events=[]
 for i in range(index):
  with Image.open(frames[i].image_path) as im:w,h=im.size
  Image.fromarray(np.zeros((h,w),np.uint8)).save(out/f'{i:06d}.png')
 with tempfile.TemporaryDirectory(prefix='staged_',dir=out) as tmp:
  stage_frames(frames,Path(tmp));staged=sorted(Path(tmp).glob('*.jpg'),key=lambda q:int(q.stem))
  if len(staged)!=len(frames) or any(q.resolve()!=f.image_path.resolve() for q,f in zip(staged,frames)):raise ValueError('Frame order changed')
  state=model.init_state(video_path=tmp,offload_video_to_cpu=True,offload_state_to_cpu=False);old_choose,old_head=model._use_multimask,model._forward_sam_heads
  try:
   with torch.inference_mode(),indexed_spatial_rope(model,positions):
    with initial_multimask(model,prompt_events):
     box=np.asarray(prompt['box'],np.float32);_,ids,logits=model.add_new_points_or_box(state,frame_idx=index,obj_id=1,box=box)
     if ids!=[1]:raise ValueError('Object changed')
     provisional=logits[0,0].cpu().numpy()>0
     if not np.array_equal(provisional,np.asarray(Image.open(pfolder/'candidate_provisional.png'))>0):raise ValueError('Candidate box mask differs')
     _,backbone,_,_,_=model._get_image_feature(state,index,batch_size=1);points,status=early_feature_points(backbone,provisional,box,seed=20261114)
     if status!=pc['point_status'] or (None if points is None else points['feature_points'])!=pc['points']:raise ValueError('Candidate points differ')
     if points is not None:
      _,ids,logits=model.add_new_points_or_box(state,frame_idx=index,obj_id=1,points=np.asarray(points['feature_points'],np.float32),labels=np.array([1,0],np.int32),clear_old_points=False)
      if ids!=[1]:raise ValueError('Object changed')
     initial=logits[0,0].cpu().numpy()>0
     if not np.array_equal(initial,np.asarray(Image.open(pfolder/'candidate_refined.png'))>0) or prompt_events!=pc['decoder_events']:raise ValueError('Candidate prompt differs')
    if model._use_multimask!=old_choose or model._forward_sam_heads!=old_head:raise ValueError('Prompt override leaked into propagation')
    if initial.any():model.add_new_mask(state,frame_idx=index,obj_id=1,mask=initial)
    gen=model.propagate_in_video(state,start_frame_idx=index,reverse=False)
    for expected in range(index,len(frames)):
     actual,ids,output=next(gen)
     if actual!=expected or ids!=[1]:raise ValueError('Prediction order changed')
     Image.fromarray((output[0,0].cpu().numpy()>0).astype(np.uint8)*255).save(out/f'{actual:06d}.png')
    try:next(gen);raise ValueError('Unexpected frame')
    except StopIteration:pass
  finally:model.reset_state(state)
 assert_parameter_integrity(model,before,out,'after');masks=[digest(out/f'{i:06d}.png') for i in range(len(frames))]
 if not positions:raise ValueError('No position events')
 for filename,data in [('positions',positions),('selections',selections),('memory_events',events),('prompt_events',prompt_events)]: (out/f'{filename}.json').write_text(json.dumps(data,indent=2)+'\n')
 if a.mode=='candidate':
  scored=[]
  for i,frame in enumerate(frames):
   if digest(frame.mask_path)!=oldrows[i]['label_sha256']:raise ValueError('Label changed')
   truth=load_label_mask(frame.mask_path,[1])==1;pred=np.asarray(Image.open(out/f'{i:06d}.png'))>0
   if pred.shape!=truth.shape:raise ValueError('Shape changed')
   den=int(pred.sum()+truth.sum());present=bool(truth.any());scored.append({'frame':i,'present':present,'dice':2*int((pred&truth).sum())/den if den else 1.,'false_alarm':bool(pred.any()) if not present else None,'mask_sha256':masks[i]})
  (out/'frames.json').write_text(json.dumps(scored,indent=2)+'\n')
 record={'sequence':a.sequence,'frames':len(frames),'mask_sha256':masks,'changed_masks':sum(x!=y for x,y in zip(masks,ref['mask_sha256'])),
  **{name+'_sha256':digest(out/f'{name}.json') for name in ('positions','selections','memory_events','prompt_events')},
  'indexed_reference_sha256':refdone['records_sha256'],'prompt_record_sha256':digest(pfolder/'record.json'),'frames_sha256':digest(out/'frames.json') if a.mode=='candidate' else None}
 (out/'records.json').write_text(json.dumps(record,indent=2)+'\n');(out/'completion.json').write_text(json.dumps({'passed':True,'records_sha256':digest(out/'records.json')},indent=2)+'\n')
 print(json.dumps({k:v for k,v in record.items() if k!='mask_sha256'},indent=2))
if __name__=='__main__':main()
