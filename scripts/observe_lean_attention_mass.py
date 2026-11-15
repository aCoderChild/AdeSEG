#!/usr/bin/env python3
"""Observe author per-key usage while exactly replaying indexed Lean videos."""
import argparse,json,sys
from pathlib import Path
from types import MethodType
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from modeling.lean_attention_mass import summarize_key_usage
from scripts.memory_token_io import digest
import scripts.run_lean_indexed_spatial_rope as runner
PROTOCOL=ROOT/'protocols/lean_attention_mass_diagnostic_v1.json'
def read(p):return json.loads(p.read_text())
def main():
 p=argparse.ArgumentParser()
 for x in ('checkpoint','merged','reference','indexed-reference','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--sequence',required=True);p.add_argument('--label-free-gate',action='store_true');a=p.parse_args()
 if (a.label_free_gate and a.sequence!='seq3') or (not a.label_free_gate and a.sequence not in read(PROTOCOL)['cohort']):raise ValueError('Out-of-scope sequence')
 ref=read(a.indexed_reference/'records.json');done=read(a.indexed_reference/'completion.json')
 if not done['passed'] or digest(a.indexed_reference/'records.json')!=done['records_sha256']:raise ValueError('Indexed reference invalid')
 events=[];state={'frame':None,'last_usage':None};build=runner.build_author_predictor
 def observed_build(args,repo):
  model=build(args,repo);get_feature=model._get_image_feature;attn=model.memory_attention.forward;dump=model._dump_retained_tokens_record
  grid=model.memory_encoder.position_encoding(torch.zeros(1,model.mem_dim,64,64)).flatten(2).permute(2,0,1).contiguous()
  cond_pos=grid+model.maskmem_tpos_enc[model.num_maskmem-1].detach()
  def feature(self,inference_state,frame_idx,batch_size):
   out=get_feature(inference_state,frame_idx,batch_size);state['frame']=int(frame_idx);return out
  def attention(self,*args,**kwargs):
   out=attn(*args,**kwargs)
   if not kwargs.get('return_attn_stat') or not isinstance(out,tuple) or out[1] is None:raise ValueError('Author usage unavailable')
   usage=out[1].detach();pos=kwargs['memory_pos'];ptr=int(kwargs['num_obj_ptr_tokens'])
   if state['frame'] is None or pos.shape[0]!=usage.numel() or pos.shape[1]!=1 or not torch.equal(pos[:4096],cond_pos):
    raise ValueError('Conditioning segment or frame identity differs')
   summary=summarize_key_usage(usage,4096,ptr)
   events.append({'frame':state['frame'],'keys':int(usage.numel()),'conditioning_keys':4096,'pointer_keys':ptr,
                  **summary,'latest_full_memory':None})
   state['last_usage']=usage
   return out
  def record(self,**kwargs):
   dump(**kwargs)
   if not events or events[-1]['frame']!=int(kwargs['frame_idx']) or state['last_usage'] is None:raise ValueError('Selection does not match current attention read')
   usage=state['last_usage'];local=kwargs['usage_scores'].detach().reshape(-1);keep=kwargs['keep_idx'].detach().long().reshape(-1)
   end=usage.numel()-events[-1]['pointer_keys'];start=end-local.numel()
   if start<4096 or not torch.equal(usage[start:end],local):raise ValueError('Latest segment does not match author usage')
   if keep.numel()!=1024 or local.numel()!=4096 or torch.unique(keep).numel()!=1024:raise ValueError('Token budget differs')
   total=usage.sum()
   events[-1]['latest_full_memory']={'source_frame':kwargs['latest_prev'].get('frame_idx'),'start':int(start),'end':int(end),
      'mass_before_pruning':float((local.sum()/total).item()),'retained_mass_at_this_read':float((local[keep].sum()/total).item())}
   state['last_usage']=None
  model._get_image_feature=MethodType(feature,model);model.memory_attention.forward=MethodType(attention,model.memory_attention)
  model._dump_retained_tokens_record=MethodType(record,model)
  return model
 runner.build_author_predictor=observed_build
 args=['run_lean_indexed_spatial_rope.py','--mode','gate' if a.label_free_gate else 'candidate','--sequence',a.sequence]
 for key in ('checkpoint','merged','reference','cpu_verification','observed_reference','output'):args.extend(['--'+key.replace('_','-'),str(getattr(a,key))])
 args.extend(['--gate-a',str(a.indexed_gate_a),'--gate-b',str(a.indexed_gate_b)]);sys.argv=args
 runner.main();record=read(a.output/'records.json');keys=('mask_sha256','positions_sha256','selections_sha256','memory_events_sha256')
 if not a.label_free_gate:keys+=('frames_sha256',)
 for key in keys:
  if record[key]!=ref[key]:raise ValueError(f'Observation changed reference: {key}')
 if not events:raise ValueError('No attention observations')
 (a.output/'attention_events.json').write_text(json.dumps(events,indent=2)+'\n')
 result={'passed':True,'sequence':a.sequence,'label_free_gate':a.label_free_gate,'events':len(events),'selection_events':sum(x['latest_full_memory'] is not None for x in events),
  'reference_records_sha256':done['records_sha256'],'replay_records_sha256':digest(a.output/'records.json'),
  'attention_events_sha256':digest(a.output/'attention_events.json'),'protocol_sha256':digest(PROTOCOL),'source_sha256':digest(Path(__file__)),
  'summary_source_sha256':digest(ROOT/'modeling/lean_attention_mass.py')}
 (a.output/'attention_observation.json').write_text(json.dumps(result,indent=2)+'\n');print(json.dumps(result,indent=2))
if __name__=='__main__':main()
