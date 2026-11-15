#!/usr/bin/env python3
"""Locate state changes caused by extra no-op Lean attention or decoder calls."""
import argparse,hashlib,json,sys
from contextlib import contextmanager
from pathlib import Path
from types import MethodType
import torch
ROOT=Path(__file__).resolve().parents[1];sys.path.insert(0,str(ROOT))
from scripts.memory_token_io import digest
import scripts.run_lean_indexed_spatial_rope as runner

def read(p):return json.loads(p.read_text())
def snapshot(value):
 if torch.is_tensor(value):return value.clone()
 if isinstance(value,(list,tuple)):return [snapshot(x) for x in value]
 if isinstance(value,dict):return {k:snapshot(v) for k,v in value.items()}
 return value
def equal(a,b):
 if torch.is_tensor(a):return torch.equal(a,b)
 if isinstance(a,(list,tuple)):return len(a)==len(b) and all(equal(x,y) for x,y in zip(a,b))
 if isinstance(a,dict):return a.keys()==b.keys() and all(equal(a[k],b[k]) for k in a)
 return a==b

def main():
 p=argparse.ArgumentParser()
 for x in ('checkpoint','merged','reference','indexed-reference','cpu-verification','observed-reference','indexed-gate-a','indexed-gate-b','output'):p.add_argument('--'+x,type=Path,required=True)
 p.add_argument('--operation',choices=('attention','decoder','attention_perturbed','decoder_perturbed','attention_three','decoder_three','combined'),required=True);a=p.parse_args();events=[];orig_index=runner.indexed_spatial_rope
 @contextmanager
 def probe(model,positions):
  with orig_index(model,positions):
   attention=model.memory_attention.forward;head=model._forward_sam_heads;get_feature=model._get_image_feature;state={'frame':None,'attention_done':False,'decoder_done':False,'shadow':None}
   def feature(self,inference_state,frame_idx,batch_size):
    out=get_feature(inference_state,frame_idx,batch_size);state['frame']=int(frame_idx);return out
   def observe(self,*args,**kwargs):
    out=attention(*args,**kwargs)
    if a.operation in ('attention','attention_perturbed','attention_three','combined') and state['frame']==1 and not state['attention_done']:
     before_rng=torch.get_rng_state().clone();before_args=snapshot((args,kwargs));before_out=snapshot(out);n=len(positions)
     changed=dict(kwargs)
     if a.operation in ('attention_perturbed','combined'):
      changed['memory']=kwargs['memory'].clone();changed['memory'][:4096]=0
     repeated=None
     for _ in range(3 if a.operation=='attention_three' else 1):
      repeated=attention(*args,**changed)
      if len(positions)!=n+1:raise ValueError('Unexpected position trace')
      del positions[n:]
     event={'operation':a.operation,'frame':1,'rng_unchanged':torch.equal(before_rng,torch.get_rng_state()),'inputs_unchanged':equal(before_args,(args,kwargs)),
            'prior_output_unchanged':equal(before_out,out),'repeated_output_equal':equal(before_out,repeated)}
     if a.operation=='combined':state['shadow']=repeated[0]
     events.append(event);state['attention_done']=True;(a.output/'state_probe_partial.json').write_text(json.dumps(events,indent=2)+'\n')
    return out
   def decode(self,*args,**kwargs):
    out=head(*args,**kwargs)
    if a.operation in ('decoder','decoder_perturbed','decoder_three','combined') and state['frame']==1 and not state['decoder_done']:
     before_rng=torch.get_rng_state().clone();before_args=snapshot((args,kwargs));before_out=snapshot(out)
     changed=dict(kwargs)
     if a.operation=='decoder_perturbed':changed['backbone_features']=kwargs['backbone_features']+0.1
     if a.operation=='combined':
      z=state['shadow'];changed['backbone_features']=z.permute(1,2,0).view(1,z.shape[2],64,64)
     repeated=None
     for _ in range(3 if a.operation=='decoder_three' else 1):repeated=head(*args,**changed)
     event={'operation':a.operation,'frame':1,'rng_unchanged':torch.equal(before_rng,torch.get_rng_state()),'inputs_unchanged':equal(before_args,(args,kwargs)),
            'prior_output_unchanged':equal(before_out,out),'repeated_output_equal':equal(before_out,repeated)}
     events.append(event);state['decoder_done']=True;(a.output/'state_probe_partial.json').write_text(json.dumps(events,indent=2)+'\n')
    return out
   model._get_image_feature=MethodType(feature,model);model.memory_attention.forward=MethodType(observe,model.memory_attention);model._forward_sam_heads=MethodType(decode,model)
   try:yield
   finally:model._get_image_feature=get_feature;model.memory_attention.forward=attention;model._forward_sam_heads=head
 runner.indexed_spatial_rope=probe
 sys.argv=['run_lean_indexed_spatial_rope.py','--mode','gate','--sequence','seq3']
 for k in ('checkpoint','merged','reference','cpu_verification','observed_reference','output'):sys.argv.extend(['--'+k.replace('_','-'),str(getattr(a,k))])
 sys.argv.extend(['--gate-a',str(a.indexed_gate_a),'--gate-b',str(a.indexed_gate_b)])
 runner.main();reference=read(a.indexed_reference/'records.json');actual=read(a.output/'records.json')
 for key in ('mask_sha256','selections_sha256','positions_sha256','memory_events_sha256'):
  if reference[key]!=actual[key]:raise ValueError('Extra no-op changed full-video reference: '+key)
 if len(events)!=(2 if a.operation=='combined' else 1):raise ValueError('Missing probe')
 (a.output/'state_probe.json').write_text(json.dumps({'passed':True,'operation':a.operation,'events':events,'source_sha256':digest(Path(__file__))},indent=2)+'\n');print(json.dumps(events,indent=2))
if __name__=='__main__':main()
