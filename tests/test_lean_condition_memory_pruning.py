import types
import pytest
import torch
from modeling.lean_condition_memory_pruning import choose_condition_indices,condition_memory_pruning

def test_uniform_and_usage_select_distinct_valid_spatial_indices():
 u=torch.arange(4096,dtype=torch.float32)
 assert torch.equal(choose_condition_indices(u,'usage'),torch.topk(u,1024).indices)
 assert torch.equal(choose_condition_indices(u,'uniform'),torch.arange(4096).reshape(64,64)[::2,::2].reshape(-1))
 assert choose_condition_indices(u,'full').numel()==4096
 with pytest.raises(ValueError):choose_condition_indices(torch.ones(10),'usage')

class Encoding:
 def __call__(self,_):
  v=torch.arange(4096).float().reshape(1,1,64,64)
  return torch.cat([v,v*2],dim=1)
class Attention:
 def forward(self,*,memory_pos,num_obj_ptr_tokens,return_attn_stat):
  return torch.tensor(0),torch.arange(memory_pos.shape[0]).float()+1
 def __call__(self,**kw):return self.forward(**kw)
class Model:
 def __init__(self):
  self.mem_dim=2;self.num_maskmem=2;self.maskmem_tpos_enc=torch.tensor([[[[0.,0.]]],[[[.25,.5]]]])
  self.memory_encoder=types.SimpleNamespace(position_encoding=Encoding());self.memory_attention=Attention()
  self.position=Encoding()(None).flatten(2).permute(2,0,1).contiguous();self.features=self.position+3
 def _get_image_feature(self,inference_state,frame_idx,batch_size):return None
 def _materialize_memory_entry(self,prev,device,*,is_cond=False):return self.features,self.position

@pytest.mark.parametrize('strategy',['full','usage','uniform'])
def test_context_aligns_features_and_positions_without_mutation(strategy):
 model=Model();before=model.features.clone();old=(model._materialize_memory_entry,model._get_image_feature,model.memory_attention.forward);events=[]
 with condition_memory_pruning(model,strategy,events):
  model._get_image_feature({},1,1);f,p=model._materialize_memory_entry({},'cpu',is_cond=True)
  assert torch.equal(f,before) and torch.equal(p,model.position)
  usage=model.memory_attention(memory_pos=model.position+model.maskmem_tpos_enc[1],num_obj_ptr_tokens=0,return_attn_stat=True)[1]
  f,p=model._materialize_memory_entry({},'cpu',is_cond=True);idx=choose_condition_indices(usage[:4096],strategy)
  assert torch.equal(f,model.features[idx]) and torch.equal(p,model.position[idx])
  assert torch.equal(model.features,before)
 assert (model._materialize_memory_entry,model._get_image_feature,model.memory_attention.forward)==old
 assert [x['event'] for x in events]==['conditioning_read','selection','conditioning_read']
 assert events[-1]['tokens']==(4096 if strategy=='full' else 1024)

def test_restores_on_condition_memory_mismatch():
 model=Model();old=model._materialize_memory_entry
 with pytest.raises(ValueError,match='conditioning memory grid'):
  with condition_memory_pruning(model,'usage',[]):
   model.features=model.features[:10];model._materialize_memory_entry({},'cpu',is_cond=True)
 assert model._materialize_memory_entry==old
