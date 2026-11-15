"""Matched-key nonconditioning memory swaps between frozen reads."""
import torch

def _equal(a,b):
 if torch.is_tensor(a):return torch.is_tensor(b) and torch.equal(a,b)
 if isinstance(a,(list,tuple)):return type(a) is type(b) and len(a)==len(b) and all(_equal(x,y) for x,y in zip(a,b))
 return a==b

def swap_nonconditioning_spatial(recipient,donor):
 """Replace noncondition spatial keys/positions, retaining recipient pointers.

 Both reads must share the actual current query and full conditioning segment.
 The swap is a diagnostic between trajectory states, not an online predictor.
 """
 for key in ('curr','curr_pos','memory','memory_pos','num_obj_ptr_tokens','return_attn_stat'):
  if key not in recipient or key not in donor:raise ValueError('Missing attention input')
 if not _equal(recipient['curr'],donor['curr']) or not _equal(recipient['curr_pos'],donor['curr_pos']) or recipient['return_attn_stat']!=donor['return_attn_stat']:raise ValueError('Current image query differs')
 a,ap=recipient['memory'],recipient['memory_pos'];b,bp=donor['memory'],donor['memory_pos'];ptr=recipient['num_obj_ptr_tokens']
 if type(ptr) is not int or ptr!=donor['num_obj_ptr_tokens'] or a.ndim!=3 or ap.ndim!=3 or b.ndim!=3 or bp.ndim!=3 or a.shape!=b.shape or ap.shape!=bp.shape or a.shape[:2]!=ap.shape[:2] or a.shape[1]!=1 or not 0<=ptr<len(a)-4096:raise ValueError('Memory geometry or pointer count differs')
 if not torch.equal(a[:4096],b[:4096]) or not torch.equal(ap[:4096],bp[:4096]):raise ValueError('Conditioning segment differs')
 end=len(a)-ptr if ptr else len(a)
 if end<=4096:raise ValueError('No nonconditioning spatial memory')
 result=dict(recipient);result['memory']=torch.cat((a[:4096],b[4096:end],a[end:]),dim=0);result['memory_pos']=torch.cat((ap[:4096],bp[4096:end],ap[end:]),dim=0)
 if not torch.equal(result['memory'][end:],a[end:]) or not torch.equal(result['memory_pos'][end:],ap[end:]):raise ValueError('Recipient pointer tail changed')
 return result
