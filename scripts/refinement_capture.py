"""Read-only capture of actual native memory reads and decoder input features."""
import hashlib
import inspect
from unittest.mock import patch

import numpy as np
import torch

from modeling.region_bias import feature_key
from modeling.region_definitions import mixed_cell_regions
from modeling.regional_refinement import pooled_context


class RefinementCapture:
    def __init__(self, model, sequence):
        self.model, self.sequence = model, sequence
        self.prepare = model._prepare_memory_conditioned_features
        self.signature = inspect.signature(self.prepare)
        self.frame = None
        self.sample = None
        self.labels = {}

    def prepare_read(self,*args,**kwargs):
        bound=self.signature.bind(*args,**kwargs);bound.apply_defaults()
        c=bound.arguments;self.frame=c['frame_idx'];self.sample=None;self.bank={}
        if not c['is_init_cond_frame']:
            for outputs in c['output_dict'].values():
                for f,stored in outputs.items():
                    if f>=self.frame or stored.get('maskmem_features') is None:continue
                    vectors=stored['maskmem_features'].flatten(2).transpose(1,2)
                    key=feature_key(vectors)
                    if key in self.bank:raise RuntimeError('Ambiguous memory frame identity')
                    if f not in self.labels:
                        self.labels[f]=torch.from_numpy(mixed_cell_regions(stored['pred_masks'][0,0])['labels']).flatten()
                    self.bank[key]=(f,self.labels[f])
        return self.prepare(*args,**kwargs)

    def memory_read(self,module,args,kwargs):
        if args:raise RuntimeError('Unexpected native memory signature')
        memory=kwargs['memory'].detach();pointers=int(kwargs.get('num_obj_ptr_tokens',0))
        spatial=len(memory)-pointers
        if spatial<=0 or spatial%4096:raise RuntimeError('Invalid native spatial grid')
        features=[];labels=[];shuffled=[];frame_ids=[]
        for start in range(0,spatial,4096):
            chunk=memory[start:start+4096].transpose(0,1)
            f,lab=self.bank[feature_key(chunk)]
            features.append(chunk[0].float().cpu());labels.append(lab);frame_ids.append(f)
            seed=int.from_bytes(hashlib.sha256(f'{self.sequence}/{f}/20261031'.encode()).digest()[:8],'little')%(2**63)
            permutation=torch.randperm(4096,generator=torch.Generator().manual_seed(seed))
            shuffled.append(lab[permutation])
        vectors=torch.cat(features);lab=torch.cat(labels);shuffle=torch.cat(shuffled)
        regional,valid,counts=pooled_context(vectors,lab)
        random,random_valid,random_counts=pooled_context(vectors,shuffle)
        if not torch.equal(counts,random_counts):raise RuntimeError('Shuffled counts changed')
        generic=vectors.mean(0)[:,None,None].expand(64,3,1).clone()
        self.sample={'contexts':{'regional':regional,'shuffled':random,'generic':generic},
                     'valid':{'regional':valid,'shuffled':random_valid,'generic':torch.ones(3,dtype=torch.bool)},
                     'counts':counts,'memory_frames':frame_ids,'spatial_keys':spatial,'pointer_keys':pointers}

    def decoder_input(self,module,args,kwargs):
        if self.sample is not None:
            self.sample['query']=kwargs['image_embeddings'].detach().float().cpu().clone()[0]

    def __enter__(self):
        self.override=patch.object(self.model,'_prepare_memory_conditioned_features',self.prepare_read)
        self.override.start()
        self.handles=[self.model.memory_attention.register_forward_pre_hook(self.memory_read,with_kwargs=True),
                      self.model.sam_mask_decoder.register_forward_pre_hook(self.decoder_input,with_kwargs=True)]
        return self

    def __exit__(self,*exc):
        for h in self.handles:h.remove()
        self.override.stop()
