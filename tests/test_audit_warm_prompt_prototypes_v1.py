import numpy as np
import pytest
import torch
from modeling.prompt_feature_refinement import reference_prototypes
from scripts.audit_warm_prompt_prototypes_v1 import validate_prompt_prototypes
from scripts.memory_token_io import digest


def test_prototypes_are_reconstructed_from_archived_mask_and_prompt_early_feature(tmp_path):
    torch.manual_seed(21);early=torch.rand(32,256,256)
    mask=np.zeros((72,84),dtype=bool);mask[15:30,20:40]=True
    fg,bg=reference_prototypes(early,mask)
    ep=tmp_path/'prompt_early.pt';pp=tmp_path/'prototypes.pt'
    torch.save({'frame':0,'early':early},ep);torch.save({'foreground':fg,'background':bg},pp)
    rec={'prompt_frame':0,'prompt_early_sha256':digest(ep),'prototypes_sha256':digest(pp)}
    assert torch.equal(validate_prompt_prototypes(tmp_path,rec,mask)['foreground'],fg)
    altered=np.zeros_like(mask);altered[35:50,45:65]=True
    with pytest.raises(ValueError,match='do not derive from archived'):validate_prompt_prototypes(tmp_path,rec,altered)
    rec['prompt_frame']=1
    with pytest.raises(ValueError,match='geometry differs'):validate_prompt_prototypes(tmp_path,rec,mask)
