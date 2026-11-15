import numpy as np
import torch
from modeling.lean_rgb_contrast_control import rgb_reference,rgb_contrast


def test_rgb_control_contrasts_prompt_and_context_in_fixed_coordinates():
    image=np.zeros((32,32,3),np.uint8);image[:,:16,0]=200;image[:,16:,1]=200
    mask=np.zeros((32,32),bool);mask[:,:12]=True
    scores=rgb_contrast(image,rgb_reference(image,mask)).reshape(64,64)
    assert scores[:,:20].mean()>scores[:,44:].mean()
    assert torch.isfinite(scores).all() and scores.max()<=1 and scores.min()>=-1
