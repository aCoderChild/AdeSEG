"""RGB analogue of the early-FPN automatic-prompt contrast diagnostic."""
import numpy as np
import torch
import torch.nn.functional as F
from modeling.prompt_feature_refinement import reference_prototypes
from modeling.lean_early_contrast_anchor import blend_anchor_with_early_contrast


def rgb_grid(image):
    image=np.asarray(image)
    if image.ndim!=3 or image.shape[2]!=3 or image.dtype!=np.uint8:raise ValueError('Expected RGB uint8 image')
    x=torch.from_numpy(image.copy()).permute(2,0,1).float()/255
    # Same normalized image coordinates as the square video-encoder input.
    return F.interpolate(x[None],size=(256,256),mode='area')[0]


def rgb_reference(image,prompt_mask):
    return reference_prototypes(rgb_grid(image),prompt_mask)


def rgb_contrast(image,prototypes):
    feature=rgb_grid(image);positive,negative=prototypes
    return blend_anchor_with_early_contrast(torch.zeros(4096),feature,positive,negative,(64,64),1.).clamp(-1,1)
