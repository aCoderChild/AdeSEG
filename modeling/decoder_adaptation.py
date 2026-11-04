"""Ordinary decoder adaptation with pinned SAM2 author losses."""
import ast
from collections import defaultdict
from functools import lru_cache
from typing import Dict, List

import torch
from torch import nn
import torch.nn.functional as F

from modeling.region_tokens import SAM2_REVISION, _verified_source


def train_decoder_core(model):
    model.eval()
    model.requires_grad_(False)
    for name, parameter in model.sam_mask_decoder.named_parameters():
        if not name.startswith(('conv_s0.', 'conv_s1.')):
            parameter.requires_grad_(True)
    assert_feature_cache_safe(model)
    selected = [(name, p) for name, p in model.named_parameters() if p.requires_grad]
    if not selected:
        raise ValueError('Decoder has no trainable core parameters')
    return selected


def assert_feature_cache_safe(model):
    modules = (model.image_encoder, model.sam_mask_decoder.conv_s0, model.sam_mask_decoder.conv_s1)
    if model.image_encoder.training or any(p.requires_grad for module in modules for p in module.parameters()):
        raise ValueError('Cached image features require a frozen encoder and high-resolution projections')


@lru_cache(maxsize=1)
def author_loss_class():
    namespace = {'torch':torch,'dist':torch.distributed,'nn':nn,'F':F,
                 'defaultdict':defaultdict,'Dict':Dict,'List':List}
    specs = [
        ('training/utils/distributed.py', {'is_dist_avail_and_initialized','get_world_size'}),
        ('training/loss_fns.py', {'dice_loss','sigmoid_focal_loss','iou_loss','MultiStepMultiMasksAndIous'})]
    trainer_path, trainer = _verified_source('sam2', SAM2_REVISION, 'training/trainer.py')
    assignments = [node for node in ast.parse(trainer).body if isinstance(node,ast.Assign)
        and any(isinstance(t,ast.Name) and t.id=='CORE_LOSS_KEY' for t in node.targets)]
    if len(assignments)!=1:
        raise RuntimeError('Author loss key differs')
    exec(compile(ast.Module(body=assignments,type_ignores=[]),str(trainer_path),'exec'),namespace)
    for name, required in specs:
        path, source = _verified_source('sam2', SAM2_REVISION, name)
        nodes = [n for n in ast.parse(source).body if isinstance(n,(ast.FunctionDef,ast.ClassDef)) and n.name in required]
        if {n.name for n in nodes}!=required:
            raise RuntimeError('Pinned author loss definitions differ')
        exec(compile(ast.Module(body=nodes,type_ignores=[]),str(path),'exec'),namespace)
    return namespace['MultiStepMultiMasksAndIous'], namespace['CORE_LOSS_KEY']


def decoder_loss(captured, target, *, resolution=1024):
    if len(captured)!=1 or target.ndim!=4 or target.shape[1]!=1:
        raise ValueError('Expected one decoder call and one-channel target masks')
    masks, ious, _, score = captured[0]
    masks = F.interpolate(masks.float(),size=(resolution,resolution),mode='bilinear',align_corners=False)
    target = F.interpolate(target.float(),size=(resolution,resolution),mode='nearest')
    cls, key = author_loss_class()
    objective = cls({'loss_mask':20,'loss_dice':1,'loss_iou':1,'loss_class':1},
        supervise_all_iou=True,iou_use_l1_loss=True,pred_obj_scores=True)
    outputs = {'multistep_pred_multimasks_high_res':[masks],
        'multistep_pred_ious':[ious], 'multistep_object_score_logits':[score]}
    losses = objective([outputs],target[:,0].unsqueeze(0))
    return losses[key], losses
