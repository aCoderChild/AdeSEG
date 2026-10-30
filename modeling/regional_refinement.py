"""UACANet residual head adapted to native SAM2 memory contexts.

MIT-licensed author convolution/attention blocks are loaded from the pinned
submodule. AST adaptation replaces its current-image pooling with supplied
contexts and masks empty groups. No upstream source is modified.
"""
import ast
from functools import lru_cache

import torch
from torch import nn
import torch.nn.functional as F

from modeling.region_tokens import _verified_source, SAM2_REVISION

UACA_REVISION = '6b9e54eb8e94a29cf179a9a2ec2e98f02dfcb056'


@lru_cache(maxsize=1)
def author_components():
    namespace = {'torch':torch, 'nn':nn, 'F':F}
    path, source = _verified_source('UACANet', UACA_REVISION, 'lib/modules/layers.py')
    conv = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='conv')
    exec(compile(ast.Module(body=[conv],type_ignores=[]),str(path),'exec'),namespace)
    path, source = _verified_source('UACANet', UACA_REVISION, 'lib/modules/attention_module.py')
    cls = next(n for n in ast.parse(source).body if isinstance(n,ast.ClassDef) and n.name=='UACA')
    forward = next(n for n in cls.body if isinstance(n,ast.FunctionDef) and n.name=='forward')
    # Preserve author query/key/value, attention, feature fusion and residual output.
    # Drop only its image-derived class probabilities/context pooling.
    start = next(i for i,n in enumerate(forward.body) if isinstance(n,ast.Assign) and ast.unparse(n.targets[0])=='query')
    forward.body = forward.body[:2] + forward.body[start:]
    forward.args.args.extend([ast.arg(arg='context'), ast.arg(arg='valid')])
    softmax = next(i for i,n in enumerate(forward.body) if 'F.softmax' in ast.unparse(n))
    forward.body[softmax:softmax] = ast.parse("sim = sim.masked_fill(~valid[:, None, :], float('-inf'))").body
    exec(compile(ast.fix_missing_locations(ast.Module(body=[cls],type_ignores=[])),str(path),'exec'),namespace)
    path, source = _verified_source('sam2', SAM2_REVISION, 'training/loss_fns.py')
    dice = next(n for n in ast.parse(source).body if isinstance(n,ast.FunctionDef) and n.name=='dice_loss')
    exec(compile(ast.Module(body=[dice],type_ignores=[]),str(path),'exec'),namespace)
    return namespace['conv'], namespace['UACA'], namespace['dice_loss']


def pooled_context(features, labels):
    """Explicit row-major [tokens,64] pooling; no mixing channels with positions."""
    if features.ndim != 2 or features.shape[1] != 64 or labels.shape != features.shape[:1]:
        raise ValueError('Expected aligned [N,64] memory vectors and [N] region labels')
    if not ((labels>=0)&(labels<=2)).all() or not torch.isfinite(features).all():
        raise ValueError('Invalid labels/features')
    weights = F.one_hot(labels.long(),3).to(features.dtype).T
    counts = weights.sum(1)
    context = (weights @ features) / counts.clamp_min(1)[:,None]
    return context.T.contiguous().unsqueeze(-1), counts>0, counts


class RegionalRefinement(nn.Module):
    def __init__(self):
        super().__init__()
        conv, uaca, _ = author_components()
        self.project = conv(256,64,1,bn=False)
        self.head = uaca(64,32)
        # Logit correction must be exactly zero initially, with no output BN.
        self.head.conv_out4.bn = None
        nn.init.zeros_(self.head.conv_out4.conv.weight)

    def forward(self, query, contexts, valid, native_logits, object_score):
        if contexts.shape[1:] != (64,3,1) or valid.shape[1:] != (3,) or not valid.any(1).all():
            raise ValueError('Invalid context shape or all-empty memory')
        x = self.project(query)
        _, residual = self.head(x,torch.zeros_like(x[:,:1]),contexts,valid)
        residual = F.interpolate(residual,size=native_logits.shape[-2:],mode='bilinear',align_corners=False)
        # Keep SAM2's native presence decision; do not learn to undo -1024 suppression.
        residual = residual * (object_score.reshape(-1,1,1,1)>0)
        return native_logits + residual


def refinement_loss(logits, target):
    _, _, dice = author_components()
    return F.binary_cross_entropy_with_logits(logits,target) + dice(logits,target.flatten(1),len(logits))
