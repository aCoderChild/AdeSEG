"""Multi-scale feature aggregation for SAM2 memory attention.

Injects early/mid FPN features as additional memory key/value tokens alongside
the deep feature, so memory cross-attention can condition on fine spatial detail
from all backbone stages rather than only the deepest 32x32 level.

Lean-SAM2 TAMP pruning (mem_keep_ratio=0.25, already active in the builder)
bounds the retained token count, so computation does not explode.

No inference gating, threshold rule, or redetection. Architecture-only.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F


class MultiScaleMemoryFeatures(nn.Module):
    """Project early/mid/deep FPN features into memory-attention key/value tokens.

    Inputs (BCHW format):
        early: (B, 32, 128, 128)   - FPN stage 0
        mid:   (B, 64, 64, 64)     - FPN stage 1
        deep:  (B, 256, 32, 32)    - FPN stage 2 (top-level)

    Outputs:
        memory_tokens: (B, total_tokens, 256)  - appended to memory as K/V
        memory_pos:    (B, total_tokens, 256)  - positional encodings (zeros for
            non-deep scales; TAMP pruning uses usage+anchor scores, not position)
    """

    def __init__(self, early_channels=32, mid_channels=64, deep_channels=256,
                 token_dim=256):
        super().__init__()
        self.token_dim = token_dim
        self.early_proj = nn.Conv2d(early_channels, token_dim, 1)
        self.mid_proj = nn.Conv2d(mid_channels, token_dim, 1)
        self.deep_proj = nn.Conv2d(deep_channels, token_dim, 1)

    def forward(self, early, mid, deep):
        B = deep.shape[0]
        device = deep.device
        dtype = deep.dtype

        early_p = self.early_proj(early)   # (B, 256, 128, 128)
        mid_p = self.mid_proj(mid)         # (B, 256, 64, 64)
        deep_p = self.deep_proj(deep)      # (B, 256, 32, 32)

        early_t = early_p.flatten(2).transpose(1, 2)   # (B, 16384, 256)
        mid_t = mid_p.flatten(2).transpose(1, 2)       # (B, 4096, 256)
        deep_t = deep_p.flatten(2).transpose(1, 2)     # (B, 1024, 256)

        memory_tokens = torch.cat([early_t, mid_t, deep_t], dim=1)  # (B, 21504, 256)
        memory_pos = torch.zeros_like(memory_tokens)

        return memory_tokens, memory_pos

    def token_count(self):
        return 128 * 128 + 64 * 64 + 32 * 32  # 21504


class MultiScaleMemoryHook:
    """Scoped context manager injecting multi-scale features into memory attention.

    Patches ``model._prepare_memory_conditioned_features`` so that early/mid FPN
    features (available in ``current_vision_feats``) are appended as additional
    memory key/value tokens before cross-attention. TAMP pruning bounds retained
    tokens. Always restores the original method on exit.
    """

    def __init__(self, model, aggregator):
        if not isinstance(model, torch.nn.Module):
            raise ValueError('model must be a torch.nn.Module')
        if not isinstance(aggregator, MultiScaleMemoryFeatures):
            raise ValueError('aggregator must be a MultiScaleMemoryFeatures')
        self.model = model
        self.aggregator = aggregator
        self._orig_method = None
        self._had_attr = False
        self._orig_attr = None
        self.calls = {'count': 0}

    def __enter__(self):
        from types import MethodType
        self._orig_method = self.model._prepare_memory_conditioned_features
        self._had_attr = (
            '_prepare_memory_conditioned_features' in self.model.__dict__
        )
        self._orig_attr = self.model.__dict__.get(
            '_prepare_memory_conditioned_features'
        )

        def patched(self_, frame_idx, is_init_cond_frame, current_vision_feats,
                    current_vision_pos_embeds, feat_sizes, output_dict,
                    num_frames, track_in_reverse=False, inference_state=None,
                    obj_idx=None):
            self.calls['count'] += 1

            # Extract multi-scale features before calling original.
            # current_vision_feats is a list: [early, mid, deep] (BCHW).
            ms_tokens = None
            ms_pos = None
            if (isinstance(current_vision_feats, (list, tuple))
                    and len(current_vision_feats) >= 3):
                early = current_vision_feats[0]
                mid = current_vision_feats[1]
                deep = current_vision_feats[2]
                ms_tokens, ms_pos = self.aggregator(early, mid, deep)
                # Transpose to (tokens, B, C) format expected internally
                self.model._ms_tokens = ms_tokens.transpose(0, 1)
                self.model._ms_pos = ms_pos.transpose(0, 1)
            else:
                self.model._ms_tokens = None
                self.model._ms_pos = None

            # Patch memory_attention.forward to append multi-scale tokens
            old_mem_attn_forward = self_.memory_attention.forward

            def mem_attn_forward_with_ms(_self, curr, memory, curr_pos=None,
                                         memory_pos=None,
                                         num_obj_ptr_tokens=0,
                                         return_attn_stat=False):
                mt = getattr(self.model, '_ms_tokens', None)
                mp = getattr(self.model, '_ms_pos', None)
                if mt is not None and mp is not None:
                    memory = torch.cat([memory, mt], dim=0)
                    if memory_pos is not None:
                        memory_pos = torch.cat([memory_pos, mp], dim=0)
                return old_mem_attn_forward(
                    curr=curr, memory=memory, curr_pos=curr_pos,
                    memory_pos=memory_pos,
                    num_obj_ptr_tokens=num_obj_ptr_tokens,
                    return_attn_stat=return_attn_stat,
                )

            self_.memory_attention.forward = mem_attn_forward_with_ms
            try:
                result = self._orig_method(
                    frame_idx, is_init_cond_frame, current_vision_feats,
                    current_vision_pos_embeds, feat_sizes, output_dict,
                    num_frames, track_in_reverse, inference_state, obj_idx,
                )
            finally:
                self_.memory_attention.forward = old_mem_attn_forward
                self.model._ms_tokens = None
                self.model._ms_pos = None

            return result

        self.model._prepare_memory_conditioned_features = MethodType(
            patched, self.model
        )
        return self



    def clear_multi_scale_tokens(self):
        """Clear the stored multi-scale tokens from the model."""
        self.model._ms_tokens = None
        self.model._ms_pos = None

    def __exit__(self, exc_type, exc_val, exc_tb):
        from types import MethodType
        if self._had_attr:
            self.model._prepare_memory_conditioned_features = self._orig_attr
        else:
            self.model.__dict__.pop('_prepare_memory_conditioned_features', None)
        return False