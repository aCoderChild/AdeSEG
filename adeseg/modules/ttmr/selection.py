"""TTMR reference policy: causal, target-aware, temporally balanced token ranking.

This is an AUDITABLE OFFLINE CANDIDATE SELECTOR, not an installed Lean-SAM2
memory hook. Native TAMP/TCIM/TARR and their reported results remain unchanged.
Only previous-frame memory, predicted target anchors and native usage scores
are inputs; GT masks and future frames are forbidden.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
import torch
import torch.nn.functional as F


@dataclass(frozen=True)
class TTMRConfig:
    budget: int
    anchor_weight: float = 0.5
    redundancy_weight: float = 0.15
    recency_weight: float = 0.05
    multiscale_weight: float = 0.5
    time_constant: float = 5.0
    minimum_per_frame: int = 1
    maximum_per_frame: int | None = None

    def __post_init__(self):
        if (type(self.budget) is not int or not 1 <= self.budget <= 512
                or type(self.minimum_per_frame) is not int
                or self.minimum_per_frame < 1):
            raise ValueError("TTMR CPU-reference budget must be 1..512")
        if self.maximum_per_frame is not None and (
                type(self.maximum_per_frame) is not int
                or self.maximum_per_frame < self.minimum_per_frame):
            raise ValueError("TTMR max per frame cannot be below min")
        for name in ("anchor_weight","redundancy_weight","recency_weight","multiscale_weight"):
            if not 0 <= getattr(self,name) <= 1:
                raise ValueError(f"TTMR {name} must be between zero and one")
        if not math.isfinite(self.time_constant) or self.time_constant <= 0:
            raise ValueError("TTMR time constant must be positive")


@torch.no_grad()
def rank_temporal_memory(features, anchor_tokens, usage_scores, source_frames,
                         *, current_frame, config: TTMRConfig,
                         multiscale_scores=None):
    """Select fixed-budget, diverse historical features with temporal coverage.

    Reserve at least one high-value token for a spread of past frames, then
    greedily fill with target/usage scores minus cosine redundancy to selected
    tokens. No variable budget, no current/future tokens, and no hidden GT.
    Returns indices into the ORIGINAL flattened memory candidates.
    """
    if not isinstance(config, TTMRConfig):
        raise TypeError("Pass a predeclared TTMRConfig")
    if (not isinstance(features,torch.Tensor) or features.ndim != 2
            or features.shape[1] < 2 or not torch.is_floating_point(features)
            or not torch.isfinite(features).all()):
        raise ValueError("Expected finite memory vectors [N,C]")
    count, channels=features.shape
    if count==0 or count>4096 or count<config.budget:
        raise ValueError("CPU prototype expects 1..4096 candidates and budget <= N")
    if (not isinstance(anchor_tokens,torch.Tensor)
            or anchor_tokens.ndim != 2 or anchor_tokens.shape[1] != channels
            or anchor_tokens.shape[0] < 1
            or not torch.isfinite(anchor_tokens).all()):
        raise ValueError("Target anchor vectors do not match memory channels")
    if (not isinstance(usage_scores,torch.Tensor)
            or tuple(usage_scores.shape)!=(count,)
            or not torch.isfinite(usage_scores).all()
            or (usage_scores < 0).any() or (usage_scores > 1).any()):
        raise ValueError("Native usage must be pre-normalized to [0,1]")
    if (not isinstance(source_frames,torch.Tensor)
            or tuple(source_frames.shape)!=(count,)
            or source_frames.dtype not in (torch.int16,torch.int32,torch.int64)
            or type(current_frame) is not int
            or current_frame<=0
            or (source_frames<0).any()
            or (source_frames>=current_frame).any()):
        raise ValueError("TTMR uses only historical frame tokens")
    # Multiscale context changes scores, never the number or shape of stored tokens.
    if multiscale_scores is not None and (
            not isinstance(multiscale_scores,torch.Tensor)
            or tuple(multiscale_scores.shape)!=(count,)
            or not torch.isfinite(multiscale_scores).all()
            or (multiscale_scores < -1).any()
            or (multiscale_scores > 1).any()):
        raise ValueError("Multiscale anchor scores must be finite cosine values [-1,1]")
    # Reference path: scores and diversity are evaluated on CPU FP32.
    vectors=F.normalize(features.detach().cpu().float(),dim=-1,eps=1e-6)
    anchors=F.normalize(anchor_tokens.detach().cpu().float(),dim=-1,eps=1e-6)
    frame_ids=source_frames.detach().cpu().long()
    anchor=(vectors @ anchors.T).max(dim=1).values
    if multiscale_scores is not None:
        multiscale=multiscale_scores.detach().cpu().float()
        anchor=(1-config.multiscale_weight)*anchor+config.multiscale_weight*multiscale
    use=usage_scores.detach().cpu().float()
    freshness=torch.exp(-(current_frame-frame_ids).float()/config.time_constant)
    base=((1-config.anchor_weight)*use
          +config.anchor_weight*anchor
          +config.recency_weight*freshness)
    unique=sorted(set(frame_ids.tolist()))
    # Round-robin across temporally spread frames, including oldest/newest;
    # no claim that this is better than Lean TCIM's source selection.
    if len(unique)*config.minimum_per_frame > config.budget:
        # Evenly spread the frame-group reservation rather than silently
        # dropping the oldest historical context.
        groups=[unique[round(i*(len(unique)-1)/(config.budget-1))]
                for i in range(config.budget)] if config.budget>1 else [unique[-1]]
        minimum=1
    else:
        groups=unique
        minimum=config.minimum_per_frame
    scores=base.clone()
    selected=[]
    counts={frame:0 for frame in unique}

    def best(eligible):
        candidates=eligible.nonzero(as_tuple=False).reshape(-1)
        if not len(candidates):
            raise ValueError("Retention constraints leave no feasible candidate")
        if selected and config.redundancy_weight:
            similarity=(vectors[candidates] @ vectors[selected].T).max(dim=1).values
            adjusted=scores[candidates]-config.redundancy_weight*similarity.clamp(min=0)
        else:
            adjusted=scores[candidates]
        # Linear-time deterministic selection; ties favor the smallest index.
        ordered = zip(candidates.tolist(), adjusted.tolist())
        return min(ordered, key=lambda item: (-item[1], item[0]))[0]

    used=torch.zeros(count,dtype=torch.bool)
    for frame in groups:
        for _ in range(minimum):
            eligible=(~used)&(frame_ids==frame)
            if not eligible.any():
                raise ValueError("Not enough candidates for per-frame reservation")
            idx=best(eligible)
            used[idx]=True
            selected.append(idx)
            counts[frame]+=1
    while len(selected)<config.budget:
        eligible=~used
        if config.maximum_per_frame is not None:
            for frame,amount in counts.items():
                if amount>=config.maximum_per_frame:
                    eligible &= frame_ids!=frame
        if not eligible.any():
            raise ValueError("Maximum-per-frame cap prevents fixed token budget")
        idx=best(eligible)
        used[idx]=True
        selected.append(idx)
        counts[int(frame_ids[idx])]+=1
    result=torch.tensor(selected,dtype=torch.long)
    if result.numel()!=config.budget or result.unique().numel()!=result.numel():
        raise RuntimeError("TTMR failed fixed-budget uniqueness invariant")
    return {
        "indices":result,
        "selected_scores":base[result].clone(),
        "retained_per_frame":{str(k):v for k,v in sorted(counts.items()) if v},
        "selected_token_count":int(result.numel()),
        "candidate_token_count":count,
        "active_frame_groups":len([v for v in counts.values() if v]),
        "status":"offline_shadow_selection_only",
    }
