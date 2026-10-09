"""Guards for the RDE-VOS memory update built from the official code."""

import math
import sys
from pathlib import Path
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "external" / "MedSAM2")]

from modeling.rde_memory import RDEMemoryUpdate  # noqa: E402
from model.modules import SAM, MemCrompress  # noqa: E402  (official, put on the path by rde_memory)


def test_rde_is_the_official_memcrompress_key_branch():
    torch.manual_seed(0)
    official = MemCrompress().eval()
    update = RDEMemoryUpdate(memory_channels=64, image_channels=256)
    assert isinstance(update.key_encoder, SAM)
    branch = {k: v for k, v in official.state_dict().items() if k.startswith(("key_encoder.", "compress_key."))}
    assert set(branch) == set(update.state_dict())
    update.load_state_dict(branch)

    mean, candidate = torch.randn(1, 64, 8, 8), torch.randn(1, 64, 8, 8)
    value = torch.zeros(1, 1, 512, 2, 8, 8)
    expected, _ = official(torch.stack([mean, candidate], dim=2), value)
    out = update.update(mean, None, candidate)
    assert torch.allclose(out["mean"], expected.squeeze(2), atol=1e-5)


def test_two_frames_compress_reads_prompt_twice_rde_and_last_memory_frame():
    update = RDEMemoryUpdate(memory_channels=4, image_channels=8, mode="two-frames-compress", mem_every=3)
    anchor, rde, recent = torch.zeros(1, 4, 2, 2), torch.ones(1, 4, 2, 2), torch.full((1, 4, 2, 2), 2.0)
    state = SimpleNamespace(anchor=anchor, mean=rde, recent=None, recent_frame_idx=None)
    assert [ago for _, ago in update.memory_slots(state, 2, 7)] == [0, 6]
    state.recent, state.recent_frame_idx = recent, 3
    slots = update.memory_slots(state, 5, 7)
    assert [ago for _, ago in slots] == [0, 0, 6, 2]
    assert slots[3][0] is recent
    assert [update.updates_at(t) for t in range(1, 7)] == [False, False, True, False, False, True]
    assert update.keeps_recent and not RDEMemoryUpdate(4, 8, mode="gt-compress").keeps_recent


def test_earlier_checkpoints_load_as_gt_compress_every_frame():
    legacy_names = {
        "aggregate.non_local.theta.weight": torch.zeros(32, 64, 1, 1, 1),
        "aggregate.aspp.branches.2.weight": torch.zeros(16, 64, 1, 3, 3),
        "squeeze.bias": torch.zeros(64),
    }
    config, state = RDEMemoryUpdate.upgrade({"initial_gain": 0.835}, legacy_names)
    assert config == {"mode": "gt-compress", "mem_every": 1, "repeat": 0}
    assert set(state) == {"key_encoder.non_local.theta.weight", "key_encoder.conv1.aspp3.atrous_conv.weight",
                          "compress_key.bias"}
    assert set(state) <= set(RDEMemoryUpdate(64, 256, **config).state_dict())


def test_official_objective_bootstraps_and_distils_toward_the_teacher():
    from training.memory_trainer import RDELoss

    def frame(score, mask_logit, feature):
        masks = torch.full((1, 1, 8, 8), mask_logit, requires_grad=True)
        return {"pred_masks_high_res": masks,
                "multistep_object_score_logits": [torch.tensor([[score]])], "pix_feat_with_mem": feature}

    masks = torch.zeros(3, 8, 8)
    masks[1:, 2:4, 2:4] = 1
    feature = torch.randn(1, 16, 4, 4)
    student = [frame(2.0, 0.0, feature) for _ in range(3)]
    loss, parts = RDELoss(start_warm=10, end_warm=20)(student, student, masks, step=0)
    assert math.isclose(parts["distill"], 0.0, abs_tol=1e-6)
    assert parts["bootstrap_p"] == 1.0  # plain CE before start_warm
    loss.backward()
    assert student[1]["pred_masks_high_res"].grad.abs().sum() > 0
    _, late = RDELoss(start_warm=10, end_warm=20, top_p=0.15)(student, student, masks, step=30)
    assert math.isclose(late["bootstrap_p"], 0.15, rel_tol=1e-6)
