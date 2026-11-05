import hashlib

import pytest
import torch
import torch.nn.functional as F

from modeling.soft_mask_aware_tome import (
    NativeSoftMaskAwareToMe,
    compress_memory_kv,
    occupancy_from_mask_logits,
    soft_mask_similarity_metric,
)


def test_occupancy_is_continuous_and_cell_averaged():
    logits = torch.zeros(1, 1, 1024, 1024)
    occupancy = occupancy_from_mask_logits(logits, (64, 64))
    assert occupancy.shape == (4096,)
    assert torch.allclose(occupancy, torch.full((4096,), 0.5))


def test_occupancy_pooling_matches_memory_grid_footprints():
    logits = torch.full((128, 64), -12.0)
    logits[:, :32] = 12.0
    occupancy = occupancy_from_mask_logits(logits, (8, 4)).reshape(8, 4)
    assert torch.all(occupancy[:, :2] > 0.999)
    assert torch.all(occupancy[:, 2:] < 0.001)


def test_occupancy_rejects_mask_that_does_not_tile_memory_grid():
    with pytest.raises(ValueError, match="tile the native memory grid"):
        occupancy_from_mask_logits(torch.zeros(1, 1, 255, 256), (64, 64))


def test_mask_similarity_is_soft_not_three_way_argmax():
    keys = torch.tensor([[[[1.0, 0.0], [1.0, 0.0], [1.0, 0.0]]]])
    occupancy = torch.tensor([[0.0, 0.5, 1.0]])
    metric = soft_mask_similarity_metric(keys, occupancy, role_weight=0.5)
    metric = F.normalize(metric, dim=-1)
    similarity = metric @ metric.transpose(-1, -2)
    assert similarity[0, 0, 1] > similarity[0, 0, 2]
    assert torch.allclose(similarity[0, 0, 1], torch.tensor(0.8535534), atol=1e-6)
    assert torch.allclose(similarity[0, 0, 2], torch.tensor(0.5), atol=1e-6)


def test_soft_and_generic_modes_match_budget_preserve_pointers_and_mass():
    torch.manual_seed(31)
    keys = torch.randn(1, 2, 18, 8)
    values = torch.randn(1, 2, 18, 8)
    occupancy = torch.linspace(0.0, 1.0, 16)[None]
    spans = [{"frame": 4, "t_pos": 2, "start": 0, "stop": 16}]
    outputs = {}

    for mode in ("soft_mask_aware", "generic"):
        out_k, out_v, bias, audit = compress_memory_kv(
            keys,
            values,
            occupancy,
            spans,
            mode=mode,
            pointers=2,
        )
        outputs[mode] = (out_k, out_v, bias, audit)
        assert out_k.shape == (1, 2, 10, 8)
        assert out_v.shape == out_k.shape
        assert torch.equal(out_k[..., -2:, :], keys[..., -2:, :])
        assert torch.equal(out_v[..., -2:, :], values[..., -2:, :])
        assert torch.allclose(bias.exp().sum(), torch.tensor(18.0))
        assert audit["spatial_key_mass_sum"] == 16.0
        assert audit["key_mass_sum"] == 18.0
        assert audit["frames"][0]["windows"][0]["merges"] == 8

    assert outputs["soft_mask_aware"][3]["spatial_keys_after"] == outputs["generic"][3]["spatial_keys_after"]


def test_invalid_occupancy_is_rejected():
    keys = torch.ones(1, 1, 16, 4)
    with pytest.raises(ValueError, match="Occupancy"):
        soft_mask_similarity_metric(keys, torch.full((1, 16), 1.5))


def test_memory_writer_masks_are_captured_with_their_source_frame():
    class FakePredictor:
        def _prepare_memory_conditioned_features(self):
            return None

        binarize_mask_from_pts_for_mem_enc = False

        def track_step(self, frame_idx, pred_masks_high_res, is_mask_from_pts=False):
            return self._encode_new_memory(
                pred_masks_high_res=pred_masks_high_res,
                is_mask_from_pts=is_mask_from_pts,
            )

        def _run_memory_encoder(self, frame_idx, high_res_masks):
            return self._encode_new_memory(pred_masks_high_res=high_res_masks)

        def _encode_new_memory(self, pred_masks_high_res, is_mask_from_pts=False):
            return torch.zeros(1, 4, 2, 2), None

    model = FakePredictor()
    observer = NativeSoftMaskAwareToMe(model)
    model.track_step = observer._track_step_with_write_frame
    model._run_memory_encoder = observer._run_memory_encoder_with_write_frame
    model._encode_new_memory = observer._capture_encode_new_memory_input

    predicted = torch.full((1, 1, 8, 8), 2.0)
    prompted = torch.full((1, 1, 8, 8), -2.0)
    model.track_step(frame_idx=7, pred_masks_high_res=predicted)
    model._run_memory_encoder(frame_idx=3, high_res_masks=prompted)

    assert set(observer.mask_cache) == {3, 7}
    assert torch.all(observer.mask_cache[7][0] > 0.88)
    assert torch.all(observer.mask_cache[3][0] < 0.12)
    for frame, logits in ((7, predicted), (3, prompted)):
        metadata = observer.mask_cache[frame][1]
        source = logits[0, 0].contiguous()
        expected_hash = hashlib.sha256(source.view(torch.uint8).numpy().tobytes()).hexdigest()
        assert metadata["frame"] == frame
        assert metadata["mask_source_field"] == "pred_masks_high_res"
        assert metadata["capture_stage"] == "input_to_sam2__encode_new_memory"
        assert metadata["memory_write_mask_sha256"] == expected_hash
        assert metadata["memory_write_mask_shape"] == [8, 8]
    assert observer.active_write_frame is None
    model.binarize_mask_from_pts_for_mem_enc = True
    with pytest.raises(RuntimeError, match="binarization is incompatible"):
        model.track_step(frame_idx=9, pred_masks_high_res=predicted, is_mask_from_pts=True)
