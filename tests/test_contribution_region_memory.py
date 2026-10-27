from types import SimpleNamespace

import numpy as np
import pytest
import torch

from modeling.contribution_region_memory import ContributionRegionMemory, contribution_cells
from modeling.matched_region_memory import MatchedRegionMemory, random_cells
from scripts.run_matched_region_experiment import load_reference, verified_completed


@pytest.mark.parametrize("label,count", [(0, 4096), (1, 4096), (2, 0)])
def test_contribution_selection_by_region(label, count):
    keep = contribution_cells(torch.full((64, 64), label))
    assert keep.dtype == torch.bool and keep.shape == (4096,)
    assert int(keep.sum()) == count


def test_mixed_selection_preserves_far_background_without_dilation():
    labels = torch.zeros(64, 64, dtype=torch.long)
    labels[20:30, 20:30] = 1
    labels[20, 20:30] = 2
    keep = contribution_cells(labels).reshape(64, 64)
    assert keep[0, 0] and keep[25, 25]
    assert torch.equal(keep, labels != 2)


def test_bad_shape_or_labels_rejected():
    for labels in (torch.zeros(32, 32), torch.full((64, 64), 3)):
        with pytest.raises(ValueError):
            contribution_cells(labels)


def test_actual_stored_prediction_controls_selector_and_trace():
    model = SimpleNamespace(num_maskmem=7, memory_temporal_stride_for_eval=1)
    controller = ContributionRegionMemory(model, 0, "seq2")
    features = torch.randn(1, 64, 64, 64)
    mask = torch.full((1, 1, 256, 256), -10.)
    stored = {"maskmem_features": features, "maskmem_pos_enc": [torch.zeros_like(features)],
              "pred_masks": mask}
    state = {"output_dict_per_obj": {0: {"cond_frame_outputs": {0: stored}, "non_cond_frame_outputs": {}}}}
    record = controller.capture(state, 0)
    assert record["policy"] == "contribution_v2"
    assert record["retained_tokens"] == 4096  # absent frame retains background, unlike v1
    assert controller.bank[0]["features"].data_ptr() == features.data_ptr()
    assert torch.equal(stored["pred_masks"], mask)
    assert np.unpackbits(np.frombuffer(bytes.fromhex(record["keep_bits_hex"]), dtype=np.uint8)).sum() == 4096


def test_random_control_keeps_reference_count_not_new_geometry():
    model = SimpleNamespace(num_maskmem=7, memory_temporal_stride_for_eval=1)
    reference = {0: {"frame": 0, "retained_tokens": 100, "allocation": None}}
    controller = ContributionRegionMemory(model, 0, "seq2", reference=reference, seed=17)
    features = torch.zeros(1, 64, 64, 64)
    stored = {"maskmem_features": features, "maskmem_pos_enc": [features.clone()],
              "pred_masks": torch.full((1, 1, 256, 256), -10.)}
    state = {"output_dict_per_obj": {0: {"cond_frame_outputs": {0: stored}, "non_cond_frame_outputs": {}}}}
    record = controller.capture(state, 0)
    assert record["condition"] == "C3" and record["retained_tokens"] == 100
    assert torch.equal(controller.bank[0]["keep"], random_cells(100, "seq2", 0, 17))


def test_dense_frames_pointers_and_all_layers_reuse_verified_reader():
    assert ContributionRegionMemory._assemble is MatchedRegionMemory._assemble
    assert ContributionRegionMemory.__enter__ is MatchedRegionMemory.__enter__


def test_native_recent_six_window_excludes_old_nonprompt_and_current_frames():
    model = SimpleNamespace(num_maskmem=7, memory_temporal_stride_for_eval=1)
    controller = ContributionRegionMemory(model, 0, "seq2")
    assert controller.mode == "region_native" and controller.max_frames == 7
    controller.frame = 20
    frames = [0, 14, 15, 16, 17, 18, 19]
    controller.bank = {i: {"features": torch.full((1, 64, 64, 64), float(i)),
                           "pos": torch.zeros(1, 64, 64, 64),
                           "keep": torch.ones(4096, dtype=torch.bool)} for i in range(20)}
    memory = torch.cat([controller.bank[i]["features"].flatten(2).transpose(1, 2)
                        for i in frames] + [torch.zeros(1, 4, 64)], dim=1)
    for layer in range(4):
        controller._before(layer, SimpleNamespace(training=False), (),
                           {"v": memory, "q": torch.zeros(1, 4, 256), "num_k_exclude_rope": 4})
        assert controller.layers[-1]["frames"] == frames
        assert controller.layers[-1]["selected_spatial_keys"] == 28672


def test_v2_does_not_reassemble_native_memory():
    model = SimpleNamespace(num_maskmem=7, memory_temporal_stride_for_eval=1)
    controller = ContributionRegionMemory(model, 0, "seq2")
    memory = torch.randn(4096, 1, 64)
    kwargs = {"memory": memory, "memory_pos": torch.zeros_like(memory)}
    before = memory.clone()
    assert controller._assemble(None, (), kwargs) is None
    assert torch.equal(memory, before)


def test_v2_random_control_rejects_a_legacy_reference(tmp_path):
    import json
    path = tmp_path / "legacy.jsonl"
    bits = np.packbits(np.zeros(4096, dtype=bool)).tobytes().hex()
    path.write_text(json.dumps({"frame": 0, "trace": {"frame": 0, "condition": "C2",
                    "keep_bits_hex": bits, "retained_tokens": 0}}))
    with pytest.raises(ValueError, match="policy differs"):
        load_reference(path, "contribution_v2")


@pytest.mark.parametrize("boundary_count", [0, 123, 4096])
def test_actual_layer_masks_preserve_dense_prompt_latest_and_pointers(boundary_count):
    model = SimpleNamespace(num_maskmem=7, memory_temporal_stride_for_eval=1)
    controller = ContributionRegionMemory(model, 0, "seq2")
    controller.frame = 3
    controller.assembled_frames = [0, 1, 2]
    labels = torch.zeros(64, 64, dtype=torch.long)
    labels.flatten()[:boundary_count] = 2
    controller.bank = {i: {"features": torch.full((1, 64, 64, 64), float(i)),
                           "pos": torch.zeros(1, 64, 64, 64),
                           "keep": contribution_cells(labels)} for i in range(3)}
    memory = torch.cat([controller.bank[i]["features"].flatten(2).transpose(1, 2)
                        for i in range(3)] + [torch.zeros(1, 4, 64)], dim=1)
    attention = SimpleNamespace(training=False)
    for layer in range(4):
        controller._before(layer, attention, (), {"v": memory, "q": torch.zeros(1, 4, 256),
                                                "num_k_exclude_rope": 4})
        trace = controller.layers[-1]
        assert trace["per_frame_spatial_keys"] == [4096, 4096-boundary_count, 4096]
        assert trace["pointer_keys"] == 4
        assert trace["selected_spatial_keys"] == 12288-boundary_count
        assert torch.equal(attention._sparse_mask, contribution_cells(labels))


@pytest.fixture
def completed_resume_output(tmp_path):
    import json
    import zipfile
    from scripts.false_alarm_statistics import propagation_summary
    folder = tmp_path / "C2"
    folder.mkdir()
    row = {"frame": 1, "present": True, "prediction_sha256": "prediction",
           "scores": {"dice": .8, "iou": .7}}
    summary = {"sequence": "seq2", "condition": "C2", "seed": None,
               **propagation_summary([row])}
    allocation = {"frames": [0], "per_frame_spatial_keys": [4096], "selected_spatial_keys": 4096, "pointer_keys": 4}
    receipts = [{"frame": i, "prediction_sha256": "prediction", "low_res_sha256": "low",
                 "bank_sha256": "bank", "cache_sha256": "cache", "allocation": allocation,
                 "retained_tokens": 4096, "keep_bits_hex": "ff"*512,
                 "trace": {"allocation": allocation,
                           "layers": [{"layer": k, **allocation} for k in range(4)]}} for i in range(2)]
    (folder/"summary.json").write_text(json.dumps(summary))
    (folder/"frames.jsonl").write_text(json.dumps(row))
    (folder/"integrity.jsonl").write_text("\n".join(map(json.dumps, receipts)))
    (folder/"parameter_integrity_C2_C3.json").write_text(json.dumps({"unchanged": True, "changed_names": []}))
    archive = tmp_path/"completed.zip"

    def pack():
        with zipfile.ZipFile(archive, "w") as packed:
            for path in folder.iterdir():
                packed.write(path, path.name)
    pack()
    return folder, archive, summary, pack


def test_resume_accepts_only_verified_complete_archive(completed_resume_output):
    folder, archive, summary, _ = completed_resume_output
    result, integrity = verified_completed(folder, archive, "seq2", "C2", None, [0, 1], 0, summary)
    assert result == summary and sorted(integrity) == [0, 1]


def test_resume_rejects_changed_file_even_if_summary_exists(completed_resume_output):
    folder, archive, summary, _ = completed_resume_output
    (folder/"frames.jsonl").write_text("corruption")
    with pytest.raises(ValueError, match="output differs"):
        verified_completed(folder, archive, "seq2", "C2", None, [0, 1], 0, summary)


def test_resume_rejects_missing_frame_and_layer_receipts(completed_resume_output):
    import json
    folder, archive, summary, pack = completed_resume_output
    path = folder/"integrity.jsonl"
    rows = list(map(json.loads, path.read_text().splitlines()))
    rows[1]["trace"]["layers"].pop()
    path.write_text("\n".join(map(json.dumps, rows)))
    pack()
    with pytest.raises(ValueError, match="four-layer"):
        verified_completed(folder, archive, "seq2", "C2", None, [0, 1], 0, summary)
    path.write_text(json.dumps(rows[0]))
    pack()
    with pytest.raises(ValueError, match="coverage"):
        verified_completed(folder, archive, "seq2", "C2", None, [0, 1], 0, summary)
