"""CPU tests for matched with/without-module Dice ablations."""
from pathlib import Path
import json

import numpy as np
import pytest
from PIL import Image

from datasets.video import ManifestFrame, load_manifest
from scripts.run_lean_memory_fusion_ablation import (
    file_sha256, load_frozen_prompts, verify_learned_causal_effectiveness,
    verify_native_tamp_dump,
)
from scripts.score_lean_memory_ablations import score_runs, FORMAT
from scripts.compare_native_tamp_indices import compare_runs


def png(path, array):
    path.parent.mkdir(parents=True, exist_ok=True)
    Image.fromarray((np.asarray(array, dtype=np.uint8) * 255)).save(path)


def benchmark(tmp_path):
    images = tmp_path / "frames"
    labels = tmp_path / "labels"
    images.mkdir()
    labels.mkdir()
    manifest = tmp_path / "manifest.jsonl"
    rows, videos = [], {}
    for seq in ("seq2", "seq3"):
        frame_list = []
        for idx in range(3):
            path = images / f"{seq}_{idx}.jpg"
            Image.fromarray(np.zeros((4, 4, 3), dtype=np.uint8)).save(path)
            label = labels / f"{seq}_{idx}.png"
            truth = np.zeros((4, 4), np.uint8)
            if idx == 1:
                truth[0:2, 0:2] = 1
            # Indexed GT PNGs use label IDs 0/1, unlike prediction PNGs.
            Image.fromarray(truth).save(label)
            frame_list.append(ManifestFrame(seq, idx, path, label))
            rows.append(dict(split="dev", video_id=seq, frame_index=idx,
                             image=str(path), mask=str(label)))
        videos[seq] = frame_list
    manifest.write_text("".join(json.dumps(row)+"\n" for row in rows))
    image_manifest = tmp_path / "images_only.jsonl"
    image_manifest.write_text("".join(
        json.dumps({k:v for k,v in row.items() if k != "mask"}) + "\n"
        for row in rows
    ))
    prompt_manifest = tmp_path / "prompts.jsonl"
    prompts = []
    for seq, frames in videos.items():
        path = tmp_path / "prompt" / f"{seq}.png"
        png(path, np.zeros((4, 4), dtype=np.uint8))
        prompts.append({
            "video_id": seq, "frame_index": 0, "mask_path": str(path),
            "image_sha256": file_sha256(frames[0].image_path),
            "mask_sha256": file_sha256(path),
            "prompt_origin": "automatic_mask",
        })
    prompt_manifest.write_text("".join(json.dumps(p)+"\n" for p in prompts))
    return manifest, prompt_manifest, videos


def make_run(tmp_path, manifest, prompt_path, videos, name, arm, *, improved=False):
    root = tmp_path / name
    root.mkdir()
    prompts = load_frozen_prompts(prompt_path, videos)
    cases = {}
    for seq, frames in videos.items():
        records = []
        for frame in frames[1:]:
            target = np.zeros((4, 4), dtype=np.uint8)
            if frame.frame_index == 1 and improved:
                target[0:2, 0:2] = 1
            if frame.frame_index == 2 and not improved:
                target[0:2, 0:2] = 1
            p = root / "predictions" / seq / f"{frame.frame_index:06d}.png"
            png(p, target)
            records.append({
                "frame_index": frame.frame_index,
                "image_sha256": file_sha256(frame.image_path),
                "logits_sha256": ("f" if improved else "d")*64,
                "prediction_file": p.name,
                "prediction_sha256": file_sha256(p),
            })
        prompt = prompts[seq]
        cases[seq] = {
            "status": "completed", "prompt_frame_index": 0,
            "prompt_mask_sha256": prompt["mask_sha256"],
            "prompt_image_sha256": prompt["image_sha256"],
            "prompt_origin": "automatic_mask", "predictions": records,
            "resident_memory": {
                "resident_frame_entries": 2, "resident_spatial_tokens": 4,
                "compressed_frame_entries": 1, "protected_frame_entries": 1,
                "insurance_entries": 0,
                "note": "Resident snapshot only; does not include transient attention tensors or object pointers",
            },
            "per_frame_resident_memory": [
                {"local_frame_index": i, "source_frame_index": frames[i].frame_index,
                 "memory": {
                     "resident_frame_entries": 2, "resident_spatial_tokens": 4,
                     "compressed_frame_entries": 1, "protected_frame_entries": 1,
                     "insurance_entries": 0,
                     "note": "Resident snapshot only; does not include transient attention tensors or object pointers",
                 }}
                for i in range(3)
            ],
        }
    receipt = {
        "format": FORMAT, "track": "lean_sam2_hiera_bplus",
        "arm": arm, "split": "dev", "label_id": 1,
        "manifest_sha256": file_sha256(manifest.with_name("images_only.jsonl")),
        "prompt_manifest_sha256": file_sha256(prompt_path),
        "backbone_sha256": "a"*64,
        "runtime_signature": {
            "adeseg_head": "e"*40, "runner_sha256": "a"*64,
            "fusion_source_sha256": "b"*64,
            "torch_version": "2.10.0", "python_version": "3.11.0",
            "device": "cpu", "precision": "native build force_fp32; no autocast",
            "lean_config_sha256": "c"*64, "lvos_config_sha256": "d"*64,
            "sam_config_sha256": "e"*64, "bypass_sha256": "f"*64,
        },
        "lean_source_revision": "8d1cf2618519d291905d22fe859b8302476d7f50",
        "sequences": cases,
    }
    (root / "receipt.json").write_text(json.dumps(receipt))
    return root


def test_full_paired_dice_and_empty_false_alarms(tmp_path):
    manifest, prompts, videos = benchmark(tmp_path)
    base = make_run(tmp_path, manifest, prompts, videos, "native", "native")
    identity = make_run(tmp_path, manifest, prompts, videos, "identity", "identity")
    learned = make_run(tmp_path, manifest, prompts, videos, "fusion", "fusion", improved=True)
    report = score_runs(
        {"native": base, "identity": identity, "fusion": learned},
        manifest, tmp_path/"scores", bootstrap=200,
    )
    a = report["summary"]
    assert a["native"]["sequence_macro_present_dice"] == 0
    assert a["fusion"]["sequence_macro_present_dice"] == 1
    assert a["fusion"]["mean_paired_delta_vs_baseline"] == 1
    assert a["identity"]["sequence_macro_present_dice"] == 0
    assert a["native"]["empty_false_alarms"] == 2
    assert a["fusion"]["empty_false_alarms"] == 0
    assert (tmp_path/"scores"/"report.md").is_file()
    assert (tmp_path/"scores"/"per_sequence.csv").is_file()


def test_missing_automatic_initialization_is_reported_not_hidden(tmp_path):
    manifest, prompts, videos = benchmark(tmp_path)
    native = make_run(tmp_path, manifest, prompts, videos, "native", "native")
    fusion = make_run(tmp_path, manifest, prompts, videos, "fusion", "fusion",
                      improved=True)
    for root in (native, fusion):
        receipt_file = root / "receipt.json"
        receipt = json.loads(receipt_file.read_text())
        receipt["sequences"]["seq3"] = {
            "status": "no_initialization", "predictions": [],
        }
        receipt_file.write_text(json.dumps(receipt))
    report = score_runs(
        {"native": native, "fusion": fusion},
        manifest, tmp_path / "coverage_scores", bootstrap=200)
    row = report["summary"]["native"]
    assert row["source_videos"] == 2
    assert row["videos_with_initialization"] == 1
    assert row["videos_without_initialization"] == 1
    assert row["gt_present_frames_without_initialization"] == 1
    assert row["gt_empty_frames_without_initialization"] == 2
    assert row["post_prompt_scored_frames"] == 2
    assert row["fraction_source_frames_scored_post_prompt"] == pytest.approx(2/6)
    assert report["unpromptable_video_coverage"]["seq3"]["gt_present_frames_without_prediction"] == 1
    assert "Videos initialized: 1/2" in (
        tmp_path / "coverage_scores" / "report.md").read_text()


def test_mismatched_prompt_protocol_and_frame_coverage_rejected(tmp_path):
    manifest, prompts, videos = benchmark(tmp_path)
    base = make_run(tmp_path, manifest, prompts, videos, "native", "native")
    fusion = make_run(tmp_path, manifest, prompts, videos, "fusion", "fusion", improved=True)
    rfile = fusion/"receipt.json"
    receipt = json.loads(rfile.read_text())
    receipt["sequences"]["seq2"]["prompt_mask_sha256"] = "0"*64
    rfile.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Prompt mismatch"):
        score_runs({"native": base, "fusion": fusion}, manifest, tmp_path/"scores")
    receipt["sequences"]["seq2"]["prompt_mask_sha256"] = json.loads((base/"receipt.json").read_text())["sequences"]["seq2"]["prompt_mask_sha256"]
    receipt["sequences"]["seq2"]["predictions"] = receipt["sequences"]["seq2"]["predictions"][:1]
    rfile.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="coverage mismatch"):
        score_runs({"native": base, "fusion": fusion}, manifest, tmp_path/"scores")


def test_modified_prediction_and_wrong_protocol_rejected(tmp_path):
    manifest, prompts, videos = benchmark(tmp_path)
    base = make_run(tmp_path, manifest, prompts, videos, "native", "native")
    fusion = make_run(tmp_path, manifest, prompts, videos, "fusion", "fusion", improved=True)
    png(fusion/"predictions"/"seq2"/"000001.png", np.ones((4,4),dtype=np.uint8))
    with pytest.raises(ValueError, match="checksum"):
        score_runs({"native": base, "fusion": fusion}, manifest, tmp_path/"scores")
    b = json.loads((base/"receipt.json").read_text())
    b["backbone_sha256"] = "b"*64
    (base/"receipt.json").write_text(json.dumps(b))
    with pytest.raises(ValueError, match="unmatched backbone"):
        score_runs({"native": base, "fusion": fusion}, manifest, tmp_path/"scores")


def test_identity_logits_divergence_is_not_mask_parity(tmp_path):
    manifest, prompts, videos = benchmark(tmp_path)
    base = make_run(tmp_path, manifest, prompts, videos, "native", "native")
    identity = make_run(tmp_path, manifest, prompts, videos, "identity", "identity")
    rfile = identity / "receipt.json"
    receipt = json.loads(rfile.read_text())
    receipt["sequences"]["seq2"]["predictions"][0]["logits_sha256"] = "0"*64
    rfile.write_text(json.dumps(receipt))
    with pytest.raises(RuntimeError, match="identity diverged"):
        score_runs({"native": base, "identity": identity}, manifest,
                   tmp_path / "report")


def test_resident_budget_changes_are_reported_not_hidden(tmp_path):
    manifest, prompts, videos = benchmark(tmp_path)
    base = make_run(tmp_path, manifest, prompts, videos, "native", "native")
    fusion = make_run(tmp_path, manifest, prompts, videos, "fusion", "fusion", improved=True)
    path = fusion / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["sequences"]["seq2"]["resident_memory"]["resident_spatial_tokens"] = 6
    receipt["sequences"]["seq2"]["per_frame_resident_memory"][-1]["memory"]["resident_spatial_tokens"] = 6
    path.write_text(json.dumps(receipt))
    report = score_runs({"native": base, "fusion": fusion}, manifest,
                        tmp_path / "report")
    assert not report["summary"]["fusion"]["resident_spatial_and_insurance_matched"]
    assert report["summary"]["fusion"]["resident_spatial_token_deltas"]["seq2"] == 2
    assert not report["summary"]["fusion"]["all_frame_resident_budgets_matched"]


def test_runtime_and_gt_separation(tmp_path):
    manifest, prompts, videos = benchmark(tmp_path)
    base = make_run(tmp_path, manifest, prompts, videos, "native", "native")
    fusion = make_run(tmp_path, manifest, prompts, videos, "fusion", "fusion", improved=True)
    path = fusion / "receipt.json"
    receipt = json.loads(path.read_text())
    receipt["runtime_signature"]["device"] = "cuda"
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="unmatched backbone"):
        score_runs({"native": base, "fusion": fusion}, manifest, tmp_path/"scores")
    receipt["runtime_signature"]["device"] = "cpu"
    del receipt["sequences"]["seq2"]["predictions"][0]["logits_sha256"]
    path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="logit checksum"):
        score_runs({"native": base, "fusion": fusion}, manifest, tmp_path/"scores")
    receipt["sequences"]["seq2"]["predictions"][0]["logits_sha256"] = "f"*64
    path.write_text(json.dumps(receipt))
    result = score_runs({"native": base, "fusion": fusion}, manifest, tmp_path/"scores")
    assert result["provenance"]["evaluation_manifest_sha256"] == file_sha256(manifest)
    original_gt_digest = result["provenance"]["evaluation_label_files_sha256"]
    # The independent scorer may work from changed labels, but changes must
    # always produce a different label-byte provenance digest.
    Image.fromarray(np.zeros((4, 4), dtype=np.uint8)).save(
        videos["seq2"][1].mask_path)
    changed = score_runs({"native": base, "fusion": fusion}, manifest,
                         tmp_path / "rescore_changed_gt", bootstrap=200)
    assert changed["provenance"]["evaluation_label_files_sha256"] != original_gt_digest
    assert all("annotation_sha256" not in p
               for seq in result["per_video"] for p in
               json.loads((base/"receipt.json").read_text())["sequences"][seq]["predictions"])


def test_prompt_authenticates_images_and_rejects_oracle_path(tmp_path):
    manifest, prompts_path, videos = benchmark(tmp_path)
    assert set(load_frozen_prompts(prompts_path, videos)) == {"seq2", "seq3"}
    rows = [json.loads(x) for x in prompts_path.read_text().splitlines()]
    rows[0]["mask_path"] = str(videos["seq2"][0].mask_path)
    rows[0]["mask_sha256"] = file_sha256(videos["seq2"][0].mask_path)
    prompts_path.write_text("".join(json.dumps(p)+"\n" for p in rows))
    with pytest.raises(ValueError, match="GT-path"):
        load_frozen_prompts(prompts_path, videos)


def test_image_only_inference_manifest_never_references_ground_truth(tmp_path):
    manifest, prompts_path, videos = benchmark(tmp_path)
    image_manifest = manifest.with_name("images_only.jsonl")
    images_only = load_manifest(
        image_manifest, "dev", require_masks=False,
        require_contiguous=True, require_jpeg=True, safe_output_ids=True,
    )
    assert all(f.mask_path is None for seq in images_only.values() for f in seq)
    assert set(load_frozen_prompts(prompts_path, images_only)) == {"seq2", "seq3"}
    with pytest.raises(ValueError, match="image-only inference manifest"):
        load_manifest(manifest, "dev", require_masks=False)


def test_intermediate_memory_budget_variation_is_visible(tmp_path):
    manifest, prompts, videos = benchmark(tmp_path)
    base = make_run(tmp_path, manifest, prompts, videos, "native", "native")
    fusion = make_run(tmp_path, manifest, prompts, videos, "fusion", "fusion", improved=True)
    path = fusion / "receipt.json"
    data = json.loads(path.read_text())
    # Same final memory, different intermediate memory budgets.
    data["sequences"]["seq2"]["per_frame_resident_memory"][1]["memory"]["insurance_entries"] = 1
    path.write_text(json.dumps(data))
    report = score_runs({"native": base, "fusion": fusion}, manifest,
                        tmp_path / "mid_budget_report")
    assert report["summary"]["fusion"]["resident_spatial_and_insurance_matched"]
    assert not report["summary"]["fusion"]["all_frame_resident_budgets_matched"]


def test_paired_native_tamp_overlap_uses_exact_matched_events(tmp_path):
    manifest, prompts, videos = benchmark(tmp_path)
    native = make_run(tmp_path, manifest, prompts, videos, "native", "native")
    fusion = make_run(tmp_path, manifest, prompts, videos, "fusion", "fusion")
    for root, indices in ((native, [0, 1]), (fusion, [1, 2])):
        receipt_file = root / "receipt.json"
        receipt = json.loads(receipt_file.read_text())
        receipt["dump_tamp_indices"] = True
        for video in videos:
            path = (root / "predictions" / video / "tamp_indices"
                    / "ours_retained_tokens.jsonl")
            path.parent.mkdir(parents=True)
            path.write_text(json.dumps({
                "event": "ours_retained_latest_memory_tokens",
                "frame_idx": 2, "memory_frame_idx": 1, "object_id": 1,
                "num_total_tokens": 4, "num_retained_tokens": 2,
                "retained_token_indices": indices,
            }) + "\n", encoding="utf8")
            receipt["sequences"][video]["tamp_index_dump"] = {
                "records": 1, "sha256": file_sha256(path),
            }
        receipt_file.write_text(json.dumps(receipt))
    result = compare_runs(native, fusion, tmp_path / "tamp_compare")
    assert result["matched_events"] == 2
    assert result["mean_jaccard"] == pytest.approx(1/3)
    assert result["identical_selection_fraction"] == 0
    assert (tmp_path / "tamp_compare" / "tamp_selection_overlap.json").is_file()


def test_tamp_comparison_requires_nonempty_and_complete_paired_events(tmp_path):
    manifest, prompts, videos = benchmark(tmp_path)
    native = make_run(tmp_path, manifest, prompts, videos, "native", "native")
    fusion = make_run(tmp_path, manifest, prompts, videos, "fusion", "fusion")
    for root in (native, fusion):
        receipt_path = root / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        receipt["dump_tamp_indices"] = True
        for video in videos:
            receipt["sequences"][video]["tamp_index_dump"] = {
                "records": 0, "sha256": None,
            }
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="No paired TAMP retention events"):
        compare_runs(native, fusion, tmp_path / "empty_overlap")

    # Native and fusion each emit one event per video, but one fusion event
    # identifies a different target memory frame; that is NOT a paired event.
    for root, event_shift in ((native, 0), (fusion, 1)):
        receipt_path = root / "receipt.json"
        receipt = json.loads(receipt_path.read_text())
        for video in videos:
            path = root / "predictions" / video / "tamp_indices" / "ours_retained_tokens.jsonl"
            path.parent.mkdir(parents=True, exist_ok=True)
            shift = event_shift if video == "seq2" else 0
            path.write_text(json.dumps({
                "event": "ours_retained_latest_memory_tokens",
                "frame_idx": 2, "memory_frame_idx": 1 + shift,
                "object_id": 1, "num_total_tokens": 4,
                "num_retained_tokens": 2, "retained_token_indices": [0, 1],
            }) + "\n")
            receipt["sequences"][video]["tamp_index_dump"] = {
                "records": 1, "sha256": file_sha256(path),
            }
        receipt_path.write_text(json.dumps(receipt))
    with pytest.raises(ValueError, match="Unmatched native TAMP selection events"):
        compare_runs(native, fusion, tmp_path / "incomplete_overlap")
    partial = compare_runs(
        native, fusion, tmp_path / "partial_overlap",
        allow_incomplete_events=True,
    )
    assert partial["matched_events"] == 1
    assert partial["complete_event_coverage"] is False
    assert partial["partial_coverage_explicitly_allowed"] is True


def test_native_tamp_dump_checks_range_uniqueness_and_checksum(tmp_path):
    dump = tmp_path / "ours_retained_tokens.jsonl"
    assert verify_native_tamp_dump(dump)["records"] == 0
    row = {
        "event": "ours_retained_latest_memory_tokens",
        "num_total_tokens": 8, "num_retained_tokens": 3,
        "retained_token_indices": [0, 2, 7],
        "main_keep_indices": [0, 2],
        "usage_only_keep_indices": [2, 3],
        "diversity_complement_indices": [7],
    }
    dump.write_text(json.dumps(row) + "\n", encoding="utf8")
    valid = verify_native_tamp_dump(dump)
    assert valid["records"] == 1
    assert valid["sha256"] == file_sha256(dump)
    row["retained_token_indices"] = [0, 2, 8]
    dump.write_text(json.dumps(row) + "\n", encoding="utf8")
    with pytest.raises(RuntimeError, match="retained-token"):
        verify_native_tamp_dump(dump)
    row["retained_token_indices"] = [0, 2, 2]
    dump.write_text(json.dumps(row) + "\n", encoding="utf8")
    with pytest.raises(RuntimeError, match="retained-token"):
        verify_native_tamp_dump(dump)


def test_causal_effectiveness_guard_rejects_final_frame_only_writes():
    # Frame 2 is the last scored frame: writing memory then does not help it.
    with pytest.raises(RuntimeError, match="no BF16-visible memory changes"):
        verify_learned_causal_effectiveness({
            "seq_a": {"status": "completed", "predictions": [{}, {}],
                      "effective_memory_writes_affecting_future": 0}
        })
    # Videos with absent objects may have no effective writes; other videos
    # must still be allowed to demonstrate causal interventions.
    assert verify_learned_causal_effectiveness({
        "absent": {"status": "completed", "predictions": [{}, {}],
                   "effective_memory_writes_affecting_future": 0},
        "present": {"status": "completed", "predictions": [{}, {}, {}],
                    "effective_memory_writes_affecting_future": 1},
    }) == 2


def test_optional_anchor_drift_guard_requires_real_anchor_evidence():
    cases = {
        "good": {
            "status": "completed", "predictions": [{}, {}, {}],
            "effective_memory_writes_affecting_future": 1,
            "anchor_similarity_audit": {
                "eligible_writes": 3, "mean_abs_similarity_shift": 0.04,
            },
        }
    }
    assert verify_learned_causal_effectiveness(
        cases, max_anchor_cosine_shift=0.05) == 1
    with pytest.raises(RuntimeError, match="exceeded predeclared"):
        verify_learned_causal_effectiveness(
            cases, max_anchor_cosine_shift=0.01)
    cases["good"]["anchor_similarity_audit"] = {
        "eligible_writes": 0, "mean_abs_similarity_shift": None,
    }
    with pytest.raises(RuntimeError, match="without initialized anchors"):
        verify_learned_causal_effectiveness(
            cases, max_anchor_cosine_shift=0.05)


def test_scoring_reports_effective_memory_writes_and_anchor_drift(tmp_path):
    manifest, prompts, videos = benchmark(tmp_path)
    native = make_run(tmp_path, manifest, prompts, videos, "native", "native")
    fusion = make_run(tmp_path, manifest, prompts, videos, "fusion", "fusion")
    receipt_path = fusion / "receipt.json"
    receipt = json.loads(receipt_path.read_text())
    receipt["fusion_input_mode"] = "early0_only"
    for video in ("seq2", "seq3"):
        row = receipt["sequences"][video]
        row["memory_writes_checked"] = 4
        row["memory_writes_quantized_changed"] = 3
        row["anchor_similarity_audit"] = {
            "eligible_writes": 2, "mean_abs_similarity_shift": 0.2,
            "note": "Fixed-anchor cosine drift at WRITE; not TAMP Top-K outcomes.",
        }
    receipt_path.write_text(json.dumps(receipt))
    report = score_runs(
        {"native": native, "fusion": fusion}, manifest,
        tmp_path / "scored", bootstrap=200)
    summary = report["summary"]["fusion"]
    assert summary["fusion_input_mode"] == "early0_only"
    assert summary["memory_writes_checked"] == 8
    assert summary["memory_writes_modified_after_bf16"] == 6
    assert summary["effective_memory_write_fraction"] == 0.75
    assert summary["anchor_similarity_eligible_writes"] == 4
    assert summary["mean_abs_anchor_cosine_shift_at_write"] == pytest.approx(0.2)


def test_invalid_memory_audit_rejected(tmp_path):
    manifest, prompts, videos = benchmark(tmp_path)
    base = make_run(tmp_path, manifest, prompts, videos, "native", "native")
    identity = make_run(tmp_path, manifest, prompts, videos, "identity", "identity")
    path = identity / "receipt.json"
    data = json.loads(path.read_text())
    data["sequences"]["seq2"]["per_frame_resident_memory"][1]["local_frame_index"] = 200
    path.write_text(json.dumps(data))
    with pytest.raises(ValueError, match="Frame memory audit"):
        score_runs({"native": base, "identity": identity}, manifest,
                   tmp_path / "bad_memory")
