"""CPU tests for matched with/without-module Dice ablations."""
from pathlib import Path
import json

import numpy as np
import pytest
from PIL import Image

from datasets.video import ManifestFrame
from scripts.run_lean_memory_fusion_ablation import file_sha256, load_frozen_prompts
from scripts.score_lean_memory_ablations import score_runs, FORMAT


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
                "annotation_sha256": file_sha256(frame.mask_path),
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
        }
    receipt = {
        "format": FORMAT, "track": "lean_sam2_hiera_bplus",
        "arm": arm, "split": "dev", "label_id": 1,
        "manifest_sha256": file_sha256(manifest),
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
    path.write_text(json.dumps(receipt))
    report = score_runs({"native": base, "fusion": fusion}, manifest,
                        tmp_path / "report")
    assert not report["summary"]["fusion"]["resident_spatial_and_insurance_matched"]
    assert report["summary"]["fusion"]["resident_spatial_token_deltas"]["seq2"] == 2


def test_runtime_and_annotation_provenance_fail_closed(tmp_path):
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
    Image.fromarray(np.full((4, 4), 1, dtype=np.uint8)).save(
        videos["seq2"][1].mask_path)
    with pytest.raises(ValueError, match="Ground-truth file changed"):
        score_runs({"native": base, "fusion": fusion}, manifest, tmp_path/"scores")


def test_prompt_authenticates_images_and_rejects_oracle_path(tmp_path):
    manifest, prompts_path, videos = benchmark(tmp_path)
    assert set(load_frozen_prompts(prompts_path, videos)) == {"seq2", "seq3"}
    rows = [json.loads(x) for x in prompts_path.read_text().splitlines()]
    rows[0]["mask_path"] = str(videos["seq2"][0].mask_path)
    rows[0]["mask_sha256"] = file_sha256(videos["seq2"][0].mask_path)
    prompts_path.write_text("".join(json.dumps(p)+"\n" for p in rows))
    with pytest.raises(ValueError, match="GT-path"):
        load_frozen_prompts(prompts_path, videos)
