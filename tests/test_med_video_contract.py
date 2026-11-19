"""Hermetic protocol checks against synthetic bytes, never real private checkpoints."""
import json
from pathlib import Path
import numpy as np
from PIL import Image
import pytest

from adeseg.research.protocol import digest
from adeseg.research.med_video_contract import (
    EVAL_FIXED, MODEL_CONFIG, encode_prompt_records,
    load_med_video_contract, prompt_records, verify_med_run_receipt,
)
from datasets.video import load_manifest


def make_contract(tmp_path):
    for i in range(3):
        Image.fromarray(np.full((8,8,3), 35, np.uint8)).save(tmp_path/f"{i}.jpg")
        mask = np.full((8,8), 1 if i >= 1 else 0, np.uint8)
        Image.fromarray(mask).save(tmp_path/f"{i}.png")
    manifest=tmp_path/"frames.jsonl"
    manifest.write_text("".join(json.dumps(dict(
        split="dev", video_id="v1", frame_index=i, image=f"{i}.jpg", mask=f"{i}.png"
    ))+"\n" for i in range(3)))
    backbone=tmp_path/"dummy.pt"
    backbone.write_bytes(b"stub-backbone-not-for-inference")
    videos=load_manifest(manifest, "dev",require_contiguous=True,
                         require_jpeg=True,safe_output_ids=True)
    prompt=tmp_path/"prompts.jsonl"
    prompt.write_bytes(encode_prompt_records(prompt_records(videos, 1)))
    spec=dict(
        schema_version=1,experiment_id="test",track="medsam2_highres",
        dataset={"name":"toy","manifest":{"path":"frames.jsonl","sha256":digest(manifest)}},
        checkpoint={"path":"dummy.pt","sha256":digest(backbone)},split="dev",
        initialization={"source":"oracle_mask","prompt_budget":1},
        evaluation={**EVAL_FIXED, "model_config":MODEL_CONFIG,"label_id":1,
            "prompt_frame_manifest_path":"prompts.jsonl",
            "prompt_frame_manifest_sha256":digest(prompt)},
        stages={s:{"command":[],"artifacts":[]} for s in ("data","training","inference","evaluation")}
    )
    path=tmp_path/"protocol.json"
    path.write_text(json.dumps(spec))
    return path, prompt, digest(backbone)


def test_valid_contract_and_prompt_plan(tmp_path):
    path, prompt, sha=make_contract(tmp_path)
    result=load_med_video_contract(path,expected_backbone_sha256=sha)
    assert result["label_id"]==1
    assert result["videos"]["v1"][1].frame_index==1
    assert result["prompt_manifest_sha256"]==digest(prompt)


def test_reject_tampered_prompt_manifest(tmp_path):
    path,prompt,sha=make_contract(tmp_path)
    prompt.write_text(prompt.read_text().replace('"prompt_position":1','"prompt_position":0'))
    with pytest.raises(ValueError,match="Prompt-frame manifest"):
        load_med_video_contract(path,expected_backbone_sha256=sha)


def test_reject_semantically_invalid_prompt_even_when_rehashed(tmp_path):
    path,prompt,sha=make_contract(tmp_path)
    value=json.loads(prompt.read_text())
    value["prompt_position"]=0
    prompt.write_text(json.dumps(value,sort_keys=True,separators=(",",":"))+"\n")
    spec=json.loads(path.read_text())
    spec["evaluation"]["prompt_frame_manifest_sha256"]=digest(prompt)
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match="first-GT-positive"):
        load_med_video_contract(path,expected_backbone_sha256=sha)


def test_reject_protocol_rule_drift_and_receipt_drift(tmp_path):
    path,_,sha=make_contract(tmp_path)
    result=load_med_video_contract(path,expected_backbone_sha256=sha)
    with pytest.raises(ValueError,match="sealed contract"):
        verify_med_run_receipt(result,{"protocol":"oracle_first_present_mask_post_prompt_v1"})
    spec=json.loads(path.read_text())
    spec["evaluation"]["mask_threshold"]=0.4
    path.write_text(json.dumps(spec))
    with pytest.raises(ValueError,match="mask_threshold"):
        load_med_video_contract(path,expected_backbone_sha256=sha)
