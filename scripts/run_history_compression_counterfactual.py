"""Bounded native/dense/compressed counterfactual replay diagnostic.

Reuses native replay/bank utilities and the existing ten-map passthrough.
GT masks are read only for initialization and post-forward scoring.
No diagnostic output is written to memory.
"""

import argparse
import gc
import json
from pathlib import Path
import subprocess
import sys
import tempfile
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from datasets.video import load_manifest
from modeling.attention_distortion_observer import _sha256_tensor
from scripts.run_soft_role_affinity_cv import score_frame
import torch.nn.functional as F
from modeling.native_history_cache import cached_read, load_bank, hybrid_memory_outputs
from modeling.wider_history_tome import FixedBudgetLongHistoryToMe
from scripts.bank_integrity import fingerprint
from scripts.memory_token_io import digest, parameter_digest, validate_output
from scripts.run_attention_distortion_diagnostic import _compare_prompt
from scripts.run_author_baseline import build_author_predictor
from scripts.run_cfbi_region_pilot import prepare
from scripts.run_regional_refinement import write


def track_read(model, sample, frame):
    return model.track_step(
        frame_idx=frame, is_init_cond_frame=False,
        current_vision_feats=sample["vision"],
        current_vision_pos_embeds=sample["positions"], feat_sizes=sample["sizes"],
        point_inputs=None, mask_inputs=None, output_dict=sample["output_dict"],
        num_frames=sample["num_frames"], track_in_reverse=False,
        run_mem_encoder=False,
    )


def require_same_prediction(left, right, context):
    for name in ("pred_masks", "object_score_logits"):
        if not torch.equal(left[name], right[name]):
            raise RuntimeError(f"{context}: {name} differs")


def validate_matched_reads(dense, compressed):
    if [x["layer"] for x in dense] != [0,1,2,3] or [x["layer"] for x in compressed] != [0,1,2,3]:
        raise RuntimeError("Every counterfactual requires four layers")
    for left,right in zip(dense,compressed,strict=True):
        if left["pointer_keys"] != right["pointer_keys"]:
            raise RuntimeError("Counterfactual pointer counts differ")
        a=[(x["frame"],x["t_pos"]) for x in left["frames"]]
        b=[(x["frame"],x["t_pos"]) for x in right["frames"]]
        if left["spatial_keys_before"] != right["spatial_keys_before"]:
            raise RuntimeError("Counterfactual input spatial counts differ")
        if a != b:
            raise RuntimeError("Counterfactual source frames differ")
        if right["spatial_keys_after"] > 28672:
            raise RuntimeError("Compressed read exceeds native token budget")
        if right["spatial_key_mass_sum"] != right["spatial_keys_before"]:
            raise RuntimeError("Counterfactual loses spatial mass")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    args = parser.parse_args()
    protocol = json.loads(args.protocol.read_text())
    checksum = args.protocol.with_suffix(args.protocol.suffix+".sha256")
    if protocol.get("status") != "frozen_for_engineering_gate" or digest(args.protocol) != checksum.read_text().strip():
        raise RuntimeError("Engineering protocol is not frozen or its hash differs")
    reference_reader = protocol.get("reference_reader", "native")
    if reference_reader not in ("native", "compressed_ten"):
        raise ValueError("Unknown cached-bank reference reader")
    out = validate_output(args.output_dir)
    if not out.is_relative_to(Path(protocol["output_root"]).resolve()) or out.exists():
        raise RuntimeError("Use a new child of the frozen Drive output root")
    for name, expected in protocol["source_hashes"].items():
        if digest(ROOT/name) != expected:
            raise RuntimeError(f"Frozen source changed: {name}")
    input_path = Path(protocol["input_plan"])
    native_path = Path(protocol["native_plan"])
    if digest(input_path) != protocol["input_plan_sha256"] or digest(native_path) != protocol["native_plan_sha256"]:
        raise RuntimeError("Frozen bank/target plan differs")
    inputs = json.loads(input_path.read_text())
    native_plan = json.loads(native_path.read_text())
    donor_inputs = None
    if protocol.get("donor_input_plan"):
        donor_path = Path(protocol["donor_input_plan"])
        if digest(donor_path) != protocol["donor_input_plan_sha256"]:
            raise RuntimeError("Donor input plan changed")
        donor_inputs = json.loads(donor_path.read_text())
    config = native_plan["protocol"]
    repo = ROOT/config["sam2_repository"]
    if subprocess.check_output(["git","rev-parse","HEAD"],cwd=repo,text=True).strip() != config["sam2_revision"]:
        raise RuntimeError("SAM2 reference revision changed")
    if subprocess.check_output(["git","status","--porcelain","--untracked-files=no"],cwd=repo,text=True).strip():
        raise RuntimeError("SAM2 reference has tracked changes")
    if "sam2" in sys.modules:
        raise RuntimeError("SAM2 was imported before selecting the clean reference")
    if digest(Path(config["checkpoint"])) != config["checkpoint_sha256"]:
        raise RuntimeError("Checkpoint changed")
    manifest = ROOT/config["manifest"]
    if digest(manifest) != config["manifest_sha256"]:
        raise RuntimeError("Manifest changed")
    sys.path.insert(0,str(repo))
    torch.set_num_threads(2)
    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261111)
    videos = {**load_manifest(manifest,"dev"),**load_manifest(manifest,"test")}
    out.mkdir(parents=True)
    write(out/"setup.json", {"protocol":protocol,"protocol_sha256":digest(args.protocol)})
    results = {}
    try:
        for sequence, targets in protocol["targets"].items():
            verified = inputs["banks"][sequence]
            selected = {row["frame"]:row for row in verified["targets"]}
            if any(frame not in selected for frame in targets):
                raise RuntimeError("Gate target was not input-verified")
            for frame in targets:
                if digest(Path(selected[frame]["image"])) != selected[frame]["image_sha256"]:
                    raise RuntimeError("A verified target image changed")
            folder = Path(verified["cache_directory"])
            if digest(folder/"receipt.json") != verified["receipt_sha256"]:
                raise RuntimeError("Native bank receipt changed")
            receipt = json.loads((folder/"receipt.json").read_text())
            payload = load_bank(folder/"bank.pt",receipt,config["checkpoint_sha256"],
                                config["manifest_sha256"],config["model_config"])
            payload_before = fingerprint(payload)["sha256"]
            donor_payload = None
            if donor_inputs is not None:
                donor = donor_inputs["banks"][sequence]
                donor_folder = Path(donor["cache_directory"])
                if digest(donor_folder/"receipt.json") != donor["receipt_sha256"]:
                    raise RuntimeError("Donor receipt changed")
                donor_receipt = json.loads((donor_folder/"receipt.json").read_text())
                donor_payload = load_bank(donor_folder/"bank.pt",donor_receipt,
                    config["checkpoint_sha256"],config["manifest_sha256"],config["model_config"])
                if any(payload[k] != donor_payload[k] for k in ("object_id","prompt","num_frames")):
                    raise RuntimeError("Donor persistent object/video/prompt differs")
                donor_before = fingerprint(donor_payload)["sha256"]
            model = build_author_predictor(SimpleNamespace(
                method="sam2",device="cpu",checkpoint=Path(config["checkpoint"]),
                model_config=config["model_config"]),repo)
            weights = parameter_digest(model)
            if model.num_maskmem != 7 or model.memory_temporal_stride_for_eval != 1 or model.fill_hole_area != 0:
                raise RuntimeError("Native SAM2 evaluation settings differ")
            frames = videos[sequence]
            results[sequence] = []
            with tempfile.TemporaryDirectory(dir=out) as temporary:
                with torch.inference_mode():
                    state,prompt = prepare(model,frames,Path(temporary))
                    model.propagate_in_video_preflight(state)
                    if prompt != payload["prompt"] or len(frames) != payload["num_frames"]:
                        raise RuntimeError("Native prompt/video differs")
                    index = state["obj_id_to_idx"][payload["object_id"]]
                    _compare_prompt(state["output_dict_per_obj"][index],payload["outputs"],prompt,sequence)
                    for frame in targets:
                        write(out/"progress.json",{"sequence":sequence,"frame":frame,"status":"running"})
                        sample = cached_read(model,state,frame,payload)
                        bank_before = fingerprint(sample["output_dict"])["sha256"]
                        native = track_read(model,sample,frame)
                        if reference_reader == "native":
                            require_same_prediction(native,{"pred_masks":sample["native_logits"],
                                "object_score_logits":sample["native_object_score"]},"Native cached replay")
                        with FixedBudgetLongHistoryToMe(model,mode="global",max_noncond_maps=10) as dense:
                            dense.passthrough = True
                            reference = track_read(model,sample,frame)
                        with FixedBudgetLongHistoryToMe(model,mode="global",max_noncond_maps=10) as compressed:
                            compressed_output = track_read(model,sample,frame)
                        if reference_reader == "compressed_ten":
                            require_same_prediction(compressed_output,{"pred_masks":sample["native_logits"],
                                "object_score_logits":sample["native_object_score"]},"Archived treatment cached replay")
                        validate_matched_reads(dense.calls, compressed.calls)
                        prefix = frame-prompt <= 7
                        if prefix:
                            require_same_prediction(reference,native,"Pre-compression dense parity")
                            require_same_prediction(compressed_output,native,"Pre-compression merged parity")
                        if fingerprint(sample["output_dict"])["sha256"] != bank_before:
                            raise RuntimeError("Counterfactual changed the source bank")
                        if model.num_maskmem != 7 or parameter_digest(model) != weights:
                            raise RuntimeError("Counterfactual changed weights or temporal slots")
                        extra_outputs = []
                        if donor_payload is not None:
                            for component in ("spatial","pointer"):
                                hybrid = dict(sample)
                                hybrid["output_dict"] = hybrid_memory_outputs(
                                    sample["output_dict"],donor_payload["outputs"],component=component)
                                with FixedBudgetLongHistoryToMe(model,mode="global",max_noncond_maps=10) as swapped:
                                    changed = track_read(model,hybrid,frame)
                                validate_matched_reads(dense.calls,swapped.calls)
                                extra_outputs.append(("donor_"+component,changed))
                            if fingerprint(donor_payload)["sha256"] != donor_before:
                                raise RuntimeError("Component swap changed donor bank")
                            if fingerprint(sample["output_dict"])["sha256"] != bank_before:
                                raise RuntimeError("Component swap changed base bank")
                            if model.num_maskmem != 7 or parameter_digest(model) != weights:
                                raise RuntimeError("Component swap changed model state")
                        # All model reads finish before any post-prompt GT-mask scoring.
                        arms = {}
                        for name, prediction in [ ("native",native),("dense_ten",reference),
                                                 ("compressed_ten",compressed_output), *extra_outputs ]:
                            logits = F.interpolate(prediction["pred_masks"].float(),
                                size=(state["video_height"],state["video_width"]),
                                mode="bilinear",align_corners=False)
                            arms[name] = {**score_frame(logits,frames,frame),
                                "object_score":float(prediction["object_score_logits"].item()),
                                "lowres_mask_sha256":_sha256_tensor(prediction["pred_masks"])}
                        results[sequence].append({"frame":frame,"object_id":payload["object_id"],
                            "reference_replay_exact":True,"reference_reader":reference_reader,
                            "precompression_native_parity":prefix,
                            "source_bank_unchanged":True,"arms":arms,
                            "dense_reads":dense.calls,"compressed_reads":compressed.calls})
                        write(out/"results.json",results)
                        del sample,native,reference,compressed_output,compressed,dense
                    model.reset_state(state)
            if fingerprint(payload)["sha256"] != payload_before or parameter_digest(model) != weights:
                raise RuntimeError("Bank/model changed during the gate")
            del model,state,payload
            gc.collect()
        if any(digest(ROOT/name)!=expected for name,expected in protocol["source_hashes"].items()):
            raise RuntimeError("A frozen source changed during the gate")
        write(out/"completion.json",{"passed":True,"scope":"fixed-bank immediate-read diagnostic; no propagated Dice inference",
            "targets":sum(map(len,results.values())),"layers_per_target":4,
            "initial_prompt_gt_mask_read":True,"post_prompt_gt_masks_read":True, "post_prompt_gt_use":"post-forward scoring only",
            "no_memory_writes":True,
            "protocol_sha256":digest(args.protocol),"results_sha256":digest(out/"results.json")})
        write(out/"progress.json",{"status":"complete"})
    except Exception as exc:
        write(out/"failure.json",{"error":str(exc),"type":type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
