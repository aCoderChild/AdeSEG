#!/usr/bin/env python3
"""Official Efficient-SAM2 SMR with dense encoder; reuse matched author runner.

Device-only vendor allocation patch. No replacement model or attention code.
"""

import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
from types import SimpleNamespace

import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_author_baseline as author  # noqa: E402

REVISION = "8e1ad06089b9c958ff5ebfc72b9b7d0b904f5ce5"
PATCH_FILE = "sam2/sam2_video_predictor.py"
OLD = "u_mode = torch.zeros((64,64), dtype=torch.bool, device='cuda')"
NEW = "u_mode = torch.zeros((64,64), dtype=torch.bool, device=self.device)"
SETTINGS = dict(
    apply_WB=True,
    disable_WB=True,
    selected_layers=[],
    sam2_model="base+",
    win_sel_layer=[6],
    fpn_feat_layer=None,
    scale_layer=None,
    WB_all_layer=False,
    small_bypass=False,
    Mem_filter=False,
    prune_memory=True,
    pool_memory=False,
    print_WS=False,
    random_mask=False,
    uniform_mask=False,
    topk_mask=True,
    set_drop_ratio=0.95,
    MTP_theta=0.5,
    Mem_Frame_Prune=False,
    num_frame_to_prune=0,
    WB_theta=0.7,
    obj_theta=5,
    sel_max_iou=False,
)


def verify_source(repo):
    revision = subprocess.check_output(
        ["git", "-C", str(repo), "rev-parse", "HEAD"], text=True
    ).strip()
    changed = subprocess.check_output(
        ["git", "-C", str(repo), "diff", "HEAD", "--name-only"], text=True
    ).splitlines()
    original = subprocess.check_output(
        ["git", "-C", str(repo), "show", f"HEAD:{PATCH_FILE}"], text=True
    )
    if revision != REVISION or changed != [PATCH_FILE] or original.count(OLD) != 1:
        raise ValueError("Unexpected author revision or source changes")
    if (repo / PATCH_FILE).read_text() != original.replace(OLD, NEW):
        raise ValueError("Vendor modification exceeds the declared allocation patch")
    return repo, revision


def build(args, repo):
    from sam2.build_sam import build_sam2_video_predictor
    from build_model.window_bypass import build_WB_model

    torch.use_deterministic_algorithms(True)
    torch.manual_seed(20261021)
    model = build_sam2_video_predictor(
        author.MODEL_CONFIG,
        str(args.checkpoint),
        device=args.device,
        apply_postprocessing=False,
    )
    for name, value in SETTINGS.items():
        setattr(model, name, value)
    model.init_memory_info(enable_MeP_info=True)
    # Empty selected_layers means no encoder blocks/attention are replaced.
    # Author factory still installs its own SMR predictor and attention classes.
    build_WB_model(SimpleNamespace(**SETTINGS), model, selected_layers=[])
    model.time_log = {}
    model.WS_log = {name: [] for name in ("Sel", "Mask", "PF")}
    for layer in model.memory_attention.layers:
        layer.cross_attn_image.drop_ratio_log = []
    model.eval().requires_grad_(False)
    if not model.disable_WB or model.image_encoder.trunk.selected_layers:
        raise RuntimeError("Image routing must remain disabled")
    if any(m.training for m in model.modules()):
        raise RuntimeError("Training module in inference")
    return model


def check_pruning(records):
    for layer in range(4):
        rows = [r for r in records if r["layer"] == layer]
        if not rows or not any(r["retained_keys"] < r["original_keys"] for r in rows):
            raise RuntimeError(f"SMR inactive in layer {layer}")
        if any(
            r["retained_keys"] > r["original_keys"]
            or r["retained_keys"] < r["pointer_keys"]
            for r in rows
        ):
            raise RuntimeError("Invalid retained key count")


def main():
    method = "efficient_smr"
    author.REPOSITORIES[method] = ("external/Efficient-SAM2", REVISION)
    original_source = author.source_identity
    author.source_identity = lambda name: (
        verify_source(ROOT / "external/Efficient-SAM2")
        if name == method
        else original_source(name)
    )
    original_build = author.build_author_predictor
    author.build_author_predictor = lambda args, repo: (
        build(args, repo) if args.method == method else original_build(args, repo)
    )
    original_iterator = author.iter_predictions
    author.iter_predictions = lambda model, state, name, prompt: (
        model.propagate_in_video_wj(state, start_frame_idx=prompt, reverse=False)
        if name == method
        else original_iterator(model, state, name, prompt)
    )
    original_signature = author.signature

    def signature(*args):
        return {
            **original_signature(*args),
            "smr_adapter_sha256": author.digest(Path(__file__)),
            "vendor_device_patch_sha256": author.digest(
                ROOT / "external/Efficient-SAM2" / PATCH_FILE
            ),
            "smr_settings": SETTINGS,
            "deterministic_algorithms": True,
            "mps_environment": {
                key: os.environ.get(key)
                for key in (
                    "PYTORCH_MPS_PREFER_METAL",
                    "PYTORCH_MPS_FAST_MATH",
                    "PYTORCH_ENABLE_MPS_FALLBACK",
                )
            },
        }

    author.signature = signature
    original_start = author.start_wandb

    def start(project, entity, name, mode, url, config, output, tags):
        config["efficient_smr"] = {
            "settings": SETTINGS,
            "precision": "FP32 model; native BF16 memory storage",
            "routing": "disabled; no encoder blocks replaced",
            "adaptation": "one device allocation; MPS/device-specific replay required",
        }
        (output / "setup.json").write_text(json.dumps(config, indent=2) + "\n")
        shutil.copy2(__file__, output / "smr_adapter_source.py")
        patch = subprocess.check_output(
            ["git", "-C", str(ROOT / "external/Efficient-SAM2"), "diff", "HEAD"],
            text=True,
        )
        (output / "vendor_device.patch").write_text(patch)
        return original_start(
            project,
            entity,
            f"10 | Efficient-SAM2 SMR-only | {config['scope']} | {config['device']}",
            mode,
            url,
            config,
            output,
            tags + ["efficient-smr", "dense-encoder"],
        )

    author.start_wandb = start
    original_rollout = author.rollout

    def rollout(
        model,
        name,
        frames,
        prompt,
        folder,
        device,
        run,
        repeat=False,
        capture_logits=False,
    ):
        from sam2.modeling.backbones.hieradet import Hiera, MultiScaleBlock

        if any(
            type(block) is not MultiScaleBlock
            for block in model.image_encoder.trunk.blocks
        ):
            raise RuntimeError("Encoder block was replaced")
        records, handles, pending, encoder = [], [], {}, []

        def encoder_check(module, args, result):
            if encoder:
                return
            reference = Hiera.forward(module, args[0])
            differences = [
                float((a - b).abs().max()) for a, b in zip(result, reference)
            ]
            exact = len(result) == len(reference) and all(
                torch.equal(a, b) for a, b in zip(result, reference)
            )
            encoder.append(
                {
                    "dense_encoder_bit_identical": exact,
                    "maximum_absolute_difference": differences,
                }
            )
            if not exact:
                raise RuntimeError(f"Dense encoder parity failed: {differences}")

        handles.append(model.image_encoder.trunk.register_forward_hook(encoder_check))
        for layer, block in enumerate(model.memory_attention.layers):

            def before(module, args, kwargs, layer=layer):
                pending[layer] = {
                    "layer": layer,
                    "original_keys": kwargs["k"].shape[1],
                    "pointer_keys": int(kwargs.get("num_k_exclude_rope", 0)),
                }

            def projected(module, args, layer=layer):
                records.append({**pending[layer], "retained_keys": args[0].shape[1]})

            handles.append(
                block.cross_attn_image.register_forward_pre_hook(
                    before, with_kwargs=True
                )
            )
            handles.append(
                block.cross_attn_image.k_proj.register_forward_pre_hook(projected)
            )
        try:
            result = original_rollout(
                model, name, frames, prompt, folder, device, run, repeat, capture_logits
            )
            check_pruning(records)
            if not encoder or not encoder[0]["dense_encoder_bit_identical"]:
                raise RuntimeError("No encoder parity receipt")
            return result
        finally:
            for handle in handles:
                handle.remove()
            suffix = "repeat" if repeat else "first"
            (folder / f"smr_verification_{suffix}.json").write_text(
                json.dumps({"encoder": encoder, "key_counts": records}, indent=2) + "\n"
            )

    author.rollout = rollout
    original_finalize = author.finalize_tracking
    author.finalize_tracking = lambda run, output, summary, names: original_finalize(
        run,
        output,
        summary,
        names
        + ["smr_adapter_source.py", "vendor_device.patch"]
        + [
            str(p.relative_to(output))
            for p in output.glob("seq*/smr_verification_*.json")
        ],
    )
    author.main()


if __name__ == "__main__":
    main()
