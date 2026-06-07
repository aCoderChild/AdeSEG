#!/usr/bin/env python3
"""Run the PolypGen comparison experiment for YOLO_SAM2 and MedSAM2."""

import argparse
import json
import os
import subprocess
import sys
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_CONFIG = PROJECT_ROOT / "experiments" / "polypgen_medsam2_yolo_sam2.json"


def repo_path(value):
    path = Path(value)
    return path if path.is_absolute() else PROJECT_ROOT / path


def method_output_root(method_cfg, config):
    if config.get("_output_root_override"):
        return repo_path(config["_output_root_override"])
    return repo_path(method_cfg.get("output_root", config["output_root"]))


def load_config(path):
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def sequence_numbers(config):
    seq_cfg = config.get("sequences", {})
    if "ids" in seq_cfg:
        return [int(seq) for seq in seq_cfg["ids"]]
    start = int(seq_cfg.get("start", 1))
    end = int(seq_cfg.get("end", start))
    return list(range(start, end + 1))


def enabled_methods(config, selected_methods):
    methods = config.get("methods", {})
    if selected_methods:
        unknown = sorted(set(selected_methods) - set(methods))
        if unknown:
            raise SystemExit(f"Unknown method(s): {', '.join(unknown)}")
        method_names = selected_methods
    else:
        method_names = [name for name, cfg in methods.items() if cfg.get("enabled", True)]
    return [(name, merged_method_config(config, methods[name])) for name in method_names]


def merged_method_config(config, method_cfg):
    defaults = config.get("method_defaults", {})
    runner = method_cfg.get("runner")
    merged = {}
    merged.update(defaults.get("all", {}))
    if runner:
        merged.update(defaults.get(runner, {}))
    merged.update(method_cfg)
    return merged


def run_command(cmd, env, dry_run=False):
    print("\n$ " + " ".join(str(part) for part in cmd), flush=True)
    if dry_run:
        return
    subprocess.run(cmd, cwd=PROJECT_ROOT, env=env, check=True)


def base_env(extra_pythonpath):
    env = os.environ.copy()
    paths = [str(PROJECT_ROOT), str(repo_path(extra_pythonpath))]
    if env.get("PYTHONPATH"):
        paths.append(env["PYTHONPATH"])
    env["PYTHONPATH"] = os.pathsep.join(paths)
    env.setdefault("PYTORCH_ENABLE_MPS_FALLBACK", "1")
    env.setdefault("MPLCONFIGDIR", "/tmp/matplotlib")
    env.setdefault("XDG_CACHE_HOME", "/tmp/cache")
    return env


def prepare_bboxes(config, seqs, python_bin, dry_run):
    data_root = repo_path(config["data_root"])
    bbox_root = repo_path(config["bbox_root"])
    for seq in seqs:
        cmd = [
            python_bin,
            str(PROJECT_ROOT / "scripts" / "utils" / "ground_truth_bbox_gen.py"),
            "--seq_num",
            str(seq),
            "--data_root",
            str(data_root),
            "--output_root",
            str(bbox_root),
        ]
        run_command(cmd, os.environ.copy(), dry_run=dry_run)


def run_yolo_sam2(method_cfg, config, seqs, python_bin, dry_run):
    env = base_env("external/YOLO_SAM2")
    data_root = repo_path(config["data_root"])
    output_root = method_output_root(method_cfg, config)
    script = repo_path(method_cfg["script"])
    cmd = [
        python_bin,
        str(script),
        "--base_dir",
        str(data_root),
        "--sam2_cfg",
        method_cfg["sam2_cfg"],
        "--sam2_checkpoint",
        str(repo_path(method_cfg["sam2_checkpoint"])),
        "--yolo_checkpoint",
        str(repo_path(method_cfg["yolo_checkpoint"])),
        "--output_root",
        str(output_root),
        "--method_name",
        method_cfg.get("output_name", "YOLO_SAM2"),
        "--prompt_source",
        method_cfg.get("prompt_source", "yolo"),
        "--bbox_root",
        str(repo_path(method_cfg.get("bbox_root", config["bbox_root"]))),
        "--seq_nums",
        *[str(seq) for seq in seqs],
        "--yolo_conf",
        str(method_cfg.get("yolo_conf", 0.5)),
        "--yolo_imgsz",
        str(method_cfg.get("yolo_imgsz", 640)),
        "--max_yolo_boxes_per_frame",
        str(method_cfg.get("max_yolo_boxes_per_frame", 1)),
    ]
    run_command(cmd, env, dry_run=dry_run)


def run_medsam2(method_cfg, config, seqs, python_bin, dry_run):
    env = base_env("external/MedSAM2")
    data_root = repo_path(config["data_root"])
    output_root = method_output_root(method_cfg, config)
    script = repo_path(method_cfg["script"])
    output_name = method_cfg.get("output_name", "MedSAM2")
    cmd = [
        python_bin,
        str(script),
        "--base_video_dir",
        str(data_root),
        "--seq_nums",
        *[str(seq) for seq in seqs],
        "--sam2_cfg",
        method_cfg["sam2_cfg"],
        "--sam2_checkpoint",
        str(repo_path(method_cfg["sam2_checkpoint"])),
        "--yolo_checkpoint",
        str(repo_path(method_cfg["yolo_checkpoint"])),
        "--output_root",
        str(output_root),
        "--method_name",
        output_name,
        "--log_dir",
        str(output_root / output_name / "logs"),
        "--prompt_source",
        method_cfg.get("prompt_source", "yolo"),
        "--bbox_root",
        str(repo_path(method_cfg.get("bbox_root", config["bbox_root"]))),
        "--yolo_conf",
        str(method_cfg.get("yolo_conf", 0.5)),
        "--yolo_imgsz",
        str(method_cfg.get("yolo_imgsz", 640)),
        "--max_yolo_boxes_per_frame",
        str(method_cfg.get("max_yolo_boxes_per_frame", 1)),
        "--video_prompt_source",
        method_cfg.get("video_prompt_source", "mask"),
        "--video_prompt_stride",
        str(method_cfg.get("video_prompt_stride", 1)),
        "--video_prompt_limit",
        str(method_cfg.get("video_prompt_limit", 0)),
    ]
    if method_cfg.get("save_intermediate_mask_prompts", False):
        cmd.append("--save_intermediate_mask_prompts")
    run_command(cmd, env, dry_run=dry_run)


def run_medsam3(method_cfg, config, seqs, python_bin, dry_run):
    env = base_env("external/MedSAM3")
    data_root = repo_path(config["data_root"])
    output_root = method_output_root(method_cfg, config)
    script = repo_path(method_cfg["script"])
    cmd = [
        python_bin,
        str(script),
        "--base_dir",
        str(data_root),
        "--seq_nums",
        *[str(seq) for seq in seqs],
        "--output_root",
        str(output_root),
        "--method_name",
        method_cfg.get("output_name", "MedSAM3_TEXT_POLYP"),
        "--bbox_root",
        str(repo_path(method_cfg.get("bbox_root", config["bbox_root"]))),
        "--config",
        str(repo_path(method_cfg["config"])),
        "--prompt",
        *method_cfg.get("prompt", ["polyp"]),
        "--threshold",
        str(method_cfg.get("threshold", 0.5)),
        "--nms_iou",
        str(method_cfg.get("nms_iou", 0.5)),
        "--resolution",
        str(method_cfg.get("resolution", 1008)),
        "--max_detections_per_frame",
        str(method_cfg.get("max_detections_per_frame", 0)),
    ]
    if method_cfg.get("weights"):
        cmd.extend(["--weights", str(repo_path(method_cfg["weights"]))])
    if method_cfg.get("base_checkpoint"):
        cmd.extend(["--base_checkpoint", str(repo_path(method_cfg["base_checkpoint"]))])
    if method_cfg.get("blank_on_frame_error", False):
        cmd.append("--blank_on_frame_error")
    if method_cfg.get("run_eval", False):
        cmd.append("--run_eval")
    run_command(cmd, env, dry_run=dry_run)


def run_inference(method_name, method_cfg, config, seqs, python_bin, dry_run):
    runner = method_cfg.get("runner", method_name)
    method_cfg = dict(method_cfg)
    method_cfg.setdefault("output_name", method_name)
    if runner == "YOLO_SAM2":
        run_yolo_sam2(method_cfg, config, seqs, python_bin, dry_run)
    elif runner == "MedSAM2":
        run_medsam2(method_cfg, config, seqs, python_bin, dry_run)
    elif runner == "MedSAM3":
        run_medsam3(method_cfg, config, seqs, python_bin, dry_run)
    else:
        raise SystemExit(f"No runner implemented for method: {method_name} (runner={runner})")


def evaluate(config, method_names, seqs, python_bin, eval_type, dry_run, output_root=None):
    output_root = repo_path(output_root or config["output_root"])
    cmd = [
        python_bin,
        str(PROJECT_ROOT / "scripts" / "utils" / "eval_metrics.py"),
        "--methods",
        *method_names,
        "--data_root",
        str(repo_path(config["data_root"])),
        "--bbox_root",
        str(repo_path(config["bbox_root"])),
        "--outputs_root",
        str(output_root),
        "--num_sequences",
        str(max(seqs)),
        "--seqs",
        *[str(seq) for seq in seqs],
        "--eval_type",
        eval_type,
    ]
    run_command(cmd, base_env("."), dry_run=dry_run)


def main():
    parser = argparse.ArgumentParser(
        description="Prepare, run, and evaluate the MedSAM2 vs YOLO_SAM2 experiment."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG), help="Experiment JSON config.")
    parser.add_argument(
        "--stage",
        choices=["all", "prepare", "infer", "eval"],
        default="all",
        help="Which stage to run.",
    )
    parser.add_argument(
        "--methods",
        nargs="+",
        help="Restrict inference/evaluation to selected methods.",
    )
    parser.add_argument("--seqs", nargs="+", type=int, help="Override sequence ids.")
    parser.add_argument("--python", default=sys.executable, help="Python interpreter to use.")
    parser.add_argument(
        "--output_root",
        help="Override all method output roots, including method-specific output_root settings.",
    )
    parser.add_argument(
        "--eval_type",
        choices=["bbox", "mask", "both"],
        default=None,
        help="Evaluation type passed to eval_metrics.py. Defaults to each method's config or both.",
    )
    parser.add_argument("--dry_run", action="store_true", help="Print commands without running them.")
    args = parser.parse_args()

    config = load_config(repo_path(args.config))
    if args.output_root:
        config["_output_root_override"] = args.output_root
    seqs = args.seqs or sequence_numbers(config)
    methods = enabled_methods(config, args.methods)
    method_names = [name for name, _ in methods]

    print(f"Experiment: {config.get('name', Path(args.config).stem)}")
    print(f"Sequences: {', '.join('seq' + str(seq) for seq in seqs)}")
    print(f"Methods: {', '.join(method_names)}")

    if args.stage in ("all", "prepare"):
        prepare_bboxes(config, seqs, args.python, args.dry_run)

    if args.stage in ("all", "infer"):
        for method_name, method_cfg in methods:
            run_inference(method_name, method_cfg, config, seqs, args.python, args.dry_run)

    if args.stage in ("all", "eval"):
        if args.eval_type is not None:
            grouped_methods = {}
            for method_name, method_cfg in methods:
                grouped_methods.setdefault(method_output_root(method_cfg, config), []).append(method_name)
            for output_root, grouped_method_names in grouped_methods.items():
                evaluate(
                    config,
                    grouped_method_names,
                    seqs,
                    args.python,
                    args.eval_type,
                    args.dry_run,
                    output_root=output_root,
                )
        else:
            for method_name, method_cfg in methods:
                method_eval_type = method_cfg.get(
                    "eval_type",
                    "mask" if method_cfg.get("prompt_source") == "gt_bbox" else "both",
                )
                evaluate(
                    config,
                    [method_name],
                    seqs,
                    args.python,
                    method_eval_type,
                    args.dry_run,
                    output_root=method_output_root(method_cfg, config),
                )


if __name__ == "__main__":
    main()
