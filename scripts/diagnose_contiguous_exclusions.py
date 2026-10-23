"""Replay one failed target through the unchanged exclusion conditions."""

import argparse
import json
from pathlib import Path
import sys
from types import SimpleNamespace

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts import run_contiguous_exclusions as runner  # noqa: E402
from training.tracking import start_wandb, finalize_tracking  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument("--manifest", type=Path, required=True)
    parser.add_argument("--reference-dir", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--sequence", default="seq12")
    parser.add_argument("--target", type=int, default=15)
    parser.add_argument(
        "--history", action="store_true", help="Include all preceding target exclusions"
    )
    args = parser.parse_args()
    output = runner.validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    runner.torch.set_num_threads(2)
    for variable in ("WANDB_CACHE_DIR", "WANDB_DATA_DIR"):
        runner.os.environ[variable] = str(output / variable.lower())
    repo, revision = runner.author.source_identity("sam2")
    runtime = SimpleNamespace(
        method="sam2", device="cpu", checkpoint=args.checkpoint, manifest=args.manifest
    )
    identity = runner.author.signature(runtime, repo, revision)
    reference = json.loads((args.reference_dir / "setup.json").read_text())
    if any(reference[k] != v for k, v in identity.items()):
        raise RuntimeError("Reference identity mismatch")
    identity.update(
        diagnostic_sequence=args.sequence,
        diagnostic_target=args.target,
        source_hashes={name: runner.digest(ROOT / name) for name in runner.SOURCES},
    )
    (output / "setup.json").write_text(json.dumps(identity, indent=2) + "\n")
    run = start_wandb(
        "ade-seg",
        "adenoid-hypertrophy",
        "17 | Exclusion failure diagnosis | seq12 frame15",
        "online",
        "https://forge.coreweave.com/api/wandb",
        identity,
        output,
        ["integrity-diagnosis", "cpu-fp32"],
    )
    runner.sys.path.insert(0, str(repo))
    model = runner.author.build_author_predictor(runtime, repo)
    videos = {
        **runner.load_manifest(args.manifest, "dev"),
        **runner.load_manifest(args.manifest, "test"),
    }
    protocol = json.loads((ROOT / runner.PROTOCOL).read_text())
    assignment_rows = [
        json.loads(line)
        for line in (args.reference_dir / args.sequence / "assignment_frames.jsonl")
        .read_text()
        .splitlines()
    ]
    prompt = min(row["frame"] for row in assignment_rows)
    protocol["gate"]["selected_targets"] = {
        args.sequence: list(range(prompt + 1, args.target + 1))
        if args.history
        else [args.target]
    }
    original_propagate = model.propagate_in_video

    def limited_propagation(*positional, **keywords):
        iterator = original_propagate(*positional, **keywords)
        try:
            for row in iterator:
                yield row
                if row[0] == args.target:
                    break
        finally:
            iterator.close()

    model.propagate_in_video = limited_propagation
    native_infer = runner.infer
    expected = {
        r["frame"]: r["prediction_sha256"]
        for r in map(
            json.loads,
            (args.reference_dir / args.sequence / "frames.jsonl")
            .read_text()
            .splitlines(),
        )
    }
    counter = 0

    def observed_infer(model, state, frame):
        nonlocal counter
        counter += 1
        value = native_infer(model, state, frame)
        row = {
            "call": counter,
            "frame": frame,
            "prediction_sha256": runner.author.tensor_hash(value),
        }
        row["matches_native"] = row["prediction_sha256"] == expected[frame]
        with (output / "calls.jsonl").open("a") as stream:
            stream.write(json.dumps(row) + "\n")
        return value

    runner.infer = observed_infer
    try:
        result = runner.sequence(
            model,
            args.sequence,
            videos[args.sequence],
            args.reference_dir,
            output,
            protocol,
            True,
            run,
        )
        summary = {"completed": True, "diagnostic_passed": True, "sequence": result}
    except BaseException as error:
        summary = {
            "completed": True,
            "diagnostic_passed": False,
            "error": str(error),
            "type": type(error).__name__,
        }
    (output / "diagnosis.json").write_text(json.dumps(summary, indent=2) + "\n")
    files = ["setup.json", "diagnosis.json", "calls.jsonl"]
    failure = output / args.sequence / "integrity_failure.json"
    if failure.exists():
        files.append(str(failure.relative_to(output)))
    finalize_tracking(run, output, summary, files)
    run.finish()


if __name__ == "__main__":
    main()
