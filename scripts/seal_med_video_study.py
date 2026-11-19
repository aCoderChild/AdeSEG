"""Seal already-executed MedSAM2 oracle-video runs with authenticated receipts.

Does not execute the GPU model. It refuses to certify an incomplete, mismatched
or unverified native/identity/trained set. "Sealed" is NOT "improved".
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path
import subprocess
import sys

from adeseg.research.protocol import digest
from adeseg.research.med_video_contract import load_med_video_contract
from scripts.evaluate_med_video import evaluate_run
from scripts.compare_med_runs import compare_runs

ROOT = Path(__file__).resolve().parents[1]
SOURCE_PATHS = (
    "modeling/medical_highres/adapter.py",
    "modeling/medical_highres/blocks.py",
    "modeling/medical_highres/integration.py",
    "modeling/medical_highres/checkpoints.py",
    "scripts/infer_med_video.py",
    "scripts/evaluate_med_video.py",
    "scripts/compare_med_runs.py",
    "adeseg/research/med_video_contract.py",
)


def source_revision():
    """Fail closed if executing from a dirty checkout instead of a stable commit."""
    revision = subprocess.check_output(
        ["git", "-C", str(ROOT), "rev-parse", "HEAD"], text=True).strip()
    for args in (["diff", "--quiet"], ["diff", "--cached", "--quiet"]):
        result = subprocess.run(["git", "-C", str(ROOT), *args], check=False)
        if result.returncode != 0:
            raise RuntimeError("Cannot seal study from a modified tracked checkout")
    if len(revision) != 40:
        raise ValueError("Invalid Git source revision")
    return revision


def collect_sha256(root, expected):
    """Hash only explicitly expected report files; never upload image data."""
    files = {}
    for relative in expected:
        path = root / relative
        if not path.is_file():
            raise FileNotFoundError(path)
        files[relative] = digest(path)
    return files


def seal_study(protocol, native, zero, output, *, trained=None):
    contract = load_med_video_contract(protocol)
    output = Path(output).expanduser().resolve()
    if output.exists():
        raise FileExistsError("Refusing to overwrite an existing reproduction receipt")
    native, zero = Path(native).resolve(strict=True), Path(zero).resolve(strict=True)
    if native == zero:
        raise ValueError("Native and identity control must be independent runs")
    if trained is not None:
        trained = Path(trained).resolve(strict=True)
        if trained in (native, zero):
            raise ValueError("Trained output must not alias a control run")
    manifest = contract["manifest"]
    # Each call re-reads source/prediction hashes and checks every raw logit file.
    native_summary, _, _ = evaluate_run(manifest, native, protocol=protocol)
    zero_summary, _, _ = evaluate_run(manifest, zero, protocol=protocol)
    identity = compare_runs(manifest, native, zero, protocol=protocol,
                            require_logit_identity=True)
    if not identity["identical_binary_masks"] or not identity["identical_float32_logits"]:
        raise RuntimeError("Zero-adapter parity is not exact")
    trained_summary, trained_comparison = None, None
    if trained is not None:
        trained_summary, _, _ = evaluate_run(manifest, trained, protocol=protocol)
        trained_comparison = compare_runs(manifest, native, trained, protocol=protocol)
    revision = source_revision()
    sources = {name: digest(ROOT / name) for name in SOURCE_PATHS}
    try:
        import torch
        version = {"python": sys.version, "torch": torch.__version__,
                   "cuda": torch.version.cuda,
                   "cuda_available": bool(torch.cuda.is_available())}
    except ImportError:
        version = {"python": sys.version, "torch": "unavailable"}

    input_artifacts = {}
    for label, root in (("native", native), ("zero", zero), ("trained", trained)):
        if root is None:
            continue
        input_artifacts[label] = collect_sha256(root, ["run.json"])
    receipt = {
        "status": "completed_research_verification_not_clinical_validation",
        "git_revision": revision,
        "source_sha256": sources,
        "protocol_sha256": contract["protocol_sha256"],
        "prompt_manifest_sha256": contract["prompt_manifest_sha256"],
        "dataset_manifest_sha256": contract["spec"]["dataset"]["manifest"]["sha256"],
        "backbone_sha256": contract["spec"]["checkpoint"]["sha256"],
        "environment": version,
        "run_receipt_sha256": input_artifacts,
        "scope": "oracle_GT_first_positive_mask; post_prompt_predictions_only",
        "baseline": native_summary,
        "zero_adapter": zero_summary,
        "identity_check": identity,
        "trained": trained_summary,
        "trained_comparison": trained_comparison,
        "caveats": [
            "This is a verification receipt, not proof that a trained adapter improves accuracy.",
            "A short DEV smoke run cannot establish full-cohort or clinical generalization.",
            "Patient/source independence must be audited outside the declared IDs.",
            "Comparisons exclude videos with no oracle-eligible foreground prompt.",
        ],
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(receipt, indent=2, allow_nan=False) + "\n",
                      encoding="utf-8")
    return receipt


def main(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--protocol", type=Path, required=True)
    p.add_argument("--native", type=Path, required=True)
    p.add_argument("--zero", type=Path, required=True)
    p.add_argument("--trained", type=Path)
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args(argv)
    record = seal_study(a.protocol, a.native, a.zero, a.output, trained=a.trained)
    print(json.dumps({
        "status": record["status"], "git_revision": record["git_revision"],
        "identity": record["identity_check"]["identical_float32_logits"],
        "trained_evaluated": record["trained"] is not None,
        "output": str(a.output),
    }, indent=2))


if __name__ == "__main__":
    main()
