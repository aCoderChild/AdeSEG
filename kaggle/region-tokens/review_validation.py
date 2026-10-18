"""Reuse the pinned runner for validation or explicit PolypGen diagnostic phases."""

import json
import os
from pathlib import Path
import urllib.request


REVISION = "17d74aa6a2864df411fc5927e89b1d1d46cf5bf3"
print("[review] kernel started; pinned revision=" + REVISION, flush=True)
os.environ["ADESEG_REVISION"] = REVISION
os.environ["REGION_E1_RUN_ID"] = "7orfgsk2"
url = (
    "https://raw.githubusercontent.com/aCoderChild/AdeSEG/"
    + REVISION + "/kaggle/region-tokens/runner.py"
)
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read().decode("utf-8")

phase = os.environ.get("REGION_REVIEW_PHASE", "validation")
if phase not in {"validation", "frame_matched", "corruption", "corruption_validation"}:
    raise ValueError("Unknown REGION_REVIEW_PHASE")
if phase != "validation":
    # Source-index partitions, not selected by observed effects. seq1/7 have
    # no labeled foreground. The small seq3 validation is not pooled here.
    batches = (
        ["seq2", "seq3", "seq4", "seq5", "seq6", "seq8", "seq9", "seq10", "seq11", "seq12"],
        ["seq13", "seq14", "seq15", "seq16", "seq17", "seq18", "seq19", "seq20", "seq21", "seq22", "seq23"],
    )
    # One selected condition per corruption submission keeps each notebook
    # within a bounded workload. Each submission still has independent E0.
    condition = "c0" if phase == "frame_matched" else os.environ.get("REGION_CORRUPTION", "c1")
    if condition not in {"c0", "c1", "c2", "natural"}:
        raise ValueError("Unknown REGION_CORRUPTION")
    conditions = [condition]
    if phase == "corruption_validation":
        batches = (["seq3"],)
        conditions = ["c0", "c1", "c2", "natural"]
    for condition in conditions:
        for batch, sequences in enumerate(batches, 1):
            job = dict(
                stage="E2" if phase == "frame_matched" else "E3",
                dataset="polypgen", split="test", sequences=sequences,
                mode="single_step", intervention="attention_mask", memory_condition=condition,
                band=1, thresholds=[0.1, 0.9], label_ids=[1],
                boundary_definition="mixed_coverage",
                spatial_match="frame_distance" if phase == "frame_matched" else "pooled_distance",
            )
            output = f"region_tokens_{phase}_{condition}_batch{batch:02d}_v1"
            os.environ.pop("REGION_E0_TARGET_FRAME", None)
            os.environ.pop("REGION_E0_SEQUENCE", None)
            os.environ.update(REGION_STAGE="study_shard", REGION_JOB=json.dumps(job), REGION_OUTPUT_NAME=output)
            print("[diagnostics] starting explicit batch", json.dumps(job), flush=True)
            exec(compile(source, "region_tokens_runner.py", "exec"), {"__name__": "__main__"})
            status = json.loads((Path("/kaggle/working") / output / "status.json").read_text())
            assert status.get("state") == "complete", "Diagnostic batch failed; stop"
    print("[diagnostics] selected phase complete; E6 remains deferred; full study incomplete", flush=True)
    raise SystemExit(0)

os.environ.pop("REGION_E0_TARGET_FRAME", None)
os.environ.update(
    REGION_STAGE="E0", REGION_E0_SEQUENCE="seq3",
    REGION_OUTPUT_NAME="region_tokens_review_e0_v1",
)
exec(compile(source, "region_tokens_runner.py", "exec"), {"__name__": "__main__"})
checks = [json.loads(line) for line in Path(
    "/kaggle/working/region_tokens_review_e0_v1/E0/sanity.jsonl"
).read_text().splitlines() if line]
assert checks and all(c["passed"] for c in checks), "Fresh GPU E0 failed"
os.environ.pop("REGION_E0_SEQUENCE", None)

for name, boundary, spatial in (
    ("contour", "contour_union", "pooled_distance"),
    ("frame_distance", "mixed_coverage", "frame_distance"),
):
    output = "region_tokens_review_" + name + "_seq3_v1"
    job = dict(
        stage="E2", dataset="polypgen", split="test", sequences=["seq3"],
        mode="single_step", intervention="attention_mask", memory_condition="c0",
        band=1, thresholds=[0.1, 0.9], label_ids=[1],
        boundary_definition=boundary, spatial_match=spatial,
    )
    os.environ.update(
        REGION_STAGE="study_shard", REGION_JOB=json.dumps(job),
        REGION_OUTPUT_NAME=output,
    )
    print("[review] validating separate sensitivity", json.dumps(job), flush=True)
    exec(compile(source, "region_tokens_runner.py", "exec"), {"__name__": "__main__"})
    folder = Path("/kaggle/working") / output / "study" / "seq3__object1"
    rows = [json.loads(line) for line in (folder / "frames.jsonl").read_text().splitlines() if line]
    assert rows and all(
        r["boundary_definition"] == boundary and r["spatial_match"] == spatial
        for r in rows
    ), "Sensitivity provenance mismatch"
    assert any(r["state"] == "complete" and r["category"] == "spatial_random" for r in rows), "No feasible spatial control"
    checks = [json.loads(line) for line in (folder / "sanity.jsonl").read_text().splitlines() if line]
    assert checks and all(c["passed"] for c in checks), "Sensitivity exactness failed"
    print("[review] validated " + name + "; full sensitivity coverage remains pending", flush=True)

print("[review] GPU validation complete; not a full E0-E6 result", flush=True)
