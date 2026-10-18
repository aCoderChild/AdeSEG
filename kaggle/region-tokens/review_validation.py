"""Separate GPU validation of reviewed region-token controls; not a full study."""

import json
import os
from pathlib import Path
import urllib.request


REVISION = "752d6c0b6cf53ddef2bc70da2667246c4dd2667a"
print("[review] kernel started; pinned revision=" + REVISION, flush=True)
os.environ["ADESEG_REVISION"] = REVISION
os.environ["REGION_E1_RUN_ID"] = "7orfgsk2"
url = (
    "https://raw.githubusercontent.com/aCoderChild/AdeSEG/"
    + REVISION + "/kaggle/region-tokens/runner.py"
)
with urllib.request.urlopen(url, timeout=60) as response:
    source = response.read().decode("utf-8")

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
