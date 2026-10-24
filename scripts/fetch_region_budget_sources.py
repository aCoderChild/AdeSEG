"""Retrieve the smaller native geometry archives for offline E1; no inference.

Reuse W&B artifact downloads and compare every predicted PNG with the already
verified native SAM2 archive. Only native assignments/capture receipts are
extracted; original Drive files are never overwritten.
"""

import argparse
import base64
import hashlib
import io
import json
import os
from pathlib import Path
import sys
import zipfile

import numpy as np
from PIL import Image

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.memory_token_io import digest, validate_output  # noqa: E402


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output-dir", required=True, type=Path)
    parser.add_argument("--native-archive-dir", required=True, type=Path)
    args = parser.parse_args()
    output = validate_output(args.output_dir)
    output.mkdir(parents=True, exist_ok=False)
    os.environ["WANDB_CACHE_DIR"] = str(output / "wandb_cache")
    os.environ["WANDB_DATA_DIR"] = str(output / "wandb_data")
    import wandb
    api = wandb.Api(overrides={"base_url": "https://forge.coreweave.com/api/wandb"}, timeout=60)
    run = api.run("adenoid-hypertrophy/ade-seg/oq7ekeyo")
    if run.state != "finished":
        raise RuntimeError("Native geometry source is not finished")
    source = json.loads((args.native_archive_dir / "opportunities.json").read_text())
    native_sources = {r["sequence"]: r for r in source["methods"]["sam2"]["archive_provenance"]}
    receipts = []
    for artifact in run.logged_artifacts():
        for file in artifact.files():
            if not file.name.startswith("seq") or not file.name.endswith(".zip"):
                continue
            sequence = Path(file.name).stem
            entry = artifact.get_entry(file.name)
            path = Path(entry.download(root=str(output / "archives"), skip_cache=True))
            if base64.b64encode(hashlib.md5(path.read_bytes()).digest()).decode() != entry.digest:
                raise RuntimeError("Downloaded geometry archive digest differs")
            baseline = args.native_archive_dir / "remote_sources/sam2" / f"{sequence}.zip"
            if digest(baseline) != native_sources[sequence]["sha256"]:
                raise RuntimeError("Baseline archive checksum differs")
            folder = output / "assignments" / sequence
            folder.mkdir(parents=True, exist_ok=False)
            with zipfile.ZipFile(path) as geometry, zipfile.ZipFile(baseline) as native:
                if geometry.testzip():
                    raise RuntimeError("Geometry archive CRC failure")
                rows = [json.loads(line) for line in native.read(f"{sequence}/frames.jsonl").splitlines()]
                checked = 0
                for row in rows:
                    # Native SAM2 archive uses the manifest-ordered index.
                    left = np.asarray(Image.open(io.BytesIO(geometry.read(f"{sequence}/native_mask_{row['frame']:06d}.png")))) > 0
                    right = np.asarray(Image.open(io.BytesIO(native.read(f"{sequence}/frame_{row['frame']:06d}.png")))) > 0
                    if not np.array_equal(left, right):
                        raise RuntimeError("Geometry/native baseline prediction masks differ")
                    checked += 1
                for name in geometry.namelist():
                    member = Path(name)
                    if member.parent.as_posix() == sequence and (member.name == "assignment_frames.jsonl" or (member.name.startswith("assignments_") and member.suffix == ".pt")):
                        (folder / member.name).write_bytes(geometry.read(name))
            receipts.append({"sequence": sequence, "artifact": artifact.qualified_name,
                             "md5": entry.digest, "sha256": digest(path), "native_pngs_identical": checked})
            (output / "retrieval.json").write_text(json.dumps({"complete": False, "sources": receipts}, indent=2)+"\n")
            print("Recovered", sequence, "matching native predictions", checked, flush=True)
    if len(receipts) != 21 or sum(r["native_pngs_identical"] for r in receipts) != 2012:
        raise RuntimeError("Incomplete recovered native cohort")
    (output / "retrieval.json").write_text(json.dumps({"complete": True, "sources": receipts}, indent=2)+"\n")


if __name__ == "__main__":
    main()
