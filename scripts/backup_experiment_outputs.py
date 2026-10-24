"""Copy closed experiment evidence to a second Drive folder and verify SHA256.

Uses shutil, not a new storage implementation. A same-account snapshot is not
an independent disk/account backup, and never proves Google Drive uploaded it.
"""

import argparse
import hashlib
import json
import os
from pathlib import Path
import shutil


def digest(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def backup(source, destination):
    source, destination = Path(source).resolve(), Path(destination).resolve()
    if not source.is_dir() or destination.exists() or destination.is_relative_to(source):
        raise ValueError("Require an existing source and a new destination outside it")
    destination.mkdir(parents=True)
    manifest = []
    skipped = []

    def files():
        for root, directories, names in os.walk(source, followlinks=False):
            for name in list(directories):
                if name.startswith(("ordered_frames_", "memory_probe_frames_")) or name in {
                    "wandb_cache_dir", "wandb_data_dir",
                }:
                    directories.remove(name)
                    skipped.append(str((Path(root) / name).relative_to(source)))
                elif (Path(root) / name).is_symlink():
                    directories.remove(name)
                    yield Path(root) / name
            for name in names:
                if name != ".DS_Store":
                    yield Path(root) / name

    for file in files():
        relative = file.relative_to(source)
        target = destination / relative
        target.parent.mkdir(parents=True, exist_ok=True)
        if file.is_symlink():
            link = os.readlink(file)
            target.symlink_to(link)
            if os.readlink(target) != link:
                raise RuntimeError(f"Symlink copy mismatch: {relative}")
            manifest.append({"path": str(relative), "bytes": 0, "symlink": link})
            continue
        before = (file.stat().st_size, file.stat().st_mtime_ns)
        shutil.copy2(file, target)
        checksum = digest(file)
        if before != (file.stat().st_size, file.stat().st_mtime_ns) or digest(target) != checksum:
            raise RuntimeError(f"Source changed or copy mismatch: {relative}")
        manifest.append({"path": str(relative), "bytes": before[0], "sha256": checksum})
        if len(manifest) % 100 == 0:
            print(f"Verified {len(manifest)} files", flush=True)
    manifest_path = destination / "backup_manifest.json"
    manifest_path.write_text(json.dumps(manifest, indent=2) + "\n")
    receipt = {
        "source_root": str(source), "backup_root": str(destination),
        "backup_manifest": str(manifest_path),
        "backup_manifest_sha256": digest(manifest_path),
        "backup_verified": True, "files": len(manifest),
        "bytes": sum(f["bytes"] for f in manifest),
        "independent_account_or_device": False,
        "excluded_regenerable_staging_and_caches": skipped,
        "drive_web_verified": False, "drive_folder_url": None,
        "limits": "Second mounted Drive copy only; upload must be verified separately",
    }
    (destination / "protection_receipt.json").write_text(json.dumps(receipt, indent=2) + "\n")
    print(json.dumps(receipt), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--source", required=True, type=Path)
    parser.add_argument("--destination", required=True, type=Path)
    args = parser.parse_args()
    backup(args.source, args.destination)


if __name__ == "__main__":
    main()
