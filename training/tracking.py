import json
import os
from pathlib import Path


def download_forge_run_file(file, destination):
    """Fetch a Forge run file with explicit auth and its advertised checksum.

    Kaggle's W&B service downloader returned a project-not-found 404 for this
    same URL; the authenticated HTTP endpoint returns the verified payload.
    """
    import base64
    import hashlib
    from urllib.parse import urlsplit

    import requests

    url = urlsplit(file.url)
    if url.scheme != "https" or url.netloc != "forge.coreweave.com":
        raise ValueError("Refusing to send Forge credentials to another origin")
    key = os.environ.get("WANDB_API_KEY")
    if not key:
        raise RuntimeError("Forge run-file download requires WANDB_API_KEY")
    path = (Path(destination).resolve() / file.name).resolve()
    if not path.is_relative_to(Path(destination).resolve()):
        raise ValueError("Unsafe run-file path")
    response = requests.get(
        file.url,
        headers={"Authorization": f"Bearer {key}"},
        timeout=30,
        # Requests strips Authorization on cross-origin redirects to the signed
        # storage URL. Verify the final payload against W&B's size and checksum.
        allow_redirects=True,
    )
    response.raise_for_status()
    content = response.content
    if (
        response.status_code != 200
        or len(content) != file.size
        or base64.b64encode(hashlib.md5(content).digest()).decode() != file.md5
    ):
        raise ValueError("Forge run-file checksum or size mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(content)
    return path


def latest_results_artifact(source, required_any=()):
    """Ignore automatic W&B history/config artifacts when loading results."""
    candidates = []
    for artifact in source.logged_artifacts():
        if artifact.type != "experiment-results":
            continue
        if required_any and not any(
            name.endswith(suffix)
            for name in artifact.manifest.entries
            for suffix in required_any
        ):
            continue
        candidates.append(artifact)
    if not candidates:
        raise ValueError(
            "No experiment-results artifact contains the required evidence"
        )
    return max(candidates, key=lambda artifact: artifact.created_at)


def start_wandb(project, entity, name, mode, base_url, config, output_dir, tags):
    if mode == "online" and not os.environ.get("WANDB_API_KEY"):
        raise RuntimeError(
            "W&B online mode requires WANDB_API_KEY. Attach the Kaggle secret or "
            "pass --wandb-mode offline/disabled explicitly."
        )
    if base_url:
        os.environ["WANDB_BASE_URL"] = base_url
    import wandb

    wandb_dir = Path(output_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    run = wandb.init(
        project=project,
        entity=entity,
        name=name,
        mode=mode,
        config=config,
        dir=str(wandb_dir),
        tags=tags,
    )
    tracking = {
        "mode": mode,
        "entity": entity,
        "project": project,
        "run_id": run.id,
        "url": run.url,
    }
    (Path(output_dir) / "tracking.json").write_text(
        json.dumps(tracking, indent=2) + "\n"
    )
    return run


def log_results_artifact(run, output_dir, names):
    import wandb

    output_dir = Path(output_dir)
    artifact = wandb.Artifact(f"utility-results-{run.id}", type="experiment-results")
    for name in names:
        artifact.add_file(str(output_dir / name), name=name)
    run.log_artifact(artifact).wait()


def log(run, prefix, values, step):
    if run:
        run.log(
            {
                f"{prefix}/{key}": value
                for key, value in values.items()
                if value is not None
            },
            step=step,
        )


def finish(run):
    if run:
        run.finish()
