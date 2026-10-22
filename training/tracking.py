import json
import os
from pathlib import Path


def start_wandb(
    project,
    entity,
    name,
    mode,
    base_url,
    config,
    output_dir,
    tags,
    *,
    resume_run_id=None,
):
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
        **({"id": resume_run_id, "resume": "must"} if resume_run_id else {}),
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


def finalize_tracking(run, output_dir, summary, names):
    """Use W&B's mapping-only summary API; confirm upload before completion."""
    output_dir = Path(output_dir)
    run.summary.update(dict(summary))
    log_results_artifact(run, output_dir, names)
    run.summary["wandb_artifact_upload_complete"] = True
    (output_dir / "delivery.json").write_text(
        json.dumps(
            {
                "completed": True,
                "wandb_artifact_upload_complete": True,
                "run_id": run.id,
            },
            indent=2,
        )
        + "\n"
    )
    (output_dir / "progress.json").write_text(json.dumps({"state": "completed"}) + "\n")
