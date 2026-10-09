import os
from pathlib import Path


def start_wandb(project, entity, name, mode, base_url, config, output_dir, tags):
    if base_url:
        os.environ["WANDB_BASE_URL"] = base_url
    import wandb

    wandb_dir = Path(output_dir) / "wandb"
    wandb_dir.mkdir(parents=True, exist_ok=True)
    return wandb.init(project=project, entity=entity, name=name, mode=mode,
                      config=config, dir=str(wandb_dir), tags=tags)


def log(run, prefix, values, step):
    if run:
        run.log({f"{prefix}/{key}": value for key, value in values.items() if value is not None}, step=step)


def finish(run):
    if run:
        run.finish()
