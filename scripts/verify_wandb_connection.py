#!/usr/bin/env python3
"""Fail-fast W&B connectivity check for remote experiment runners."""

from __future__ import annotations

import argparse
import json
import os

import requests


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--name", default="ade-seg-connectivity-check")
    parser.add_argument("--check-only", action="store_true", help="Validate access without creating a W&B run")
    args = parser.parse_args()

    api_key = os.environ.get("WANDB_API_KEY")
    base_url = os.environ.get("WANDB_BASE_URL", "https://api.wandb.ai").rstrip("/")
    project = os.environ.get("WANDB_PROJECT", "ade-seg")
    entity = os.environ.get("WANDB_ENTITY", "adenoid-hypertrophy")
    if not api_key:
        raise RuntimeError("WANDB_API_KEY is not set")

    query = {
        "query": (
            "query($entity: String!, $project: String!) { "
            "project(name: $project, entityName: $entity) { name } }"
        ),
        "variables": {"entity": entity, "project": project},
    }
    response = requests.post(
        f"{base_url}/graphql",
        json=query,
        headers={"Authorization": f"Bearer {api_key}"},
        timeout=30,
    )
    response.raise_for_status()
    result = response.json()
    if result.get("errors") or not result.get("data", {}).get("project"):
        raise RuntimeError(f"W&B GraphQL validation failed: {result}")

    if args.check_only:
        print(json.dumps({"state": "connected", "entity": entity, "project": project}))
        return

    import wandb

    run = wandb.init(
        project=project,
        entity=entity,
        name=args.name,
        tags=["utility", "kaggle", "connectivity", "non-claim-bearing"],
        settings=wandb.Settings(init_timeout=45),
    )
    run.log({"status/connected": 1})
    run.finish()
    print(json.dumps({"state": "complete", "entity": entity, "project": project}))


if __name__ == "__main__":
    main()
