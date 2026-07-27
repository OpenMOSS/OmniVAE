from __future__ import annotations

import hashlib
import os
from pathlib import Path

import torch
from accelerate import Accelerator


def get_weight_dtype(accelerator: Accelerator) -> torch.dtype:
    if accelerator.mixed_precision == "fp16":
        return torch.float16
    if accelerator.mixed_precision == "bf16":
        return torch.bfloat16
    return torch.float32


def configure_wandb_env(config: dict) -> None:
    if config["accelerate"]["log_with"] != "wandb":
        return

    base_url = config.get("wandb", {}).get("base_url")
    if base_url:
        os.environ.setdefault("WANDB_BASE_URL", base_url)


def _derive_wandb_run_id(config: dict) -> str:
    output_dir = str(Path(config["experiment"]["output_dir"]).expanduser().resolve())
    project = str(config.get("wandb", {}).get("project") or config["experiment"]["name"])
    digest = hashlib.sha1(f"{project}:{output_dir}".encode("utf-8")).hexdigest()
    return digest[:16]


def get_tracker_init(project_name: str, config: dict, logging_dir: Path):
    init_kwargs = {}
    if config["accelerate"]["log_with"] == "wandb":
        wandb_config = config.get("wandb", {})
        project_name = wandb_config.get("project") or project_name
        run_name = wandb_config.get("run_name") or config["experiment"]["name"]
        run_id = wandb_config.get("run_id") or _derive_wandb_run_id(config)
        resume = wandb_config.get("resume", "allow")
        init_kwargs = {
            "wandb": {
                "entity": wandb_config.get("entity"),
                "name": run_name,
                "id": run_id,
                "resume": resume,
                "dir": str(logging_dir),
            }
        }
        init_kwargs["wandb"] = {key: value for key, value in init_kwargs["wandb"].items() if value is not None}
    return project_name, init_kwargs
