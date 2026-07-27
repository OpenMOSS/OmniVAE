import os
import sys
from typing import Iterable, List, Optional

from omegaconf import OmegaConf

from omnivae_sync.training.train_sync import train
from omnivae_sync.training.train_utils import get_curr_time_w_random_shift
from omnivae_sync.utils.utils import cfg_sanity_check_and_patch


SUPPORTED_ACTIONS = {"train_avsync_model_vae"}


def set_env_variables() -> None:
    """Populate torch distributed variables when launched from Slurm."""
    if "RANK" in os.environ and "WORLD_SIZE" in os.environ:
        return
    if "SLURM_JOB_ID" in os.environ:
        os.environ["LOCAL_RANK"] = os.environ["SLURM_LOCALID"]
        os.environ["RANK"] = os.environ["SLURM_PROCID"]
        os.environ["WORLD_SIZE"] = os.environ["SLURM_NPROCS"]


def _normalize_cli(argv: Iterable[str]) -> List[str]:
    """Accept both OmegaConf dotlist args and common --key value syntax."""
    argv = list(argv)
    out: List[str] = []
    i = 0
    while i < len(argv):
        arg = argv[i]
        if arg.startswith("--"):
            stripped = arg[2:]
            if "=" in stripped:
                out.append(stripped)
                i += 1
            elif i + 1 < len(argv) and not argv[i + 1].startswith("--"):
                out.append(f"{stripped}={argv[i + 1]}")
                i += 2
            else:
                out.append(f"{stripped}=true")
                i += 1
        else:
            out.append(arg)
            i += 1
    return out


def register_resolvers() -> None:
    try:
        OmegaConf.register_new_resolver("add", lambda *args: sum(args), replace=True)
    except TypeError:
        try:
            OmegaConf.clear_resolver("add")
        except Exception:
            pass
        OmegaConf.register_new_resolver("add", lambda *args: sum(args))


def get_config(argv: Optional[Iterable[str]] = None):
    register_resolvers()
    cli = OmegaConf.from_dotlist(_normalize_cli(sys.argv[1:] if argv is None else argv))
    if "config" not in cli or cli.config is None:
        raise ValueError("Missing config path. Use: config=./configs/sync_24fps_nonspeech_vae.yaml")
    cfg_yml = OmegaConf.load(cli.config)
    cfg = OmegaConf.merge(cfg_yml, cli)
    if "start_time" not in cfg or cfg.start_time is None:
        cfg.start_time = get_curr_time_w_random_shift()
    OmegaConf.resolve(cfg)
    return cfg


def main(cfg=None) -> None:
    cfg = get_config() if cfg is None else cfg
    if cfg.action not in SUPPORTED_ACTIONS:
        raise NotImplementedError(
            f"Unsupported action={cfg.action!r}. OmniVAE-Sync release supports: {sorted(SUPPORTED_ACTIONS)}"
        )
    set_env_variables()
    cfg_sanity_check_and_patch(cfg)
    train(cfg)


if __name__ == "__main__":
    if os.environ.get("DEBUG", "0") == "DEBUG" and os.environ.get("RANK") == "0":
        from omnivae_sync.utils.helpers import waiting_for_debug

        waiting_for_debug(
            os.environ.get("DEBUG_HOST", "localhost"),
            int(os.environ.get("DEBUG_PORT", "32431")),
        )
    main()
