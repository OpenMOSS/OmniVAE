import os
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import torch

_project_root = os.path.dirname(
    os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
)
_PROJECT_ROOT_PATH = Path(_project_root).resolve()


def _default_path_vars() -> Dict[str, str]:
    repo_root = Path(os.environ.get("OMNIVAE_REPO_ROOT", _project_root)).resolve()
    return {
        "OMNIVAE_REPO_ROOT": str(repo_root),
        "OMNIVAE_CKPT_ROOT": os.environ.get("OMNIVAE_CKPT_ROOT", str(repo_root / "ckpts")),
        "OMNIVAE_DATA_ROOT": os.environ.get("OMNIVAE_DATA_ROOT", str(repo_root / "data")),
        "OMNIVAE_EXP_ROOT": os.environ.get("OMNIVAE_EXP_ROOT", str(repo_root / "exp")),
        "OMNIVAE_SEMANTIC_MODEL": os.environ.get(
            "OMNIVAE_SEMANTIC_MODEL", str(repo_root / "ckpts" / "qwen3_avencoder_service")
        ),
    }


def _expand_known_path_vars(value: str) -> str:
    """Expand user/env vars while preserving non-env OmegaConf-style references."""
    defaults = _default_path_vars()

    def replace(match: re.Match[str]) -> str:
        name = match.group(1) or match.group(2)
        return os.environ.get(name, defaults.get(name, match.group(0)))

    value = re.sub(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}|\$([A-Za-z_][A-Za-z0-9_]*)", replace, value)
    return os.path.expanduser(os.path.expandvars(value))


_PATH_KEY_TOKENS = (
    "path",
    "paths",
    "dir",
    "root",
    "file",
    "checkpoint",
    "ckpt",
    "yaml",
)
_NON_PATH_KEY_TOKENS = ("api_url", "url")
_LOCAL_PATH_EXTENSIONS = (
    ".yaml",
    ".yml",
    ".json",
    ".jsonl",
    ".pth",
    ".pt",
    ".ckpt",
    ".safetensors",
    ".bin",
)


def _is_path_key(key: Optional[str], parent_key: Optional[str]) -> bool:
    joined = " ".join(k for k in (parent_key, key) if k).lower()
    if not joined:
        return False
    if any(token in joined for token in _NON_PATH_KEY_TOKENS):
        return False
    return any(token in joined for token in _PATH_KEY_TOKENS)


def _looks_like_local_path(value: str) -> bool:
    stripped = value.strip()
    if not stripped:
        return False
    if stripped.startswith(("http://", "https://")):
        return False
    if stripped.startswith(("/", "./", "../", "~")):
        return True
    if any(stripped.endswith(ext) for ext in _LOCAL_PATH_EXTENSIONS):
        return True
    return False


def resolve_path_value(value: Any, *, base_dir: Optional[Path] = None) -> Any:
    """Resolve a local path-like scalar while leaving URLs and HF repo IDs intact."""
    if not isinstance(value, str):
        return value
    expanded = _expand_known_path_vars(value)
    if not _looks_like_local_path(expanded):
        return expanded
    path = Path(expanded)
    if path.is_absolute():
        return str(path)
    base = base_dir or _PROJECT_ROOT_PATH
    return str((base / path).resolve())


def resolve_config_paths(cfg: Any, *, base_dir: Optional[Path] = None, key: Optional[str] = None,
                         parent_key: Optional[str] = None) -> Any:
    """Recursively expand local paths in a plain YAML config dict/list."""
    if isinstance(cfg, dict):
        return {
            k: resolve_config_paths(v, base_dir=base_dir, key=str(k), parent_key=key)
            for k, v in cfg.items()
        }
    if isinstance(cfg, list):
        return [
            resolve_config_paths(item, base_dir=base_dir, key=key, parent_key=parent_key)
            for item in cfg
        ]
    if isinstance(cfg, str):
        expanded = _expand_known_path_vars(cfg)
        if _is_path_key(key, parent_key) or _looks_like_local_path(expanded):
            return resolve_path_value(expanded, base_dir=base_dir)
        return expanded
    return cfg


def exists(val):
    return val is not None


def accum_log(log, new_logs):
    for key, new_value in new_logs.items():
        old_value = log.get(key, 0.)
        log[key] = old_value + new_value
    return log


def _resolve_cfg_reference(raw_value: Any, cfg: Optional[Dict[str, Any]]) -> Any:
    """Resolve ${a.b.c} style references against plain dict config."""
    if cfg is None or not isinstance(raw_value, str):
        return raw_value
    match = re.fullmatch(r"\$\{([^}]+)\}", raw_value.strip())
    if not match:
        return raw_value
    cur: Any = cfg
    ref_path = match.group(1).split(".")
    for key in ref_path:
        if not isinstance(cur, dict) or key not in cur:
            raise ValueError(f"Cannot resolve config reference '{raw_value}'")
        cur = cur[key]
    return cur


def _parse_positive_int_list(
    raw_value: Any,
    *,
    default_value: int,
    field_name: str,
    cfg: Optional[Dict[str, Any]] = None,
) -> List[int]:
    """
    Parse supported int-list formats into a deduplicated List[int]:
      - int (e.g. 64)
      - comma-separated string (e.g. "64,256")
      - list/tuple (e.g. [64, 256])
      - ${...} reference when cfg is provided
    """
    value = _resolve_cfg_reference(raw_value, cfg)
    if value is None:
        value = default_value

    if isinstance(value, int):
        parsed = [value]
    elif isinstance(value, str):
        tokens = [tok.strip() for tok in value.split(",") if tok.strip()]
        if not tokens:
            raise ValueError(f"{field_name} must be a positive int or non-empty int list, got '{value}'")
        try:
            parsed = [int(tok) for tok in tokens]
        except ValueError as err:
            raise ValueError(f"{field_name} contains non-integer token(s): '{value}'") from err
    elif isinstance(value, (list, tuple)):
        parsed = []
        for item in value:
            item = _resolve_cfg_reference(item, cfg)
            try:
                parsed.append(int(item))
            except (TypeError, ValueError) as err:
                raise ValueError(f"{field_name} contains non-integer item: {item}") from err
    else:
        raise ValueError(f"{field_name} must be int, string, or list/tuple; got {type(value)}")

    deduped: List[int] = []
    seen = set()
    for v in parsed:
        if v <= 0:
            raise ValueError(f"{field_name} expects positive integers, got {v}")
        if v not in seen:
            deduped.append(v)
            seen.add(v)

    if not deduped:
        raise ValueError(f"{field_name} cannot be empty")
    return deduped


def _format_int_list_suffix(values: List[int]) -> str:
    """Format [64,256] -> '64-256' for filenames."""
    return "-".join(str(v) for v in values)


_DTYPE_MAP = {
    'bf16': torch.bfloat16, 'bfloat16': torch.bfloat16,
    'fp16': torch.float16, 'float16': torch.float16,
    'fp32': torch.float32, 'float32': torch.float32,
}


def _parse_dtype(dtype_str: Optional[str]) -> Optional[torch.dtype]:
    """Parse a dtype string to torch.dtype; returns None when input is None."""
    if dtype_str is None:
        return None
    if dtype_str not in _DTYPE_MAP:
        raise ValueError(f"Unknown dtype: {dtype_str!r}. Expected one of {list(_DTYPE_MAP.keys())}")
    return _DTYPE_MAP[dtype_str]


def find_latest_checkpoint(results_folder: Path, prefix: str = "Trainer_") -> Optional[str]:
    """按目录名末尾的步数（数值）排序，返回 step 最大的 ckpt 目录"""
    ckpts = list(results_folder.glob(f'{prefix}*'))
    if not ckpts:
        return None

    def _step_key(p: Path) -> int:
        m = re.search(r'(\d+)\s*$', p.name)
        return int(m.group(1)) if m else -1

    ckpts.sort(key=_step_key)
    return str(ckpts[-1])


def checkpoint_num_steps(checkpoint_path: str) -> int:
    """从 checkpoint 路径提取步数"""
    results = re.findall(r'\d+', str(checkpoint_path))
    return int(results[-1]) if results else 0
