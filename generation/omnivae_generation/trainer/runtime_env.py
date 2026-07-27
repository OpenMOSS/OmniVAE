from __future__ import annotations

import os
import re
import tempfile
from pathlib import Path


def _cache_root() -> Path:
    return Path(os.environ.get("XDG_CACHE_HOME", Path.home() / ".cache")).expanduser()


def _default_hf_home() -> Path:
    return Path(os.environ.get("OMNIGEN_HF_HOME", _cache_root() / "huggingface")).expanduser()


def _default_torch_home() -> Path:
    return Path(os.environ.get("OMNIGEN_TORCH_HOME", _cache_root() / "torch")).expanduser()


def _default_torchinductor_cache_dir() -> Path:
    return Path(os.environ.get("OMNIGEN_TORCHINDUCTOR_CACHE_DIR", _cache_root() / "torchinductor")).expanduser()


def ensure_torch_home() -> str:
    resolved = str(_default_torch_home().expanduser())
    os.environ.setdefault("TORCH_HOME", resolved)
    return os.environ["TORCH_HOME"]


def _torchinductor_builtin_default_cache_dir() -> str:
    # Match torch._inductor.runtime.cache_dir_utils.default_cache_dir() without importing torch
    # before we get a chance to override launcher-provided defaults.
    import getpass

    sanitized_username = re.sub(r'[\\/:*?"<>|]', "_", getpass.getuser())
    return os.path.join(tempfile.gettempdir(), f"torchinductor_{sanitized_username}")


def ensure_torchinductor_cache_dir() -> str:
    resolved = str(_default_torchinductor_cache_dir().expanduser())
    existing = os.environ.get("TORCHINDUCTOR_CACHE_DIR")
    if existing is None or str(Path(existing).expanduser()) == _torchinductor_builtin_default_cache_dir():
        os.environ["TORCHINDUCTOR_CACHE_DIR"] = resolved
    return os.environ["TORCHINDUCTOR_CACHE_DIR"]


def ensure_hf_home() -> str:
    resolved = str(_default_hf_home().expanduser())
    os.environ.setdefault("HF_HOME", resolved)
    if "OMNIGEN_HF_HUB_OFFLINE" in os.environ:
        os.environ.setdefault("HF_HUB_OFFLINE", os.environ["OMNIGEN_HF_HUB_OFFLINE"])
    ensure_torch_home()
    ensure_torchinductor_cache_dir()
    return os.environ["HF_HOME"]
