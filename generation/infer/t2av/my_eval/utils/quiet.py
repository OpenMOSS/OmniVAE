"""Helpers for keeping the dispatcher tty readable.

* ``fd_redirect(log_file)`` -- redirects fd 1/2 (so subprocess children inherit
  the redirection) to a file for the duration of the block. Use it around
  noisy model.forward() calls or any subprocess that prints unconditionally.

* ``open_rank_log(target_dir, kind, rank)`` -- canonical path of the per-task
  per-rank log file. Idempotent (parent dir is created on first call).

* ``silence_known_warnings()`` -- one-shot filter for the dozens of
  Future/UserWarnings that torch / transformers / timm / pyiqa / audiobox emit
  during normal operation. Doesn't hide actual errors.
"""
from __future__ import annotations

import contextlib
import os
import sys
import warnings
from pathlib import Path


@contextlib.contextmanager
def fd_redirect(log_file):
    """Send fd 1 + fd 2 (and therefore any subprocess output) to ``log_file``.

    The dispatcher's own ``log()`` calls keep their original fd because they
    go through Python's ``print(..., file=sys.stdout)`` only outside the
    context (we restore fd 1/2 in ``finally``).
    """
    sys.stdout.flush()
    sys.stderr.flush()
    saved_stdout = os.dup(1)
    saved_stderr = os.dup(2)
    log_fd = log_file.fileno()
    try:
        os.dup2(log_fd, 1)
        os.dup2(log_fd, 2)
        yield
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        os.dup2(saved_stdout, 1)
        os.dup2(saved_stderr, 2)
        os.close(saved_stdout)
        os.close(saved_stderr)


def open_rank_log(target_dir: Path, kind: str, rank: int) -> Path:
    log_dir = target_dir / "logs"
    log_dir.mkdir(parents=True, exist_ok=True)
    return log_dir / f"{kind}.rank{rank}.log"


def silence_known_warnings() -> None:
    """Filter the warnings the vendored toolchain emits on every import / call.

    None of these are actionable for the end user; they all stem from drift
    between the vendored 2022-era code and the 2024+ torch / transformers /
    pyiqa packages.
    """
    warnings.filterwarnings("ignore", category=FutureWarning)
    warnings.filterwarnings("ignore", category=DeprecationWarning)
    # User warnings are noisier per category; allow real errors through by
    # only silencing the well-known modules.
    for mod in (
        "transformers", "torch", "torchaudio", "torchvision",
        "timm", "pyiqa", "pkg_resources", "audiobox_aesthetics",
        "imagebind", "pytorchvideo",
    ):
        warnings.filterwarnings("ignore", category=UserWarning, module=fr"{mod}.*")
    # Subprocesses inherit this via PYTHONWARNINGS (the dispatcher shell also
    # sets it, this is the in-process belt-and-braces).
    os.environ.setdefault(
        "PYTHONWARNINGS",
        "ignore::FutureWarning,ignore::DeprecationWarning,ignore::UserWarning",
    )
    # Transformers nags about TRANSFORMERS_CACHE every import; silence the
    # actual env var since HF_HOME already supersedes it.
    os.environ.pop("TRANSFORMERS_CACHE", None)
