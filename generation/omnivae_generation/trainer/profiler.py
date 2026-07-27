from __future__ import annotations

import atexit
import time
from pathlib import Path

import torch

from omnivae_generation.trainer.utils import ensure_dir


_ACTIVITY_MAP = {
    "cpu": torch.profiler.ProfilerActivity.CPU,
    "cuda": torch.profiler.ProfilerActivity.CUDA,
}


class TorchProfilerController:
    def __init__(self, profiler=None, trace_dir: Path | None = None):
        self._profiler = profiler
        self.trace_dir = trace_dir
        self.enabled = profiler is not None
        self._started = False
        self._stopped = False
        self._atexit_registered = False

    def start(self) -> None:
        if not self.enabled or self._started:
            return
        self._profiler.start()
        self._started = True
        if not self._atexit_registered:
            atexit.register(self.stop)
            self._atexit_registered = True

    def step(self) -> None:
        if not self.enabled or not self._started or self._stopped:
            return
        self._profiler.step()

    def stop(self) -> None:
        if not self.enabled or not self._started or self._stopped:
            return
        self._profiler.stop()
        self._stopped = True


def build_profiler(accelerator, config: dict, output_dir: Path) -> TorchProfilerController:
    profiler_config = config.get("profiler", {})
    if not profiler_config.get("enabled", False):
        return TorchProfilerController()

    if not accelerator.is_local_main_process and not profiler_config.get("all_ranks", False):
        return TorchProfilerController()

    trace_dir = _resolve_trace_dir(profiler_config, output_dir)
    trace_prefix = time.strftime("%Y%m%d-%H%M%S")
    activities = _resolve_activities(profiler_config)
    schedule = torch.profiler.schedule(
        wait=int(profiler_config["wait"]),
        warmup=int(profiler_config["warmup"]),
        active=int(profiler_config["active"]),
        repeat=int(profiler_config["repeat"]),
        skip_first=int(profiler_config["skip_first"]),
    )

    def on_trace_ready(prof) -> None:
        trace_path = trace_dir / (
            f"{trace_prefix}-rank{accelerator.process_index:02d}-step{prof.step_num:08d}.json"
        )
        prof.export_chrome_trace(str(trace_path))

    profiler = torch.profiler.profile(
        activities=activities,
        schedule=schedule,
        on_trace_ready=on_trace_ready,
        record_shapes=bool(profiler_config.get("record_shapes", False)),
        profile_memory=bool(profiler_config.get("profile_memory", False)),
        with_stack=bool(profiler_config.get("with_stack", False)),
        with_flops=bool(profiler_config.get("with_flops", False)),
        with_modules=bool(profiler_config.get("with_modules", False)),
    )
    return TorchProfilerController(profiler=profiler, trace_dir=trace_dir)


def _resolve_trace_dir(profiler_config: dict, output_dir: Path) -> Path:
    configured = profiler_config.get("trace_dir")
    if configured:
        candidate = Path(configured).expanduser()
        if not candidate.is_absolute():
            candidate = output_dir / candidate
    else:
        candidate = output_dir / "profiler"
    return ensure_dir(candidate.resolve())


def _resolve_activities(profiler_config: dict) -> list[torch.profiler.ProfilerActivity]:
    resolved = []
    configured = profiler_config.get("activities") or ["cpu", "cuda"]
    for item in configured:
        normalized = str(item).strip().lower()
        if normalized not in _ACTIVITY_MAP:
            raise ValueError(f"Unsupported profiler activity: {item}")
        if normalized == "cuda" and not torch.cuda.is_available():
            continue
        resolved.append(_ACTIVITY_MAP[normalized])

    if not resolved:
        resolved.append(torch.profiler.ProfilerActivity.CPU)
    return resolved
