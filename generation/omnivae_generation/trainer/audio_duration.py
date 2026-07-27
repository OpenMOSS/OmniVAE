"""Per-prompt target-duration estimators shared by training-time validation
and the off-line inference sweep.

The output of the estimator becomes the ``duration: X.Xs`` suffix that the
text encoder sees for each prompt. The latent length is *separately* sized
with a fixed budget (``validation.duration_seconds`` / ``--duration-seconds``);
the estimator only controls the conditioning signal, not the tensor shape.

Strategies
----------
``fixed``
    Returns the same value for every prompt. This was the legacy behavior of
    ``omnivae_generation.trainer.audio_validation.run_audio_validation`` (every prompt got "30.0s"
    no matter how long the text was).

``auto``
    Whitespace word-count divided by ``words_per_second``, plus a margin.
    Cheap and decent on English; degrades on no-space scripts.

``f5``
    Port of F5-TTS's ``utils_infer.py`` formula (see comments in
    :func:`make_duration_estimator`). Uses UTF-8 byte length and a global
    "bytes per second" prior since pure T2A has no reference clip to derive
    a per-utterance frames-per-byte ratio from. Includes the official short-
    text downshift (``v <- short_text_local_speed`` when bytes < threshold).

All strategies clamp into ``[min_seconds, max_seconds]``.
"""
from __future__ import annotations

from typing import Any, Callable


def count_words(text: str) -> int:
    """Whitespace word count, with a floor of 1.

    The floor keeps the ``auto`` strategy from producing a zero target
    duration on empty / whitespace-only prompts -- callers always clamp
    ``>= min_seconds`` afterwards anyway, but starting from 0 makes the
    pre-clamp value misleading in logs.
    """
    return max(1, len(str(text).split()))


def make_duration_estimator(
    *,
    strategy: str,
    fixed_duration: float,
    words_per_second: float = 3.0,
    margin_seconds: float = 0.5,
    min_seconds: float = 3.0,
    max_seconds: float | None = None,
    bytes_per_second: float = 17.0,
    f5_local_speed: float = 1.0,
    f5_short_text_threshold_bytes: int = 10,
    f5_short_text_local_speed: float = 0.3,
    f5_margin_seconds: float = 0.0,
) -> Callable[[str], float]:
    """Build a callable used by ``_generate_one_set`` to fill the
    ``duration: X.Xs`` suffix per prompt.

    ``max_seconds=None`` defaults to ``fixed_duration`` (the latent budget),
    matching what the inference sweep does -- never request a target longer
    than the latent can hold.

    See module docstring for what each strategy does.
    """
    hi_default = float(fixed_duration if max_seconds is None else max_seconds)
    lo = max(0.1, float(min_seconds))
    hi = max(lo, hi_default)

    if strategy == "fixed":
        const = float(fixed_duration)

        def _const(_text: str) -> float:
            return const

        return _const

    if strategy == "auto":
        rate = max(0.1, float(words_per_second))
        margin = max(0.0, float(margin_seconds))

        def _estimate_auto(text: str) -> float:
            est = float(count_words(text)) / rate + margin
            return max(lo, min(hi, est))

        return _estimate_auto

    if strategy == "f5":
        bps = max(0.1, float(bytes_per_second))
        base_v = max(0.05, float(f5_local_speed))
        short_thr = max(0, int(f5_short_text_threshold_bytes))
        short_v = max(0.05, float(f5_short_text_local_speed))
        margin = max(0.0, float(f5_margin_seconds))

        def _estimate_f5(text: str) -> float:
            # F5-TTS official:
            #   duration = ref_audio_len + ref_audio_len/ref_text_len * L_gen / v
            # We have no reference utterance (pure T2A), so we replace
            # ``ref_audio_len/ref_text_len`` (frames per byte) with the global
            # prior ``1 / bytes_per_second`` (sec/byte). The short-text
            # downshift (``v <- short_text_local_speed`` when L_gen <
            # short_text_threshold_bytes) is kept verbatim.
            gen_bytes = len(str(text).encode("utf-8"))
            v = base_v if gen_bytes >= short_thr else short_v
            est = float(gen_bytes) / (bps * v) + margin
            return max(lo, min(hi, est))

        return _estimate_f5

    raise ValueError(
        f"Unknown prompt-duration strategy {strategy!r}; "
        "expected one of 'fixed', 'auto', 'f5'."
    )


def make_duration_estimator_from_validation_config(
    val_cfg: dict | None,
    *,
    fixed_duration: float,
    default_strategy: str = "f5",
) -> tuple[Callable[[str], float], dict[str, Any]]:
    """Read ``validation.prompt_duration`` from the loaded YAML config and
    return ``(estimator, summary_dict)``.

    All keys under ``validation.prompt_duration`` are optional; missing keys
    fall back to the function's defaults. The returned ``summary_dict`` is
    suitable for one-line logging on rank 0 and matches the manifest layout
    used by ``infer/audio/run_eval.py``.

    Recognized keys (all optional)::

        validation:
          duration_seconds: 30.0          # latent budget (already read by the caller)
          prompt_duration:
            strategy: f5                   # 'f5' (default) | 'auto' | 'fixed'
            min_seconds: 3.0
            max_seconds: null              # null -> duration_seconds
            words_per_second: 3.0          # 'auto' only
            margin_seconds: 0.5            # 'auto' only
            f5_bytes_per_second: 17.0
            f5_local_speed: 1.0
            f5_short_text_threshold_bytes: 10
            f5_short_text_local_speed: 0.3
            f5_margin_seconds: 0.0

    Top-level keys ``prompt_duration_strategy`` / ``prompt_duration_*`` on
    ``validation`` itself are also accepted (flat layout) so projects that
    don't want a nested block don't have to introduce one.
    """
    val_cfg = dict(val_cfg or {})
    nested = dict(val_cfg.get("prompt_duration") or {})

    def _get(key: str, default: Any) -> Any:
        # Nested wins over flat; flat wins over default. ``None`` in the
        # config is treated as "not set" so users can write ``max_seconds:
        # null`` and still get the default-derived value.
        if key in nested and nested[key] is not None:
            return nested[key]
        flat_key = f"prompt_duration_{key}"
        if flat_key in val_cfg and val_cfg[flat_key] is not None:
            return val_cfg[flat_key]
        return default

    strategy = str(_get("strategy", default_strategy)).lower()
    min_seconds = float(_get("min_seconds", 3.0))
    max_seconds_raw = _get("max_seconds", None)
    max_seconds = float(max_seconds_raw) if max_seconds_raw is not None else None
    words_per_second = float(_get("words_per_second", 3.0))
    margin_seconds = float(_get("margin_seconds", 0.5))
    f5_bytes_per_second = float(_get("f5_bytes_per_second", 17.0))
    f5_local_speed = float(_get("f5_local_speed", 1.0))
    f5_short_text_threshold_bytes = int(_get("f5_short_text_threshold_bytes", 10))
    f5_short_text_local_speed = float(_get("f5_short_text_local_speed", 0.3))
    f5_margin_seconds = float(_get("f5_margin_seconds", 0.0))

    estimator = make_duration_estimator(
        strategy=strategy,
        fixed_duration=fixed_duration,
        words_per_second=words_per_second,
        margin_seconds=margin_seconds,
        min_seconds=min_seconds,
        max_seconds=max_seconds,
        bytes_per_second=f5_bytes_per_second,
        f5_local_speed=f5_local_speed,
        f5_short_text_threshold_bytes=f5_short_text_threshold_bytes,
        f5_short_text_local_speed=f5_short_text_local_speed,
        f5_margin_seconds=f5_margin_seconds,
    )

    summary = {
        "strategy": strategy,
        "latent_duration_seconds": float(fixed_duration),
        "clamp_min_seconds": float(min_seconds),
        "clamp_max_seconds": float(
            fixed_duration if max_seconds is None else max_seconds
        ),
        "auto": {
            "words_per_second": words_per_second,
            "margin_seconds": margin_seconds,
        },
        "f5": {
            "bytes_per_second": f5_bytes_per_second,
            "local_speed": f5_local_speed,
            "short_text_threshold_bytes": f5_short_text_threshold_bytes,
            "short_text_local_speed": f5_short_text_local_speed,
            "margin_seconds": f5_margin_seconds,
        },
    }

    return estimator, summary


__all__ = [
    "count_words",
    "make_duration_estimator",
    "make_duration_estimator_from_validation_config",
]
