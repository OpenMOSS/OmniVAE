"""Single source of truth for the per-task instruction-style prefix templates.

Background
----------
The audio model shares a single transformer between TTS (read-aloud speech)
and TTA (environmental / sound-effect generation). Without an explicit
modality signal, short prompts like ``"thunder"`` or ``"a man laughing"`` are
ambiguous, and gradients from the two regimes interfere through the shared
backbone.

We solve this by **wrapping every training/validation/inference prompt with a
short English instruction that names the task** (e.g. ``"Speak the following
text: ..."`` for TTS, ``"Generate sound effects of: ..."`` for TTA,
``"Generate a synchronized video and audio clip of: ..."`` for joint
text-to-audio-video). The text encoder is frozen Qwen3.5, which has been
pretrained on huge volumes of "Task: X. ..." style instructions, so
natural-language prefixes carry far more attention signal than ad-hoc tokens
like ``[TTS]``. To keep the model from latching onto a single surface form
we keep ``10`` template variants per task and pick one uniformly at random
per sample.

Public API
----------
``TTS_TEMPLATES`` / ``TTA_TEMPLATES`` / ``T2AV_TEMPLATES``
    Tuples of ``"... {text}"`` strings. Edit here to add/remove variants;
    every call site reads from these constants so a change is one place.

``pick_template(kind, rng=None)``
    Return one template string for the requested ``kind`` (``"tts" | "tta"
    | "t2av"``; ``"legacy"`` returns ``"{text}"``). ``rng`` is an optional
    ``random.Random`` used for deterministic picks (validation/inference);
    when ``None`` we fall back to the global ``random`` module.

``apply_task_prefix(kind, text, rng=None)``
    The convenience wrapper used by callers: returns the rendered prefix for
    ``kind`` filled with ``text``. ``kind == "legacy"`` (and any unknown
    value) returns ``text`` verbatim so checkpoints trained on the old
    no-prefix data path stay reproducible.

``resolve_task_kind(set_cfg, entry)``
    Used by validation + inference to figure out which template pool to draw
    from. Order:
        1. explicit ``set_cfg["task_kind"]`` if set (``tts``/``tta``/``legacy``)
        2. ``entry["type"] == "tta"`` (case-insensitive)
        3. fallback to ``"tts"``

    Step 2 exists because the TTA validation jsonl (``tta_general_en.jsonl``)
    uses ``"type": "tta"``, but the TTS validation jsonl (``basetts_valid``)
    uses non-task labels like ``"Questions"`` / ``"Statements"`` and the
    metalst loader hardcodes ``"all"``. So we cannot rely on per-entry
    ``type`` alone — but if someone forgets ``task_kind`` on the yaml, the
    "type==tta" heuristic still gives the right answer for new TTA sets.
"""
from __future__ import annotations

import random
from typing import Iterable


# --- template pools ---------------------------------------------------------

TTS_TEMPLATES: tuple[str, ...] = (
    "Speak the following text: {text}",
    "Read aloud: {text}",
    "Synthesize speech for the following: {text}",
    "Voice the following words: {text}",
    "Read this passage out loud: {text}",
    "Generate spoken audio of: {text}",
    "Narrate the following: {text}",
    "Recite this sentence: {text}",
    "Pronounce the following text: {text}",
    "Say aloud: {text}",
)

TTA_TEMPLATES: tuple[str, ...] = (
    "Generate sound effects of: {text}",
    "Produce ambient audio of: {text}",
    "Create a sound recording depicting: {text}",
    "Synthesize the following soundscape: {text}",
    "Render an audio clip of: {text}",
    "Compose non-speech audio of: {text}",
    "Generate environmental sound of: {text}",
    "Produce a realistic recording of: {text}",
    "Generate audio that captures: {text}",
    "Synthesize the sound of: {text}",
)

# Joint text-to-audio-video templates. Used by AVPairedJsonlDataset (training
# data) and the joint_av validation prompt loader so the bridge cross-
# attention sees a clear "video AND audio" instruction signal that's
# distinct from the TTS / TTA single-modality distributions.
T2AV_TEMPLATES: tuple[str, ...] = (
    "Generate a synchronized video and audio clip of: {text}",
    "Produce a video with matching sound depicting: {text}",
    "Create a short film with synchronized audio showing: {text}",
    "Render an audiovisual recording of: {text}",
    "Synthesize video and audio jointly for: {text}",
    "Generate a cinematic clip with synchronized sound of: {text}",
    "Compose a paired video and audio scene of: {text}",
    "Make a sound film clip showing: {text}",
    "Generate an audio-visual scene depicting: {text}",
    "Produce a video clip with realistic accompanying audio of: {text}",
)

_LEGACY_TEMPLATE: str = "{text}"

KIND_TTS: str = "tts"
KIND_TTA: str = "tta"
KIND_T2AV: str = "t2av"
KIND_LEGACY: str = "legacy"
_KNOWN_KINDS: frozenset[str] = frozenset({KIND_TTS, KIND_TTA, KIND_T2AV, KIND_LEGACY})


def _normalize_kind(kind: str | None) -> str:
    if not kind:
        return KIND_LEGACY
    norm = str(kind).strip().lower()
    if norm in _KNOWN_KINDS:
        return norm
    return KIND_LEGACY


def _pool_for(kind: str) -> tuple[str, ...]:
    norm = _normalize_kind(kind)
    if norm == KIND_TTS:
        return TTS_TEMPLATES
    if norm == KIND_TTA:
        return TTA_TEMPLATES
    if norm == KIND_T2AV:
        return T2AV_TEMPLATES
    return (_LEGACY_TEMPLATE,)


# --- public helpers ---------------------------------------------------------

def pick_template(kind: str, rng: random.Random | None = None) -> str:
    """Return one template string drawn from the pool for ``kind``.

    Templates always contain a ``{text}`` placeholder so callers can
    ``.format(text=...)`` them. ``kind == "legacy"`` returns ``"{text}"`` so
    no-prefix code paths stay consistent.
    """
    pool = _pool_for(kind)
    if rng is None:
        return random.choice(pool)
    return rng.choice(pool)


def apply_task_prefix(kind: str, text: str, rng: random.Random | None = None) -> str:
    """Render a randomly-picked task prefix template with ``text``.

    Equivalent to ``pick_template(kind, rng).format(text=text)`` but with a
    safe fallback when ``text`` happens to contain ``{`` / ``}`` that would
    otherwise raise. We escape those before formatting.
    """
    template = pick_template(kind, rng=rng)
    safe_text = str(text).replace("{", "{{").replace("}", "}}")
    return template.format(text=safe_text)


def resolve_task_kind(set_cfg: dict | None, entry: dict | None) -> str:
    """Decide which template pool a validation/inference prompt belongs to.

    See module docstring for the resolution order.
    """
    if isinstance(set_cfg, dict):
        explicit = set_cfg.get("task_kind")
        if explicit:
            norm = str(explicit).strip().lower()
            if norm in _KNOWN_KINDS:
                return norm
    if isinstance(entry, dict):
        entry_type = entry.get("type")
        if entry_type is not None and str(entry_type).strip().lower() == KIND_TTA:
            return KIND_TTA
    return KIND_TTS


__all__: Iterable[str] = (
    "TTS_TEMPLATES",
    "TTA_TEMPLATES",
    "T2AV_TEMPLATES",
    "KIND_TTS",
    "KIND_TTA",
    "KIND_T2AV",
    "KIND_LEGACY",
    "pick_template",
    "apply_task_prefix",
    "resolve_task_kind",
)
