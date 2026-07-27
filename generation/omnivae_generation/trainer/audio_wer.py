from __future__ import annotations

import json
import logging
import string
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torchaudio


logger = logging.getLogger(__name__)


def _format_from_path(path: str) -> str:
    suffix = Path(path).suffix.lower()
    if suffix in {".jsonl", ".json"}:
        return "jsonl"
    if suffix in {".lst", ".metalst", ".tsv", ".txt"}:
        return "metalst"
    return "jsonl"


def _parse_metalst_line(line: str) -> tuple[str, str] | None:
    """Replicates bench/audio/seed-tts-eval-main/get_wav_res_ref_text.py column rules.

    Returns (utt, infer_text) on success, None on lines we don't know how to parse.
    """
    parts = line.strip().split("|")
    if not parts or not any(part.strip() for part in parts):
        return None
    if len(parts) == 5:
        utt, _prompt_text, _prompt_wav, infer_text, _infer_wav = parts
    elif len(parts) == 4:
        utt, _prompt_text, _prompt_wav, infer_text = parts
    elif len(parts) == 3:
        utt, infer_text, _prompt_wav = parts
        if utt.endswith(".wav"):
            utt = utt[:-4]
    elif len(parts) == 2:
        utt, infer_text = parts
    else:
        return None
    return utt.strip(), infer_text.strip()


def _coerce_index(index_label: Any, fallback: int) -> Any:
    if isinstance(index_label, int):
        return index_label
    if isinstance(index_label, float):
        return int(index_label)
    text = str(index_label).strip()
    if not text:
        return fallback
    try:
        return int(text)
    except ValueError:
        return text


def load_prompt_set(set_cfg: dict) -> list[dict]:
    """Load a single prompt set spec into the entry schema used by audio_validation.

    Output schema per entry mirrors audio_validation._load_validation_entries:
        {global_idx, type, index, text, line_number}
    """
    path_str = str(set_cfg.get("path") or "").strip()
    if not path_str:
        return []
    path = Path(path_str).expanduser()
    if not path.exists():
        raise FileNotFoundError(f"prompt set path does not exist: {path}")

    fmt = str(set_cfg.get("format") or "").strip().lower() or _format_from_path(path_str)
    text_field = str(set_cfg.get("text_field", "text"))
    type_field = str(set_cfg.get("type_field", "type"))
    index_field = str(set_cfg.get("index_field", "index"))
    num_prompts = set_cfg.get("num_prompts")

    entries: list[dict] = []
    with path.open("r", encoding="utf-8") as handle:
        if fmt == "jsonl":
            for line_number, line in enumerate(handle, start=1):
                if not line.strip():
                    continue
                item = json.loads(line)
                text = item.get(text_field)
                if text is None or not str(text).strip():
                    continue
                type_label = item.get(type_field, "all")
                index_label = item.get(index_field, line_number)
                entries.append(
                    {
                        "global_idx": len(entries),
                        "type": str(type_label),
                        "index": _coerce_index(index_label, line_number),
                        "text": str(text),
                        "line_number": line_number,
                    }
                )
        elif fmt == "metalst":
            for line_number, line in enumerate(handle, start=1):
                parsed = _parse_metalst_line(line)
                if parsed is None:
                    continue
                utt, infer_text = parsed
                if not infer_text:
                    continue
                entries.append(
                    {
                        "global_idx": len(entries),
                        "type": "all",
                        "index": _coerce_index(utt, line_number),
                        "text": infer_text,
                        "line_number": line_number,
                    }
                )
        else:
            raise ValueError(f"Unknown prompt_set format: {fmt!r} (path={path})")

    if num_prompts is not None and int(num_prompts) > 0:
        entries = entries[: int(num_prompts)]
    return entries


def _seedtts_punctuation() -> str:
    """zhon.hanzi.punctuation + string.punctuation, mirroring run_wer.py.

    Falls back to string.punctuation if zhon isn't installed; for pure EN text
    this only changes behavior on rare CJK punctuation, so the fallback is
    safe but logs a warning once.
    """
    try:
        from zhon.hanzi import punctuation as zhon_punctuation
    except ImportError:
        if not getattr(_seedtts_punctuation, "_warned", False):
            logger.warning(
                "zhon not installed; seedtts WER normalization will only strip "
                "string.punctuation. Install zhon to match upstream seed-tts-eval exactly."
            )
            _seedtts_punctuation._warned = True  # type: ignore[attr-defined]
        return string.punctuation
    return zhon_punctuation + string.punctuation


def normalize_text(text: str, mode: str) -> str:
    """EN text normalization for WER scoring.

    - seedtts: drop every char in (zhon.hanzi.punctuation + string.punctuation)
      EXCEPT the apostrophe ', collapse double spaces, lowercase.
    - simple: drop every char in string.punctuation (incl. apostrophe),
      collapse double spaces, lowercase.
    """
    text = "" if text is None else str(text)
    if mode == "seedtts":
        punct = _seedtts_punctuation()
        keep_apostrophe = True
    elif mode == "simple":
        punct = string.punctuation
        keep_apostrophe = False
    else:
        raise ValueError(f"Unknown WER normalization mode: {mode!r}")

    out = text
    for ch in punct:
        if keep_apostrophe and ch == "'":
            continue
        if ch in out:
            out = out.replace(ch, "")
    while "  " in out:
        out = out.replace("  ", " ")
    return out.strip().lower()


def compute_wer(reference: str, hypothesis: str, mode: str) -> dict[str, Any]:
    """Run jiwer on normalized strings; mirror seed-tts-eval/run_wer.py output.

    Uses ``jiwer.process_words`` (jiwer >= 3.0 / 4.0 API). The legacy
    ``compute_measures`` was removed upstream; ``process_words`` returns a
    ``WordOutput`` dataclass exposing ``wer / insertions / deletions /
    substitutions`` with identical numerics.

    ins/del/sub are reported as fractions of reference word count, matching
    ``bench/audio/seed-tts-eval-main/run_wer.py``.
    """
    from jiwer import process_words

    ref_norm = normalize_text(reference, mode)
    hyp_norm = normalize_text(hypothesis, mode)
    ref_words = ref_norm.split(" ") if ref_norm else []
    n_ref = max(1, len(ref_words))

    # jiwer raises on an empty reference; treat empty/empty as a perfect match
    # and an empty reference with a non-empty hyp as 100% insertion (consistent
    # with how upstream seed-tts-eval handles edge cases).
    if not ref_norm:
        if not hyp_norm:
            return {
                "wer": 0.0,
                "ins": 0.0,
                "del": 0.0,
                "sub": 0.0,
                "ref_norm": ref_norm,
                "hyp_norm": hyp_norm,
                "n_ref_words": 0,
            }
        n_hyp = len(hyp_norm.split(" "))
        return {
            "wer": float(n_hyp),
            "ins": float(n_hyp),
            "del": 0.0,
            "sub": 0.0,
            "ref_norm": ref_norm,
            "hyp_norm": hyp_norm,
            "n_ref_words": 0,
        }

    word_output = process_words(ref_norm, hyp_norm)
    return {
        "wer": float(word_output.wer),
        "ins": float(word_output.insertions) / n_ref,
        "del": float(word_output.deletions) / n_ref,
        "sub": float(word_output.substitutions) / n_ref,
        "ref_norm": ref_norm,
        "hyp_norm": hyp_norm,
        "n_ref_words": int(len(ref_words)),
    }


class WhisperEnAsr:
    """Lazy-loaded Whisper-large-v3 (EN) for validation-time WER scoring.

    The model is loaded on first transcribe() call and torn down by unload().
    We force language=english, task=transcribe to mirror seed-tts-eval.
    """

    def __init__(
        self,
        *,
        model_path: str,
        device: torch.device | str,
        torch_dtype: torch.dtype | None = None,
        target_sample_rate: int = 16000,
        local_files_only: bool = False,
    ) -> None:
        if not model_path:
            raise ValueError("WhisperEnAsr requires a non-empty model_path")
        self.model_path = str(model_path)
        self.device = torch.device(device) if not isinstance(device, torch.device) else device
        self.torch_dtype = torch_dtype
        self.target_sample_rate = int(target_sample_rate)
        self.local_files_only = bool(local_files_only)
        self._processor = None
        self._model = None
        self._forced_decoder_ids = None

    def _ensure_loaded(self) -> None:
        if self._model is not None:
            return
        from transformers import WhisperForConditionalGeneration, WhisperProcessor

        logger.info("Loading Whisper ASR model from %s", self.model_path)
        self._processor = WhisperProcessor.from_pretrained(
            self.model_path, local_files_only=self.local_files_only
        )
        load_kwargs: dict[str, Any] = {"local_files_only": self.local_files_only}
        if self.torch_dtype is not None:
            load_kwargs["torch_dtype"] = self.torch_dtype
        model = WhisperForConditionalGeneration.from_pretrained(self.model_path, **load_kwargs)
        model.eval()
        model.to(self.device)
        self._model = model
        self._forced_decoder_ids = self._processor.get_decoder_prompt_ids(
            language="english", task="transcribe"
        )

    @torch.no_grad()
    def transcribe(self, wave: np.ndarray, sample_rate: int) -> str:
        self._ensure_loaded()
        if wave.ndim > 1:
            wave = wave.mean(axis=tuple(range(wave.ndim - 1)))
        wave_t = torch.from_numpy(np.ascontiguousarray(wave)).float()
        if int(sample_rate) != self.target_sample_rate:
            wave_t = torchaudio.functional.resample(
                wave_t, int(sample_rate), self.target_sample_rate
            )
        wave_np = wave_t.cpu().numpy().astype(np.float32)
        features = self._processor(
            wave_np,
            sampling_rate=self.target_sample_rate,
            return_tensors="pt",
        ).input_features
        features = features.to(self.device)
        if self.torch_dtype is not None:
            features = features.to(dtype=self.torch_dtype)
        predicted_ids = self._model.generate(
            features, forced_decoder_ids=self._forced_decoder_ids
        )
        transcription = self._processor.batch_decode(predicted_ids, skip_special_tokens=True)[0]
        return str(transcription).strip()

    def unload(self) -> None:
        if self._model is not None:
            try:
                self._model.to("cpu")
            except Exception:
                pass
            del self._model
            self._model = None
        self._processor = None
        self._forced_decoder_ids = None
        if torch.cuda.is_available():
            torch.cuda.empty_cache()


def transcribe_records(
    records: list[dict],
    *,
    asr: WhisperEnAsr,
    sample_rate: int,
    progress_desc: str | None = None,
) -> None:
    """Run Whisper on each record's `wave`, writing the transcript into record["hyp"].

    GPU-bound; safe to call from any rank that owns the wave. Single-record
    failures are logged and turned into ``hyp = ""`` so downstream WER scoring
    still produces a numeric result for that entry (consistent with the
    previous main-rank-only behavior).

    ``progress_desc`` (default None) optionally enables a tqdm progress bar.
    The caller is responsible for only passing a non-None value on the rank(s)
    that should show the bar (typically the main process), to avoid overlapping
    output in multi-rank runs.
    """
    iterable = records
    if progress_desc:
        try:
            from tqdm.auto import tqdm

            iterable = tqdm(
                records,
                desc=progress_desc,
                leave=False,
                dynamic_ncols=True,
            )
        except ImportError:
            iterable = records

    for record in iterable:
        try:
            hyp = asr.transcribe(record["wave"], sample_rate)
        except Exception as exc:
            logger.warning(
                "WER ASR failed for global_idx=%s index=%s: %r",
                record.get("global_idx"),
                record.get("index"),
                exc,
            )
            hyp = ""
        record["hyp"] = hyp


def score_records(
    records: list[dict],
    *,
    mode: str,
) -> dict[str, Any]:
    """CPU-only WER scoring over records that already carry a ``hyp`` field.

    Each record must have at least:
        global_idx, type, index, text (reference), hyp (transcript).
    Returns the same ``{per_record, summary}`` shape as the previous
    ``evaluate_set`` so callers and on-disk artifacts stay backwards
    compatible.
    """
    per_record: list[dict] = []
    type_buckets: dict[str, list[float]] = {}
    wers: list[float] = []
    ins_list: list[float] = []
    del_list: list[float] = []
    sub_list: list[float] = []
    wers_below_50: list[float] = []
    n_above_50 = 0

    for record in records:
        hyp = record.get("hyp")
        if hyp is None:
            # Defensive: a record reached scoring without going through
            # transcribe_records; treat as empty transcription.
            hyp = ""
        scored = compute_wer(record["text"], hyp, mode)
        wer = scored["wer"]
        wers.append(wer)
        ins_list.append(scored["ins"])
        del_list.append(scored["del"])
        sub_list.append(scored["sub"])
        if wer > 0.5:
            n_above_50 += 1
        else:
            wers_below_50.append(wer)
        type_buckets.setdefault(str(record.get("type", "all")), []).append(wer)
        per_record.append(
            {
                "global_idx": record.get("global_idx"),
                "type": record.get("type"),
                "index": record.get("index"),
                "text": record.get("text"),
                "hyp": hyp,
                "ref_norm": scored["ref_norm"],
                "hyp_norm": scored["hyp_norm"],
                "wer": wer,
                "ins": scored["ins"],
                "del": scored["del"],
                "sub": scored["sub"],
                "n_ref_words": scored["n_ref_words"],
            }
        )

    def _mean(xs: list[float]) -> float:
        return float(sum(xs) / len(xs)) if xs else 0.0

    mean_wer = _mean(wers)
    summary: dict[str, Any] = {
        "n_records": len(per_record),
        "mean_wer": mean_wer,
        "mean_wer_below_50": _mean(wers_below_50),
        "n_above_50": n_above_50,
        "mean_ins": _mean(ins_list),
        "mean_del": _mean(del_list),
        "mean_sub": _mean(sub_list),
        "ins_ratio": float(_mean(ins_list) / mean_wer) if mean_wer > 0 else 0.0,
        "del_ratio": float(_mean(del_list) / mean_wer) if mean_wer > 0 else 0.0,
        "sub_ratio": float(_mean(sub_list) / mean_wer) if mean_wer > 0 else 0.0,
        "per_type_mean_wer": {key: _mean(values) for key, values in type_buckets.items()},
    }
    return {"per_record": per_record, "summary": summary}


def evaluate_set(
    records: list[dict],
    *,
    mode: str,
    asr: WhisperEnAsr,
    sample_rate: int,
) -> dict[str, Any]:
    """Backwards-compatible main-rank entrypoint: transcribe then score.

    Prefer the split ``transcribe_records`` + ``score_records`` pair when you
    want ASR to run on workers other than main rank.
    """
    transcribe_records(records, asr=asr, sample_rate=sample_rate)
    return score_records(records, mode=mode)
