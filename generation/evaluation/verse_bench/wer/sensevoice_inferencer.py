import os
from pathlib import Path

import torch
from funasr import AutoModel
from funasr.utils.postprocess_utils import rich_transcription_postprocess


def _resolve_vad_model(model_path):
    explicit = os.environ.get("MY_EVAL_FUNASR_VAD_MODEL")
    candidates = []
    if explicit:
        candidates.append(Path(explicit).expanduser())

    root = Path(model_path)
    candidates.extend([
        root / "speech_fsmn_vad_zh-cn-16k-common-pytorch",
        root / "fsmn-vad",
    ])

    modelscope_cache = os.environ.get("MODELSCOPE_CACHE")
    if modelscope_cache:
        candidates.append(
            Path(modelscope_cache).expanduser()
            / "models" / "iic" / "speech_fsmn_vad_zh-cn-16k-common-pytorch"
        )

    for candidate in candidates:
        if (candidate / "config.yaml").is_file() and (candidate / "model.pt").is_file():
            return str(candidate)

    # Last-resort fallback keeps the original online alias behaviour.
    return "fsmn-vad"


class SenseVoiceInferencer:
    def __init__(self, model_path, device=None):
        model_dir = f"{model_path}/SenseVoiceSmall"
        vad_model = _resolve_vad_model(model_path)
        if device is None:
            device = os.environ.get("MY_EVAL_WER_DEVICE")
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.device = str(device)

        # 如果自己下载的sensevoice模型，建议删除目录下的requirements.txt文件，否则你不知道那帮老登会给你的环境里装什么。
        model = AutoModel(
            model=model_dir,
            vad_model=vad_model,
            vad_kwargs={"max_single_segment_time": 30000},
            device=self.device,
            hub="hf",
            disable_update=True
        )
        self.model = model
        self.batch_size_s = int(os.environ.get("MY_EVAL_WER_BATCH_SIZE_S", "60"))

    def _generate(self, audio_input):
        return self.model.generate(
            input=audio_input,
            cache={},
            language="auto",  # "zn", "en", "yue", "ja", "ko", "nospeech"
            use_itn=True,
            batch_size_s=self.batch_size_s,
            merge_vad=True,  #
            merge_length_s=15,
        )

    @staticmethod
    def _postprocess_item(item):
        if isinstance(item, dict):
            text = item.get("text", "")
        else:
            text = str(item or "")
        return rich_transcription_postprocess(text)

    def _infer_one(self, audio_path):
        res = self._generate(audio_path)
        if isinstance(res, dict):
            res = [res]
        if not res:
            return ""
        return self._postprocess_item(res[0])

    def infer_batch(self, audio_paths):
        paths = list(audio_paths)
        if not paths:
            return []
        try:
            res = self._generate(paths)
            if isinstance(res, dict):
                res = [res]
            if not isinstance(res, (list, tuple)):
                raise TypeError(f"unexpected SenseVoice batch result type: {type(res)}")
            texts = [self._postprocess_item(item) for item in list(res)[:len(paths)]]
            if len(texts) < len(paths):
                texts.extend([""] * (len(paths) - len(texts)))
            return texts
        except Exception as exc:
            print(
                f"[SenseVoiceInferencer] batch generate failed "
                f"({len(paths)} audios): {exc}; retry one-by-one",
                flush=True,
            )
            return [self._infer_one(path) for path in paths]

    def infer(self, audio_path):
        return self._infer_one(audio_path)
