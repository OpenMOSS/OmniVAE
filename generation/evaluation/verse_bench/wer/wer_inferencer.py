import jiwer
import string
from wer.sensevoice_inferencer import SenseVoiceInferencer


class WERInferencer:
    def __init__(self, model_path, device=None):
        self.device = str(device or "cuda")
        print(f"[wer] backend=SenseVoice device={self.device}", flush=True)
        self.model = SenseVoiceInferencer(f"{model_path}", device=self.device)

    def get_asr(self, audio_path):
        return self.model.infer(audio_path)

    def get_asr_batch(self, audio_paths):
        paths = list(audio_paths)
        if hasattr(self.model, "infer_batch"):
            return list(self.model.infer_batch(paths))
        return [self.get_asr(path) for path in paths]

    def infer(self, audio_path1, audio_path2):
        asr1 = self.get_asr(audio_path1)
        asr2 = self.get_asr(audio_path2)
        measures = jiwer.wer(asr1, asr2)
        return measures

    @staticmethod
    def _normalise_asr(asr):
        asr = str(asr or "")
        PUNCTUATION_SET = set(string.punctuation)
        if asr and set(asr).issubset(PUNCTUATION_SET):
            asr = ""
        return asr.strip().lower()

    @staticmethod
    def _normalise_text(text):
        return str(text or "").strip().lower()

    def infer_audio_text(self, audio_path, text):
        asr = self.get_asr(audio_path)
        text = self._normalise_text(text)
        asr = self._normalise_asr(asr)
        measures = jiwer.wer(text, asr)
        return measures

    def infer_audio_text_batch(self, audio_paths, texts):
        paths = list(audio_paths)
        refs = list(texts)
        asr_texts = self.get_asr_batch(paths)
        out = []
        for ref, asr in zip(refs, asr_texts):
            out.append(jiwer.wer(self._normalise_text(ref), self._normalise_asr(asr)))
        if len(out) < len(paths):
            out.extend([float("nan")] * (len(paths) - len(out)))
        return out

    def infer_text_text(self, text1, text2):
        text1 = self._normalise_text(text1)
        text2 = self._normalise_text(text2)
        measures = jiwer.wer(text1, text2)
        return measures
