#!/usr/bin/env python
import os
import traceback
from pathlib import Path


def _run(name, fn):
    print(f"[prefetch] {name}")
    try:
        fn()
    except Exception as exc:  # pragma: no cover - setup diagnostics
        print(f"[prefetch] WARN {name}: {exc}")
        if os.environ.get("VERSE_PREFETCH_AUX_STRICT") == "1":
            traceback.print_exc()
            raise
    else:
        print(f"[prefetch] OK {name}")


def prefetch_pyiqa_musiq():
    import pyiqa

    metric = pyiqa.create_metric("musiq", device="cpu")
    del metric


def prefetch_passt():
    from hear21passt.base import get_basic_model

    model = get_basic_model(mode="logits")
    del model


def prefetch_insightface():
    from insightface.app import FaceAnalysis

    root = os.environ.get("INSIGHTFACE_HOME")
    kwargs = {"providers": ["CPUExecutionProvider"]}
    if root:
        kwargs["root"] = root
    app = FaceAnalysis(**kwargs)
    app.prepare(ctx_id=-1, det_size=(640, 480))


def prefetch_funasr_vad():
    from funasr import AutoModel

    models_path = Path(os.environ["MODELS_PATH"])
    sense_voice = models_path / "SenseVoiceSmall"
    model = AutoModel(
        model=str(sense_voice),
        vad_model="fsmn-vad",
        vad_kwargs={"max_single_segment_time": 30000},
        device="cpu",
        hub="hf",
        disable_update=True,
    )
    del model


def main():
    _run("pyiqa/musiq", prefetch_pyiqa_musiq)
    _run("hear21passt", prefetch_passt)
    _run("insightface/buffalo_l", prefetch_insightface)
    _run("funasr/fsmn-vad", prefetch_funasr_vad)


if __name__ == "__main__":
    main()
