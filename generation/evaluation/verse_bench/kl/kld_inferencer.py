import warnings

warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import contextlib
from functools import partial
from tqdm import tqdm
import pickle
import numpy as np
import librosa
from hear21passt.base import get_basic_model
import pyloudnorm as pyln

import torch
import torch.nn.functional as F

SAMPLING_RATE = 32000

try:
    from my_eval.utils.audio_video import load_wav_mono as _my_eval_load_wav_mono
except Exception:
    _my_eval_load_wav_mono = None


class _patch_passt_stft:
    """
    From version 1.8.0, return_complex must always be given explicitly
    for real inputs and return_complex=False has been deprecated.

    Decorator to patch torch.stft in PaSST that uses an old stft version.

    Adapted from: https://github.com/facebookresearch/audiocraft/blob/a2b96756956846e194c9255d0cdadc2b47c93f1b/audiocraft/metrics/kld.py
    """

    def __init__(self):
        self.old_stft = torch.stft

    def __enter__(self):
        # return_complex is a mandatory parameter in latest torch versions.
        # torch is throwing RuntimeErrors when not set.
        # see: https://pytorch.org/docs/1.7.1/generated/torch.stft.html?highlight=stft#torch.stft
        #  see: https://github.com/kkoutini/passt_hear21/commit/dce83183674e559162b49924d666c0a916dc967a
        torch.stft = partial(torch.stft, return_complex=False)

    def __exit__(self, *exc):
        torch.stft = self.old_stft


def return_probabilities(model, audio_path, window_size=10, overlap=5, collect='mean', device=None):
    if _my_eval_load_wav_mono is not None:
        audio, _ = _my_eval_load_wav_mono(audio_path, SAMPLING_RATE)
    else:
        audio, _ = librosa.load(audio_path, sr=SAMPLING_RATE, mono=True)
    audio = pyln.normalize.peak(audio, -1.0)

    # calculate the step size for the analysis windows with the specified overlap
    step_size = int((window_size - overlap) * SAMPLING_RATE)

    # iterate over the audio, creating analysis windows
    probabilities = []
    for i in range(0, max(step_size, len(audio) - step_size), step_size):
        # extract the current analysis window
        window = audio[i:i + int(window_size * SAMPLING_RATE)]

        # pad the window with zeros if it's shorter than the desired window size
        if len(window) < int(window_size * SAMPLING_RATE):
            # discard window if it's too small (avoid mostly zeros predicted as silence), as in MusicGen:
            # https://github.com/facebookresearch/audiocraft/blob/a2b96756956846e194c9255d0cdadc2b47c93f1b/audiocraft/metrics/kld.py
            if len(window) > int(window_size * SAMPLING_RATE * 0.15):
                tmp = np.zeros(int(window_size * SAMPLING_RATE))
                tmp[:len(window)] = window
                window = tmp

        # convert to a PyTorch tensor and move to the rank-local device
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        audio_wave = torch.from_numpy(window.astype(np.float32)).unsqueeze(0).to(device)

        # get the probabilities for this analysis window
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):
            with torch.no_grad(), _patch_passt_stft():
                logits = model(audio_wave)
                probabilities.append(torch.squeeze(logits))

    probabilities = torch.stack(probabilities)
    if collect == 'mean':
        probabilities = torch.mean(probabilities, dim=0)
    elif collect == 'max':
        probabilities, _ = torch.max(probabilities, dim=0)

    return F.softmax(probabilities, dim=0).squeeze().cpu()


class KLDInferencer:
    def __init__(self, device=None):
        if device is None:
            device = torch.device(f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu")
        self.device = torch.device(device)
        with open(os.devnull, 'w') as f, contextlib.redirect_stdout(f):  # capturing all useless outputs from passt
            model = get_basic_model(mode="logits")
            model.eval()
            model = model.to(self.device)
        self.model = model

    def infer(self, audio1, audio2, collect="mean"):
        ref_p = {}
        audio_path = audio1
        ref_p[id] = return_probabilities(self.model, audio_path, collect=collect, device=self.device)
        passt_kl = 0
        count = 1
        audio_path = audio2
        eval_p = return_probabilities(self.model, audio_path, collect=collect, device=self.device)
        passt_kl += F.kl_div((ref_p[id] + 1e-6).log(), eval_p, reduction='sum', log_target=False).cpu().item()
        return passt_kl / count if count > 0 else 0

