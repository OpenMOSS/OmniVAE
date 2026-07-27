"""Small local subset of python_speech_features used by SyncNet.

The upstream SyncNet code imports ``python_speech_features.mfcc``. Some of our
runtime envs do not install that package, so this vendored compatibility module
implements the MFCC path with the same defaults.
"""
from __future__ import annotations

import numpy as np
from scipy.fftpack import dct


def hz2mel(hz):
    return 2595 * np.log10(1 + hz / 700.0)


def mel2hz(mel):
    return 700 * (10 ** (mel / 2595.0) - 1)


def preemphasis(signal, coeff=0.95):
    return np.append(signal[0], signal[1:] - coeff * signal[:-1])


def framesig(sig, frame_len, frame_step, winfunc=lambda x: np.ones((x,))):
    slen = len(sig)
    frame_len = int(round(frame_len))
    frame_step = int(round(frame_step))
    if slen <= frame_len:
        num_frames = 1
    else:
        num_frames = 1 + int(np.ceil((slen - frame_len) / float(frame_step)))
    pad_len = int((num_frames - 1) * frame_step + frame_len)
    zeros = np.zeros((pad_len - slen,))
    padsignal = np.concatenate((sig, zeros))
    indices = (
        np.tile(np.arange(0, frame_len), (num_frames, 1))
        + np.tile(np.arange(0, num_frames * frame_step, frame_step), (frame_len, 1)).T
    )
    frames = padsignal[indices.astype(np.int32, copy=False)]
    return frames * winfunc(frame_len)


def magspec(frames, nfft):
    complex_spec = np.fft.rfft(frames, nfft)
    return np.absolute(complex_spec)


def powspec(frames, nfft):
    return 1.0 / nfft * np.square(magspec(frames, nfft))


def get_filterbanks(nfilt=20, nfft=512, samplerate=16000, lowfreq=0, highfreq=None):
    highfreq = highfreq or samplerate / 2
    lowmel = hz2mel(lowfreq)
    highmel = hz2mel(highfreq)
    melpoints = np.linspace(lowmel, highmel, nfilt + 2)
    bin_points = np.floor((nfft + 1) * mel2hz(melpoints) / samplerate).astype(int)

    fbank = np.zeros([nfilt, nfft // 2 + 1])
    for j in range(nfilt):
        for i in range(bin_points[j], bin_points[j + 1]):
            denom = bin_points[j + 1] - bin_points[j]
            if denom:
                fbank[j, i] = (i - bin_points[j]) / denom
        for i in range(bin_points[j + 1], bin_points[j + 2]):
            denom = bin_points[j + 2] - bin_points[j + 1]
            if denom:
                fbank[j, i] = (bin_points[j + 2] - i) / denom
    return fbank


def lifter(cepstra, L=22):
    if L <= 0:
        return cepstra
    nframes, ncoeff = np.shape(cepstra)
    n = np.arange(ncoeff)
    lift = 1 + (L / 2.0) * np.sin(np.pi * n / L)
    return lift * cepstra


def fbank(
    signal,
    samplerate=16000,
    winlen=0.025,
    winstep=0.01,
    nfilt=26,
    nfft=512,
    lowfreq=0,
    highfreq=None,
    preemph=0.97,
    winfunc=lambda x: np.ones((x,)),
):
    signal = preemphasis(np.asarray(signal, dtype=np.float64), preemph)
    frames = framesig(signal, winlen * samplerate, winstep * samplerate, winfunc)
    pspec = powspec(frames, nfft)
    energy = np.sum(pspec, axis=1)
    energy = np.where(energy == 0, np.finfo(float).eps, energy)
    fb = get_filterbanks(nfilt, nfft, samplerate, lowfreq, highfreq)
    feat = np.dot(pspec, fb.T)
    feat = np.where(feat == 0, np.finfo(float).eps, feat)
    return feat, energy


def logfbank(
    signal,
    samplerate=16000,
    winlen=0.025,
    winstep=0.01,
    nfilt=26,
    nfft=512,
    lowfreq=0,
    highfreq=None,
    preemph=0.97,
    winfunc=lambda x: np.ones((x,)),
):
    feat, _ = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc)
    return np.log(feat)


def mfcc(
    signal,
    samplerate=16000,
    winlen=0.025,
    winstep=0.01,
    numcep=13,
    nfilt=26,
    nfft=512,
    lowfreq=0,
    highfreq=None,
    preemph=0.97,
    ceplifter=22,
    appendEnergy=True,
    winfunc=lambda x: np.ones((x,)),
):
    feat, energy = fbank(signal, samplerate, winlen, winstep, nfilt, nfft, lowfreq, highfreq, preemph, winfunc)
    feat = np.log(feat)
    feat = dct(feat, type=2, axis=1, norm="ortho")[:, :numcep]
    feat = lifter(feat, ceplifter)
    if appendEnergy:
        feat[:, 0] = np.log(energy)
    return feat
