import math
import os
import shutil

import librosa
import soundfile

from syncnet.SyncNetInstance import *


def sync_video_audio(frames_path, audio_path, offset=0, sr=16000, fps=25.0, temp_audio_path="audio.wav"):
    shutil.copy(audio_path, temp_audio_path)
    audio = librosa.load(audio_path, sr=sr)[0]
    frames = os.listdir(frames_path)
    frames.sort()
    remove_frames = []
    if offset > 0:
        remove_frames.extend(frames[0:offset])
        audio = audio[0:int(-offset / fps * sr)]
    elif offset < 0:
        remove_frames.extend(frames[offset:])
        audio = audio[int(-offset / fps * sr):]
    for frame in remove_frames:
        os.remove(os.path.join(frames_path, frame))
    soundfile.write(temp_audio_path, audio, sr)
    shutil.copy(temp_audio_path, audio_path)
    os.remove(temp_audio_path)


class Synchronizer:
    def __init__(self, model_path):
        self.s = SyncNetInstance()
        self.s.loadParameters(os.path.join(model_path, 'syncnet_v2.model'))

    def synchronize(self, video_path, sr=16000, fps=25.0, batch_size=20, vshift=15, temp_audio_path="audio.wav"):
        offset, conf, dists = self.s.evaluate2(os.path.join(video_path, 'images'),
                                               os.path.join(video_path, 'audio.wav'), fps=fps,
                                               batch_size=batch_size, vshift=vshift)
        if math.fabs(offset)>=14:
            return None, None, None
        return math.fabs(offset), conf, dists
