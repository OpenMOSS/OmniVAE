from SyncNetInstance import *
import librosa
import soundfile
s = SyncNetInstance()

s.loadParameters("syncnet_v2.model")

offset,conf, dists = s.evaluate2(r"C:\Users\Administrator\PycharmProjects\syncnet_python\data\work\pytmp\demo",
                  r"C:\Users\Administrator\PycharmProjects\syncnet_python\data\work\pytmp\demo\audio.wav")


def sync_video_audio(frames_path, audio_path, offset=0, sr=16000, fps=25.0):
    # os.makedirs("frames", exist_ok=True)
    # os.system(f"ffmpeg -y -i {video_path} -q:v 2 frames/frame-%05d.jpg")
    # os.system(f"ffmpeg -y -i {video_path} -q:a 0 -ac 1 -ar 16000 -acodec pcm_s16le audio.wav")
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
    soundfile.write(audio_path, audio, sr)


sync_video_audio(r"C:\Users\Administrator\PycharmProjects\syncnet_python\data\work\pytmp\demo",
                 r"C:\Users\Administrator\PycharmProjects\syncnet_python\data\work\pytmp\demo\audio.wav",offset)

print(s.evaluate2(r"C:\Users\Administrator\PycharmProjects\syncnet_python\data\work\pytmp\demo",
                  r"C:\Users\Administrator\PycharmProjects\syncnet_python\data\work\pytmp\demo\audio.wav"))