import os
import shutil
import time
import librosa
import soundfile
from moviepy.editor import *
from insightface.app import FaceAnalysis
from syncnet.synchronizer import Synchronizer
import cv2
import numpy as np


class SyncnetInferencer:
    def __init__(self, model_path, device_id=0):
        self.synchronizer = Synchronizer(model_path)
        face_kwargs = {'providers': ['CUDAExecutionProvider', 'CPUExecutionProvider']}
        insightface_root = os.environ.get("INSIGHTFACE_HOME")
        if insightface_root:
            face_kwargs["root"] = insightface_root
        self.face_app = FaceAnalysis(**face_kwargs)
        self.face_app.prepare(device_id, det_size=(640, 480))
        self.fps = 25
        self.sr = 16000

    def align_duration(self, work_dir):
        audio = librosa.load(os.path.join(work_dir, "audio.wav"), sr=self.sr)[0]
        frames = os.listdir(os.path.join(work_dir, "images"))
        frames.sort()
        remove_frames = []
        while len(frames) / self.fps > len(audio) / self.sr:
            remove_frames.append(frames[-1])
            frames = frames[:-1]
        if len(frames) / self.fps < len(audio) / self.sr:
            audio = audio[0:int(len(frames) / self.fps * self.sr)]
        soundfile.write(f"{work_dir}/audio.wav", audio, self.sr)
        for frame in remove_frames:
            os.remove(os.path.join(work_dir, "images", frame))

    def video_to_frames_audio(self, video_path, work_dir):
        os.makedirs(f"{work_dir}/images", exist_ok=True)
        os.system(
            f"ffmpeg -y -threads 1 -i {video_path} -q:v 2 -r {self.fps} -filter:v fps={self.fps} {work_dir}/images/frame-%05d.jpg")
        os.system(
            f"ffmpeg -y -threads 1 -i {video_path} -q:a 0 -ac 1 -ar {self.sr} -acodec pcm_s16le -threads 1 {work_dir}/audio.wav")

    def crop_face(self, work_dir):
        frame_files = os.listdir(os.path.join(work_dir, "images"))
        audio = librosa.load(os.path.join(work_dir, "audio.wav"), sr=self.sr)[0]
        frame_files.sort()
        segs = [[]]
        for i, frame_file in enumerate(frame_files):
            img_path = os.path.join(work_dir, "images", frame_file)
            img = cv2.imread(img_path)
            faces = self.face_app.get(img)
            if len(faces) == 0:
                segs.append([])
                continue
            else:
                segs[-1].append([i, img, faces[0].bbox])
        scenes = []
        for j, seg in enumerate(segs):
            if len(seg) < self.fps * 2:
                continue
            os.makedirs(os.path.join(work_dir, f"scenes_{j}", "images"), exist_ok=True)
            for k, (idx, img, bbox) in enumerate(seg):
                cx = (bbox[0] + bbox[2]) / 2
                cy = (bbox[1] + bbox[3]) / 2
                cs = 0.3
                bs = max((bbox[3]-bbox[1]),(bbox[2]-bbox[0]))/2
                bsi = int(bs * (1 + 2 * cs))
                frame = np.pad(img, ((bsi, bsi), (bsi, bsi), (0, 0)), 'constant', constant_values=(110, 110))
                my = cy + bsi  # BBox center Y
                mx = cx + bsi  # BBox center X

                face = frame[int(my - bs):int(my + bs * (1 + 2 * cs)), int(mx - bs * (1 + cs)):int(mx + bs * (1 + cs))]
                face = cv2.resize(face, (224, 224))
                cv2.imwrite(os.path.join(work_dir, f"scenes_{j}", "images", f"{k:05d}.jpg"), face)
            soundfile.write(os.path.join(work_dir, f"scenes_{j}", "audio.wav"),
                            audio[int(seg[0][0] / self.fps * self.sr):int(seg[-1][0] / self.fps * self.sr)], self.sr)
            scenes.append(os.path.join(work_dir, f"scenes_{j}"))
        return scenes

    def infer(self, video_path, sr=16000, fps=25.0, batch_size=20, vshift=15, temp_audio_path="audio.wav"):
        work_dir = f"work_dir/{time.time()}"
        os.makedirs(work_dir, exist_ok=True)
        self.video_to_frames_audio(video_path, work_dir)
        self.align_duration(work_dir)
        scenes = self.crop_face(work_dir)
        if len(scenes) == 0:
            shutil.rmtree(work_dir, ignore_errors=True)
            return None, None, None
        offsets = []
        confs = []
        distss = []
        for scene in scenes:
            offset, conf, dists = self.synchronizer.synchronize(scene,sr=sr, fps=fps, batch_size=batch_size, vshift=vshift,
                                               temp_audio_path=os.path.join(work_dir, "sync-audio.wav"))
            if offset is None:
                shutil.rmtree(scene, ignore_errors=True)
                return None, None, None
            offsets.append(offset)
            confs.append(conf)
            distss.append(dists)
        shutil.rmtree(work_dir, ignore_errors=True)
        return sum(offsets) / len(offsets), sum(confs) / len(confs), np.array(distss).mean(0).mean(0).tolist()
