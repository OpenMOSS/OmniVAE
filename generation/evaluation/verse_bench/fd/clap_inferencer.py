import os
from fd.hook import CLAP_Module
from fd.clap_module.factory import load_state_dict
import torch
import librosa
import pyloudnorm as pyln
import numpy as np
from scipy import linalg

try:
    from my_eval.utils.audio_video import load_wav_mono as _my_eval_load_wav_mono
except Exception:
    _my_eval_load_wav_mono = None

def int16_to_float32(x):
    return (x / 32767.0).astype(np.float32)

def float32_to_int16(x):
    x = np.clip(x, a_min=-1., a_max=1.)
    return (x * 32767.).astype(np.int16)

def calculate_embd_statistics(embd_lst):
    if isinstance(embd_lst, list):
        embd_lst = np.array(embd_lst)
    mu = np.mean(embd_lst, axis=0)
    sigma = np.cov(embd_lst, rowvar=False)
    return mu, sigma


def calculate_frechet_distance(mu1, sigma1, mu2, sigma2, eps=1e-6):
    mu1 = np.atleast_1d(mu1)
    mu2 = np.atleast_1d(mu2)

    sigma1 = np.atleast_2d(sigma1)
    sigma2 = np.atleast_2d(sigma2)

    assert mu1.shape == mu2.shape, \
        'Training and test mean vectors have different lengths'
    assert sigma1.shape == sigma2.shape, \
        'Training and test covariances have different dimensions'

    diff = mu1 - mu2

    # product might be almost singular
    covmean, _ = linalg.sqrtm(sigma1.dot(sigma2), disp=False)
    if not np.isfinite(covmean).all():
        msg = ('fid calculation produces singular product; '
               'adding %s to diagonal of cov estimates') % eps
        print(msg)
        offset = np.eye(sigma1.shape[0]) * eps
        covmean = linalg.sqrtm((sigma1 + offset).dot(sigma2 + offset))

    # numerical error might give slight imaginary component
    if np.iscomplexobj(covmean):
        if not np.allclose(np.diagonal(covmean).imag, 0, atol=1e-3):
            m = np.max(np.abs(covmean.imag))
            raise ValueError('Imaginary component {}'.format(m))
        covmean = covmean.real

    tr_covmean = np.trace(covmean)

    return (diff.dot(diff) + np.trace(sigma1)
            + np.trace(sigma2) - 2 * tr_covmean)

class ClapInferencer:
    def __init__(self, model_path, device=None):
        if device is None:
            device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        self.device = torch.device(device)
        model = CLAP_Module(enable_fusion=True, device=str(self.device))
        pkg = load_state_dict(f"{model_path}/630k-audioset-fusion-best.pt")
        # pkg.pop('text_branch.embeddings.position_ids', None)
        model.model.load_state_dict(pkg)
        model.eval()
        self.model = model

    def get_audio_emb(self, audio_path):
        with torch.no_grad():
            if _my_eval_load_wav_mono is not None:
                audio, _ = _my_eval_load_wav_mono(audio_path, 48000)
            else:
                audio, _ = librosa.load(audio_path, sr=48000, mono=True) # sample rate should be 48000
            audio = pyln.normalize.peak(audio, -1.0)
            audio = audio.reshape(1, -1) # unsqueeze (1,T)
            audio = torch.from_numpy(int16_to_float32(float32_to_int16(audio))).float()
            audio_embeddings = self.model.get_audio_embedding_from_data(x = audio, use_tensor=True)
        return audio_embeddings

    def infer(self, audio_path, text_prompt):
        with torch.no_grad():
            embeddings = self.model.get_text_embedding([text_prompt], use_tensor=True)
        audio_embeddings = self.get_audio_emb(audio_path)

        cosine_sim = torch.nn.functional.cosine_similarity(audio_embeddings, embeddings, dim=1, eps=1e-8)[0]
        return cosine_sim.cpu().item()

    def infer_fd(self, audio1_path, audio2_path):
        emb1 = self.get_audio_emb(audio1_path)
        emb2 = self.get_audio_emb(audio2_path)
        mu1, sigma1 = calculate_embd_statistics(emb1.cpu().numpy())
        mu2, sigma2 = calculate_embd_statistics(emb2.cpu().numpy())
        fd = calculate_frechet_distance(mu1, sigma1, mu2, sigma2)
        return fd

if __name__ == '__main__':
    inferencer = ClapInferencer()
    audio_path = "xzsz.mp3"
    text_prompt = "你是一个大傻瓜。"
    score = inferencer.infer(audio_path, text_prompt)
    audio_path1 = "/data/veo3_bmk_data/bmk_aojie/303.wav"
    audio_path2 = "/data/veo3_bmk_data/bmk_aojie/300.wav"
    fd = inferencer.infer_fd(audio_path1,audio_path2)
    print(fd)
