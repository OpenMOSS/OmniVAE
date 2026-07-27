import os

from transformers import pipeline
from transformers.image_utils import load_image
from PIL import Image
import numpy as np
import torch


class DinoV3Inferencer:
    def __init__(self, model_path, device=None):
        model_path = f"{model_path}/dinov3-vitl16-pretrain-lvd1689m"
        if device is None:
            device = os.environ.get("MY_EVAL_DINO_DEVICE")
        if device is None:
            device = f"cuda:{torch.cuda.current_device()}" if torch.cuda.is_available() else "cpu"
        print(f"[DinoV3Inferencer] device={device}", flush=True)
        torch_device = torch.device(device)
        self.feature_extractor = pipeline(
            model=model_path,
            task="image-feature-extraction",
            device=device,
        )
        # transformers.Pipeline resets self.device to self.model.device when
        # torch.distributed is already initialized. my_eval initializes DDP
        # before constructing metric tasks, so the freshly loaded CPU model can
        # override the requested cuda:<local_rank>. Force the effective device
        # after construction.
        if torch_device.type == "cuda":
            torch.cuda.set_device(torch_device)
        self.feature_extractor.model.to(torch_device)
        self.feature_extractor.device = torch_device
        model_device = getattr(self.feature_extractor.model, "device", None)
        print(
            f"[DinoV3Inferencer] effective_device={self.feature_extractor.device} "
            f"model_device={model_device}",
            flush=True,
        )

    def infer(self, image_pil1, image_pil2):
        features1 = self.feature_extractor(image_pil1)[0][-2]
        features2 = self.feature_extractor(image_pil2)[0][-2]
        cosing_sim = self.infer_feature(features1, features2)
        return cosing_sim

    def infer_feature(self, feature1, feature2):
        feature1 = torch.from_numpy(np.array(feature1))
        feature2 = torch.from_numpy(np.array(feature2))
        cosing_sim = (feature1 @ feature2) / (feature1.norm() * feature2.norm())
        return cosing_sim.cpu().item()

    def get_feature(self, image_pil):
        features = self.feature_extractor(image_pil)[0][-2]
        return features
