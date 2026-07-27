import numpy as np
import pyiqa
import torch
from PIL import Image
from torchvision import transforms


class MusiqInferencer:
    def __init__(self, device='cuda'):
        self.device = device
        self.model = pyiqa.create_metric('musiq', device=device)
        self.transform = transforms.Compose([
            transforms.ToTensor(),
        ])

    def infer(self, img_pil):
        img = img_pil.convert('RGB')
        img = self.transform(img).unsqueeze(0).to(self.device)
        with torch.no_grad():
            score = self.model(img)
        return score[0][0].item()

