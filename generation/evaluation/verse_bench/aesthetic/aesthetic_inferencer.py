import os.path
from pathlib import Path
import torch
from aesthetic_predictor_v2_5 import convert_v2_5_from_siglip
from PIL import Image


class AestheticInferencer:
    def __init__(self, model_path):
        self.model, self.preprocessor = convert_v2_5_from_siglip(
            os.path.join(model_path, "aesthetic_predictor_v2_5.pth"),
            encoder_model_name=f"{model_path}/siglip-so400m-patch14-384",
            low_cpu_mem_usage=True,
            trust_remote_code=True,
        )
        self.model = self.model.to(torch.bfloat16).cuda()

    def infer(self, image_pil):
        pixel_values = (
            self.preprocessor(images=image_pil, return_tensors="pt")
            .pixel_values.to(torch.bfloat16)
            .cuda()
        )
        with torch.inference_mode():
            score = self.model(pixel_values).logits.squeeze().float().cpu().item()
        return score

    def infer_batch(self, image_pil_list):
        pixel_values = (
            self.preprocessor(images=image_pil_list, return_tensors="pt")
            .pixel_values.to(torch.bfloat16)
            .cuda()
        )
        with torch.inference_mode():
            scores = self.model(pixel_values).logits.float().cpu().tolist()
        return scores
