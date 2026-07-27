import torch
import numpy as np
from aesthetic.manica_utils.config import Config
from aesthetic.manica_utils.inference_process import ToTensor, Normalize
from aesthetic.manica_utils.maniqa import MANIQA
from torchvision import transforms
from tqdm import tqdm
from PIL import Image


class ImageSample(torch.utils.data.Dataset):
    def __init__(self, image_pil, transform, num_crops=20):
        super(ImageSample, self).__init__()
        self.img_name = ""
        self.img = np.array(image_pil)
        self.img = np.array(self.img).astype('float32') / 255
        self.img = np.transpose(self.img, (2, 0, 1))

        self.transform = transform

        c, h, w = self.img.shape
        print(self.img.shape)
        new_h = 224
        new_w = 224

        self.img_patches = []
        for i in range(num_crops):
            top = np.random.randint(0, h - new_h)
            left = np.random.randint(0, w - new_w)
            patch = self.img[:, top: top + new_h, left: left + new_w]
            self.img_patches.append(patch)

        self.img_patches = np.array(self.img_patches)

    def get_patch(self, idx):
        patch = self.img_patches[idx]
        sample = {'d_img_org': patch, 'score': 0, 'd_name': self.img_name}
        if self.transform:
            sample = self.transform(sample)
        return sample


class ManiqaInferencer:
    def __init__(self, model_path):
        config = Config({
            "num_crops": 20,
            "patch_size": 8,
            "img_size": 224,
            "embed_dim": 768,
            "dim_mlp": 768,
            "num_heads": [4, 4],
            "window_size": 4,
            "depths": [2, 2],
            "num_outputs": 1,
            "num_tab": 2,
            "scale": 0.8,

            # checkpoint path
            "ckpt_path": f"{model_path}/ckpt_koniq10k.pt",
        })
        self.config = config
        net = MANIQA(embed_dim=config.embed_dim, num_outputs=config.num_outputs, dim_mlp=config.dim_mlp,
                     patch_size=config.patch_size, img_size=config.img_size, window_size=config.window_size,
                     depths=config.depths, num_heads=config.num_heads, num_tab=config.num_tab, scale=config.scale)

        net.load_state_dict(torch.load(config.ckpt_path), strict=False)
        net = net.cuda()
        self.net = net

    def infer(self, image_path):
        Img = ImageSample(image_pil=image_path,
                          transform=transforms.Compose([Normalize(0.5, 0.5), ToTensor()]),
                          num_crops=self.config.num_crops)
        avg_score = 0
        for i in tqdm(range(self.config.num_crops)):
            with torch.no_grad():
                self.net.eval()
                patch_sample = Img.get_patch(i)
                patch = patch_sample['d_img_org'].cuda()
                patch = patch.unsqueeze(0)
                score = self.net(patch)
                avg_score += score
        result = avg_score / self.config.num_crops
        return result.cpu().item()


