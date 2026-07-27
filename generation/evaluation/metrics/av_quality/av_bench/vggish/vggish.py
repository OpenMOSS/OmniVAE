# converted from https://github.com/harritaylor/torchvggish

import numpy as np
import torch
import torch.nn as nn

from av_bench.vggish.mel_features import waveform_to_examples


class VGG(nn.Module):

    def __init__(self, features):
        super(VGG, self).__init__()
        self.features = features
        # self.embeddings = nn.Sequential(nn.Linear(512 * 4 * 6, 4096), nn.ReLU(True),
        #                                 nn.Linear(4096, 4096), nn.ReLU(True), nn.Linear(4096, 128),
        #                                 nn.ReLU(True))

        # the last activation is removed to match audioldm_eval
        self.embeddings = nn.Sequential(nn.Linear(512 * 4 * 6, 4096), nn.ReLU(True),
                                        nn.Linear(4096, 4096), nn.ReLU(True), nn.Linear(4096, 128))

    def forward(self, x):
        x = self.features(x)

        # Transpose the output from features to
        # remain compatible with vggish embeddings
        x = torch.transpose(x, 1, 3)
        x = torch.transpose(x, 1, 2)
        x = x.contiguous()
        x = x.view(x.size(0), -1)

        return self.embeddings(x)


def make_layers():
    layers = []
    in_channels = 1
    for v in [64, "M", 128, "M", 256, 256, "M", 512, 512, "M"]:
        if v == "M":
            layers += [nn.MaxPool2d(kernel_size=2, stride=2)]
        else:
            conv2d = nn.Conv2d(in_channels, v, kernel_size=3, padding=1)
            layers += [conv2d, nn.ReLU(inplace=True)]
            in_channels = v
    return nn.Sequential(*layers)


class VGGish(VGG):

    def __init__(self, device=None, pretrained=True, preprocess=True, postprocess=True):
        super().__init__(make_layers())
        if pretrained:
            state_dict = torch.hub.load('harritaylor/torchvggish', 'vggish').state_dict()
            super().load_state_dict(state_dict, strict=False)

        if device is None:
            device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.preprocess = preprocess
        self.postprocess = postprocess

        assert not postprocess, 'not using postprocess following audioldm_eval'

        self.to(self.device)

    def forward(self, x, sample_rate=16000):
        if self.preprocess:
            x = self._preprocess(x, sample_rate)

        bs, num_pieces = x.shape[:2]
        x = x.to(self.device).view(bs * num_pieces, 1, *x.shape[2:])
        x = VGG.forward(self, x)

        # each clip is chopped into X pieces, where X is the middle dimension
        x = x.view(bs, num_pieces, -1)
        return x

    def _preprocess(self, x, sample_rate=16000):
        # x = waveform_to_examples(x, sample_rate)
        x = np.stack([
            waveform_to_examples(waveform.cpu().numpy(), sample_rate, return_tensor=False)
            for waveform in x
        ], 0)
        x = torch.from_numpy(x).float()
        return x
