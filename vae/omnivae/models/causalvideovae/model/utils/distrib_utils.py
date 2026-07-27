import torch
import numpy as np

class DiagonalGaussianDistribution(object):
    def __init__(self, parameters, deterministic=False):
        self.parameters = parameters
        self.mean, self.logvar = torch.chunk(parameters, 2, dim=1)
        self.logvar = torch.clamp(self.logvar, -30.0, 20.0)
        self.deterministic = deterministic
        self.std = torch.exp(0.5 * self.logvar)
        self.var = torch.exp(self.logvar)
        if self.deterministic:
            self.var = self.std = torch.zeros_like(self.mean, device=self.parameters.device, dtype=self.parameters.dtype)

    def sample(self):
        x = self.mean + self.std * torch.randn(self.mean.shape, device=self.parameters.device, dtype=self.parameters.dtype)
        return x

    def kl(self, other=None, reduction="sum"):
        reduce_dims = tuple(range(1, self.mean.ndim))
        reduce_fn = torch.sum if reduction == "sum" else torch.mean
        if self.deterministic:
            return torch.zeros(self.mean.shape[0], device=self.parameters.device, dtype=self.parameters.dtype)
        else:
            if other is None:
                return 0.5 * reduce_fn(torch.pow(self.mean, 2)
                                       + self.var - 1.0 - self.logvar,
                                       dim=reduce_dims)
            else:
                return 0.5 * reduce_fn(
                    torch.pow(self.mean - other.mean, 2) / other.var
                    + self.var / other.var - 1.0 - self.logvar + other.logvar,
                    dim=reduce_dims)

    def nll(self, sample, dims=[1,2,3]):
        if self.deterministic:
            return torch.Tensor([0.])
        logtwopi = np.log(2.0 * np.pi)
        return 0.5 * torch.sum(
            logtwopi + self.logvar + torch.pow(sample - self.mean, 2) / self.var,
            dim=dims)

    def mode(self):
        return self.mean
