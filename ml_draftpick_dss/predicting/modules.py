from .util import sig_to_tanh_range, tanh_to_sig_range
import torch


class NegativeSigmoid(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return sig_to_tanh_range(torch.sigmoid(x))

class PositiveTanh(torch.nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        return tanh_to_sig_range(torch.tanh(x))

class NegativeBCELoss(torch.nn.BCELoss):
    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)

    def forward(self, input, target):
        input = tanh_to_sig_range(input)
        target = tanh_to_sig_range(input)
        return super().forward(input, target)

MEAN = torch.mean
PROD = torch.prod
SUM = torch.sum
MAX = torch.max

class GlobalPooling1D(torch.nn.Module):
    def __init__(self, f=MEAN, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.f = f

    def forward(self, x):
        return self.f(x, dim=-2)
    
class Residual(torch.nn.Module):
    def __init__(self, module):
        self.module = module

    def forward(self, x):
        return x + self.module(x)