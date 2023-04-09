from .util import sig_to_tanh_range, tanh_to_sig_range
import torch
from torch import nn


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
        ret = self.f(x, dim=-2)
        if isinstance(ret, tuple):
            return ret[0]
        return ret
    
class Residual(torch.nn.Module):
    def __init__(self, module, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.module = module

    def forward(self, x):
        return x + self.module(x)
    
def try_residual(module, flag):
    if flag:
        return Residual(module)
    else:
        return module
    
def create_mlp(d_input, d_output, activation=torch.nn.ReLU, bias=True, dropout=0.1, residual=True):
    mlp = try_residual(
        torch.nn.Sequential(*[
            nn.Dropout(dropout),
            nn.Linear(d_input, d_output, bias=bias),
            activation()
        ]), residual and d_input==d_output
    )
    mlp.dim = d_output
    return mlp

def create_mlp_stack(d_input, d_hid, d_output, n_layers, activation=torch.nn.ReLU, bias=True, dropout=0.1, residual=True):
    d_output = d_output or d_hid
    if n_layers <= 0:
        mlp = nn.Identity()
        mlp.dim = d_input
    elif n_layers == 1:
        mlp = create_mlp(d_input, d_output, activation=activation, bias=bias, dropout=dropout, residual=residual)
        mlp.dim = d_output
    else:
        modules = [
            create_mlp(d_input, d_hid, activation=activation, bias=bias, dropout=dropout, residual=residual),
            *[
                create_mlp(d_hid, d_hid, activation=activation, bias=bias, dropout=dropout, residual=residual)
                for i in range(max(0, n_layers-(2 if d_output else 1)))
            ],
        ]
        if d_output:
            modules.append(create_mlp(d_hid, d_output, activation=activation, bias=bias, dropout=dropout, residual=residual))
        mlp = nn.Sequential(*modules)
        mlp.dim = d_output
    return mlp

class RepeatExpander(torch.nn.Module):
    def __init__(self, n_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads

    def forward(self, x):
        if x.dim() == 2:
            x = x[:, :, None]
        return x.repeat(*((x.dim()-1)*[1]), self.n_heads)

class Scalar(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full([x.shape[-1]], self.value)
        #return torch.Tensor(self.value)

class MLPExpander(torch.nn.Module):
    def __init__(self, n_heads, *args, **kwargs):
        super().__init__()
        self.mlps = [
            create_mlp_stack(*args, **kwargs)
            for i in range(n_heads)
        ]

    def forward(self, x):
        assert x.dim() == 2
        xs = [mlp(x)[:, :, None] for mlp in self.mlps]
        x = torch.concat(xs, dim=-1)
        return x
