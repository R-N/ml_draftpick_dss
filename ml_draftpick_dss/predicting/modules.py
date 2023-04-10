from .util import sig_to_tanh_range, tanh_to_sig_range
import torch
from torch import nn
import math


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
        y = self.module(x)
        if x.shape == y.shape:
            return x + y
        return y
    
def try_residual(module, flag):
    if flag:
        return Residual(module)
    else:
        return module
    
def create_mlp(d_input, d_output, activation=torch.nn.ReLU, bias=True, dropout=0.1, residual=True, n_heads=1):
    assert n_heads >= 1
    mlp = try_residual(
        torch.nn.Sequential(*[
            nn.Dropout(dropout),
            nn.Linear(d_input, d_output, bias=bias) if n_heads==1 else\
                MultiheadLinear(d_input, d_output, n_heads=n_heads, bias=bias),
            activation()
        ]), residual and d_input==d_output
    )
    mlp.dim = d_output
    return mlp

def create_mlp_stack(d_input, d_hid, d_output, n_layers, activation=torch.nn.ReLU, bias=True, dropout=0.1, residual=True, n_heads=1):
    d_output = d_output or d_hid
    if n_layers <= 0:
        mlp = nn.Identity()
        mlp.dim = d_input
    elif n_layers == 1:
        mlp = create_mlp(d_input, d_output, activation=activation, bias=bias, dropout=dropout, residual=residual, n_heads=n_heads)
        mlp.dim = d_output
    else:
        modules = [
            create_mlp(d_input, d_hid, activation=activation, bias=bias, dropout=dropout, residual=residual, n_heads=n_heads),
            *[
                create_mlp(d_hid, d_hid, activation=activation, bias=bias, dropout=dropout, residual=residual, n_heads=n_heads)
                for i in range(max(0, n_layers-(2 if d_output else 1)))
            ],
        ]
        if d_output:
            modules.append(create_mlp(d_hid, d_output, activation=activation, bias=bias, dropout=dropout, residual=residual, n_heads=n_heads))
        mlp = nn.Sequential(*modules)
        mlp.dim = d_output
    return mlp

class RepeatExpander(torch.nn.Module):
    def __init__(self, n_heads, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.n_heads = n_heads

    def forward(self, x):
        if x.dim() == 2:
            x = torch.unsqueeze(-1)
        return x.repeat(*((x.dim()-1)*[1]), self.n_heads)

class Scalar(nn.Module):
    def __init__(self, value):
        super().__init__()
        self.value = value

    def forward(self, x):
        return torch.full([x.shape[-1]], self.value)
        #return torch.Tensor(self.value)

class MultiheadLinear(torch.nn.Module):

    def __init__(self, in_features, out_features, n_heads=1, bias=True,
                device=None, dtype=None, reverse_shape=False):
        super().__init__()
        factory_kwargs = {'device': device, 'dtype': dtype}
        self.n_heads = n_heads
        self.in_features = in_features
        self.out_features = out_features
        self.shape_w = (n_heads, 1, out_features, in_features)
        self.shape_out = (out_features, n_heads)
        self.reverse_shape = reverse_shape
        self.weight = nn.Parameter(torch.empty(self.shape_w, **factory_kwargs))
        if bias:
            self.bias = nn.Parameter(torch.empty(self.shape_out, **factory_kwargs))
        else:
            self.register_parameter('bias', None)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.kaiming_uniform_(self.weight, a=math.sqrt(5))
        if self.bias is not None:
            fan_in, _ = nn.init._calculate_fan_in_and_fan_out(self.weight)
            bound = 1 / math.sqrt(fan_in) if fan_in > 0 else 0
            nn.init.uniform_(self.bias, -bound, bound)

    def forward(self, input):
        d_input = input.dim()
        input = torch.unsqueeze(input, -1)
        output = torch.matmul(self.weight, input)
        output = torch.squeeze(output, -1)
        ds_output = tuple(range(1, output.dim()))
        output = torch.permute(output, (*ds_output, 0))
        if (d_input == 1):
            output = torch.squeeze(output, 0)
        if self.bias is not None:
            output += self.bias
        return output

class MLPExpander(torch.nn.Module):
    def __init__(self, n_heads, *args, **kwargs):
        super().__init__()
        self.mlp = create_mlp_stack(*args, n_heads=n_heads, **kwargs)

    def forward(self, x):
        return self.mlp(x)
