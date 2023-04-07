import torch
from torch import nn
from torchinfo import summary
from ..modules import Residual, create_mlp_stack
from ..mlp.model import ResultPredictorModel as _ResultPredictorModel

class Scalar(nn.Module):
    def __init__(self, value):
        self.value = value

    def forward(self, x):
        return torch.fill([x.shape[-1]], self.value)
        #return torch.Tensor(self.value)

class ResultPredictorModel(_ResultPredictorModel):
    def __init__(
        self, 
        d_input,
        d_final,
        self_gate_kwargs,
        cross_gate_kwargs,
        *args,
        cross_0=False,
        self_residual=False,
        cross_residual=False,
        **kwargs
    ):
        super().__init__(d_input, d_final, *args, **kwargs)
        self.name = "predictor_mlp_2"
        self.d_input = d_input or 171
        self.d_final = d_final
        self.model_type = 'MLP2'
        self.cross_0 = cross_0
        self.self_residual = self_residual
        self.cross_residual = cross_residual
        self.self_gate = self._create_gate(residual=self_residual, **self_gate_kwargs)
        self.cross_gate = self._create_gate(residual=cross_residual, **cross_gate_kwargs)

    def _create_gate(self, d_hid=0, n_layers=1, activation=torch.nn.ReLU, activation_final=torch.nn.Tanh, bias=True, dropout=0.1, residual=False):
        if n_layers <= 0:
            return Scalar(0 if residual else 1)
        d_hid = d_hid or self.d_final
        gate = nn.Sequential(*[
            create_mlp_stack(self.d_final, d_hid, self.d_final, n_layers-1, activation=activation, bias=bias, dropout=dropout),
            nn.Sequential(*[
                nn.Dropout(dropout),
                nn.Linear(self.d_final, 1, bias=bias),
                activation_final()
            ])
        ])
        return gate

    def forward(self, left, right):
        left = self.encoder(left)
        right = self.encoder(right)

        left_0, right_0 = left, right

        self_w_left = self.self_gate(left_0)
        self_w_right = self.self_gate(right_0)

        self_w_left = self_w_left[None, None, :]
        self_w_right = self_w_right[None, None, :]

        left_0, right_0 = left, right
        left = torch.prod(self_w_left, left)
        right = torch.prod(self_w_right, right)
        if self.self_residual:
            left += left_0
            right += right_0

        if self.cross_0:
            left_0, right_0 = left, right
        cross_w_left = self.cross_gate(left_0)
        cross_w_right = self.cross_gate(right_0)

        self_w_left = cross_w_left[None, None, :]
        self_w_right = cross_w_right[None, None, :]

        left_0, right_0 = left, right
        left = torch.prod(cross_w_left, left)
        right = torch.prod(cross_w_right, right)
        if self.cross_residual:
            left += left_0
            right += right_0
        
        return super().forward(left, right, encode=False)

