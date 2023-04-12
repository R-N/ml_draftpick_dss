import torch
from torch import nn
from torchinfo import summary
from ..modules import Residual, create_mlp_stack


class ResultPredictorModel(nn.Module):
    def __init__(
        self, 
        d_input,
        d_final,
        encoder_kwargs,
        final_kwargs,
        head_kwargs,
    ):
        super().__init__()
        self.name = "predictor_mlp"
        self.d_input = d_input or 171
        self.d_final = d_final
        self.model_type = 'MLP'
        self.encoder = create_mlp_stack(d_input=self.d_input, d_output=self.d_final, **encoder_kwargs)
        self.final = self._create_final(**final_kwargs)
        self._create_heads(**head_kwargs)

    def _create_final(self, pooling="diff_left", **kwargs):
        self.d_final_2 = self.d_final * 2
        d_input = self.d_final_2 if pooling == "concat" else self.d_final
        self.pooling = pooling
        self.final = create_mlp_stack(d_input=d_input, d_output=self.d_final, **kwargs)
        return self.final

    def _create_heads(self, d_hid=0, n_layers=1, n_heads=3, activation=torch.nn.ReLU, activation_final=torch.nn.Tanh, bias=True, dropout=0.1):
        d_hid = d_hid or self.d_final
        """
        self.head = nn.Sequential(*[
            create_mlp_stack(self.d_final, d_hid, self.d_final, n_layers-1, activation=activation, bias=bias, dropout=dropout),
            nn.Sequential(*[
                nn.Dropout(dropout),
                nn.Linear(self.d_final, n_heads, bias=bias),
                activation_final()
            ])
        ])
        """
        self.heads = [
            nn.Sequential(*[
                create_mlp_stack(self.d_final, d_hid, self.d_final, n_layers-1, activation=activation, bias=bias, dropout=dropout),
                nn.Sequential(*[
                    nn.Dropout(dropout),
                    nn.Linear(self.d_final, 1, bias=bias),
                    activation_final()
                ])
            ]) for i in range(n_heads)
        ]

    def forward(self, left, right, encode=True):
        if encode:
            left = self.encoder(left)
            right = self.encoder(right)
        
        if self.pooling == "concat":
            final = torch.cat([left, right], dim=-1)
        elif self.pooling == "diff_left":
            final = left - right 
        elif self.pooling == "diff_right":
            final = right - left 
        else:
            final = torch.stack([left, right])
            if self.pooling == "mean":
                final = torch.mean(final, dim=0)
            elif self.pooling == "prod":
                final = torch.prod(final, dim=0)
            elif self.pooling == "max":
                final = torch.max(final, dim=0)
        if isinstance(final, tuple):
            final = final[0]
        final = self.final(final)

        #output = self.head(final)
        output = [f(final) for f in self.heads]
        return output
    
    def summary(self, batch_size=64, dtype=torch.float):
        return summary(
            self, 
            [(batch_size, self.d_input) for i in range(2)], 
            dtypes=[dtype, dtype]
        )

