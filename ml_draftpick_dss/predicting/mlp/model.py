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

    def _create_final(self, pooling="concat", **kwargs):
        self.d_final_2 = self.d_final * 2
        d_input = self.d_final_2 if pooling == "concat" else self.d_final
        self.pooling = pooling
        self.final = create_mlp_stack(d_input=d_input, d_output=self.d_final, **kwargs)
        return self.final

    def _create_heads(self, d_hid=0, n_layers=1, heads=["victory", "score", "duration"], activation=torch.nn.ReLU, bias=True, dropout=0.1):
        self.head_labels = heads
        d_hid = d_hid or self.d_final_2
        self.heads = [
            nn.Sequential(*[
                create_mlp_stack(self.d_final_2, d_hid, self.d_final, n_layers-1, activation=activation, bias=bias, dropout=dropout),
                nn.Sequential(*[
                    nn.Dropout(dropout),
                    nn.Linear(self.d_final, 1, bias=bias),
                    nn.Tanh()
                ])
            ]) for i in range(len(heads))
        ]

    def forward(self, left, right):
        left = self.encoder(left)
        right = self.encoder(right)
        
        if self.pooling == "concat":
            final = torch.cat([left, right], dim=-1)
        elif self.pooling == "diff":
            final = left - right 
        else:
            final = torch.stack([left, right])
            if self.pooling == "mean":
                final = torch.mean(final, dim=0)
            elif self.pooling == "prod":
                final = torch.prod(final, dim=0)[0]
        final = self.final(final)

        output = [f(final) for f in self.heads]
        return output
    
    def summary(self, batch_size=32, dtype=torch.float):
        return summary(
            self, 
            [(batch_size, self.d_input) for i in range(2)], 
            dtypes=[dtype, dtype]
        )

