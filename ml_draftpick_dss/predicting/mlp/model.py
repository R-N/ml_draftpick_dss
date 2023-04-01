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
        self.d_input = d_input or 171
        self.d_final = d_final
        self.d_final_2 = d_final * 2
        self.model_type = 'MLP'
        self.encoder = create_mlp_stack(d_input=self.d_input, d_output=self.d_final, **encoder_kwargs)
        self.final = create_mlp_stack(d_input=self.d_final_2, d_output=self.d_final_2, **final_kwargs)
        self._create_heads(**head_kwargs)


    def _create_heads(self, heads=["victory", "score", "duration"], activation=nn.Tanh, bias=True, dropout=0.1):
        self.head_labels = heads
        self.heads = [
            nn.Sequential(*[
                nn.Dropout(dropout),
                nn.Linear(self.d_final_2, 1, bias=bias),
                activation()
            ]) for i in range(len(heads))
        ]

    def forward(self, left, right):
        left = self.encoder(left)
        right = self.encoder(right)
        
        final = torch.cat([left, right], dim=-1)
        final = self.final(final)

        output = [f(final) for f in self.heads]
        return output
    
    def summary(self, batch_size=32, team_size=5, dim=6):
        return summary(
            self, 
            [(batch_size, team_size, dim) for i in range(2)], 
            dtypes=[torch.int, torch.int]
        )

