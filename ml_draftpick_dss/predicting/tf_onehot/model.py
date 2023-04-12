from torchinfo import summary
import torch
from ..transformer.model import ResultPredictorModel as _ResultPredictorModel
from ..modules import create_mlp_stack

def create_encoder(d_input, d_hid, d_output=0, **kwargs):
    d_output = d_output or d_hid
    mlp = create_mlp_stack(d_input, d_hid, d_output, **kwargs)
    mlp.dim = d_output
    return mlp


class ResultPredictorModel(_ResultPredictorModel):
    def __init__(self, encoder, *args, dim=2, pos_encoder=True, **kwargs):
        super().__init__(
            encoder, 
            *args,
            dim=dim, 
            pos_encoder=pos_encoder, 
            **kwargs
        )
        self.name = "predictor_tf_onehot"

    def summary(self, batch_size=64, d_input=171, dtype=torch.float):
        return summary(
            self, 
            [(batch_size, d_input) for i in range(2)], 
            dtypes=[dtype, dtype]
        )
