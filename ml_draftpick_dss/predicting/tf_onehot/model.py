from torchinfo import summary
import torch
from ..transformer.model import ResultPredictorModel as _ResultPredictorModel

class ResultPredictorModel(_ResultPredictorModel):
    def __init__(self, d_input, *args, dim=2, pos_encoder=True, **kwargs):
        super().__init__(d_input, *args, dim=dim, pos_encoder=pos_encoder, **kwargs)
        self.name = "predictor_tf_onehot"
        self.d_input = d_input

    def summary(self, batch_size=32, dtype=torch.float):
        return summary(
            self, 
            [(batch_size, self.d_input) for i in range(2)], 
            dtypes=[dtype, dtype]
        )
