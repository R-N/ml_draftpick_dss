from torchinfo import summary
import torch
from ..transformer import ResultPredictorModel as _ResultPredictorModel

class ResultPredictorModel(_ResultPredictorModel):
    def __init__(self, d_input, *args, **kwargs):
        super().__init__(d_input, *args, **kwargs)
        self.d_input = d_input

    def summary(self, batch_size=32, dtype=torch.float):
        return summary(
            self, 
            [(batch_size, self.d_input) for i in range(2)], 
            dtypes=[dtype, dtype]
        )
