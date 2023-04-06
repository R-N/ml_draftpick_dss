from ..transformer.study import objective as _objective, LRS, EPOCHS, PARAM_SPACE
from .predictor import ResultPredictor
from .model import create_encoder
from ..study import BOOLEAN
import torch

PARAM_SPACE = {
    **PARAM_SPACE,
    "d_hid_encoder": ("int", 32, 256, 32),
    "n_layers_encoder": ("int", 1, 8),
    "activation_encoder": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_encoder": BOOLEAN,
    #"n_layers_tf": ("int", 1, 1),
    #"lrs": ("categorical", list(range(len(LRS)))),
    #"epochs": ("categorical", list(range(len(EPOCHS)))),
}
PARAM_SPACE.pop("s_embed")
PARAM_SPACE.pop("pos_encoder")
PARAM_SPACE.pop("n_layers_tf")

PARAM_MAP = {}
"""
LRS = LRS
EPOCHS = EPOCHS
PARAM_MAP = {
    "lrs": LRS,
    "epochs": EPOCHS,
}
"""

def objective(
    datasets, 
    d_input=171,
    d_hid_encoder=128,
    n_layers_encoder=2,
    activation_encoder=torch.nn.ReLU,
    bias_encoder=True,
    n_layers_tf=1, 
    predictor=ResultPredictor, 
    **kwargs
):
    if not isinstance(d_input, int):
        d_input = d_input.dim
    encoder = create_encoder(
        d_input,
        d_hid=d_hid_encoder,
        d_output=d_hid_encoder,
        n_layers=n_layers_encoder,
        activation=activation_encoder,
        bias=bias_encoder
    )
    return _objective(datasets, encoder, n_layers_tf=n_layers_tf, predictor=predictor, **kwargs)
