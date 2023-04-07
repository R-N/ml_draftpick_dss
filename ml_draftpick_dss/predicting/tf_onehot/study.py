from ..transformer.study import create_predictor as _create_predictor, LRS, EPOCHS, PARAM_SPACE
from .predictor import ResultPredictor
from .dataset import create_dataloader
from .model import create_encoder
from ..study import BOOLEAN
import torch

PARAM_SPACE = {
    **PARAM_SPACE,
    #"d_hid_encoder": ("int", 32, 256, 32),
    #"n_layers_encoder": ("int", 1, 8),
    #"activation_encoder": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    #"bias_encoder": BOOLEAN,
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

def create_predictor(
    d_input=171,
    n_layers_tf=1, 
    predictor=ResultPredictor, 
    **kwargs
):
    if not isinstance(d_input, int):
        d_input = d_input.dim
    return _create_predictor(d_input, n_layers_tf=n_layers_tf, predictor=predictor, **kwargs)
