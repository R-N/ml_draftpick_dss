from ..transformer.study import create_predictor as _create_predictor, LRS, EPOCHS, PARAM_SPACE
from .predictor import ResultPredictor
from .dataset import create_dataloader
from .model import create_encoder
from ..study import BOOLEAN
import torch

PARAM_SPACE = {
    **PARAM_SPACE,
    "d_hid_expander": ("int_exp_2", 32, 128),
    "n_layers_expander": ("int", 1, 4),
    "n_heads": ("int_exp_2", 4, 32),
    "n_layers_tf": ("int", 1, 2),
    "activation_expander": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_expander": BOOLEAN,
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
    d_hid_expander=128,
    n_layers_expander=2,
    activation_expander=torch.nn.ReLU,
    bias_expander=True,
    n_heads=8,
    **kwargs
):
    if not isinstance(d_input, int):
        d_input = d_input.dim
    expander_kwargs={
        "d_hid": d_hid_expander,
        "n_layers": n_layers_expander,
        "activation": activation_expander,
        "bias": bias_expander,
        "dropout": kwargs["dropout"],
    }
    return _create_predictor(d_input, n_layers_tf=n_layers_tf, predictor=predictor, expander_kwargs=expander_kwargs, n_heads=n_heads, **kwargs)
