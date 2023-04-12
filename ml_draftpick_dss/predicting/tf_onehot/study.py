from ..transformer.study import create_predictor as _create_predictor, LRS, EPOCHS, PARAM_SPACE, PARAMS_DEFAULT
from .predictor import ResultPredictor
from .dataset import create_dataloader
from .model import create_encoder
from ..study import BOOLEAN
import torch

PARAM_SPACE = {
    **PARAM_SPACE,
    "s_embed": ("int", 2, 4),
    "d_hid_encoder": ("int_exp_2", 8, 64),
    "n_layers_encoder": ("int", 1, 4),
    "activation_encoder": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_encoder": BOOLEAN,
    "d_hid_tf": ("int_exp_2", 8, 64),
    "n_layers_tf": ("int", 1, 2),
    "activation_tf": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "d_hid_final": ("int_exp_2", 8, 64),
    "n_layers_final": ("int_exp_2", 2, 16),
    "activation_final": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_final": BOOLEAN,
    "n_layers_head": ("int", 2, 6),
    "dropout": ("float", 0.0, 0.3),
    "pos_encoder": BOOLEAN,
    "bidirectional": BOOLEAN,
    "pooling": ("pooling", ["global_average", "global_product", "global_max"]),
    "lrs": ("lrs", list(range(len(LRS)))),
    "optimizer": ("optimizer", ["adam", "adamw", "sgd"]),
    "grad_clipping": ("bool_float", 1e-5, 1.0),
    "batch_size": ("int_exp_2", 32, 64),
    "d_hid_expander": ("int_exp_2", 32, 64),
    "n_layers_expander": ("int", 1, 4),
    "n_heads_tf": ("int_exp_2", 4, 32),
    "n_layers_tf": ("int", 1, 2),
    "activation_expander": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_expander": BOOLEAN,
    "d_hid_reducer": ("int_exp_2", 8, 64),
    "n_layers_reducer": ("int", 1, 4),
    "activation_reducer": ("activation", ["identity", "relu", "tanh", "leakyrelu", "elu"]),
    "bias_reducer": BOOLEAN,
    "dropout_reducer": ("float", 0.0, 0.3),
}
PARAM_SPACE.pop("s_embed")
PARAM_SPACE.pop("pos_encoder")
PARAM_SPACE.pop("n_layers_tf")

PARAMS_DEFAULT = {
    **PARAMS_DEFAULT,
}

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
    n_heads_tf=8,
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
    return _create_predictor(d_input, n_layers_tf=n_layers_tf, predictor=predictor, expander_kwargs=expander_kwargs, n_heads_tf=n_heads_tf, **kwargs)
