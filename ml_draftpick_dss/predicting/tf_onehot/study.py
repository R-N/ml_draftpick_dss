from ..transformer.study import create_predictor as _create_predictor, LRS, EPOCHS, PARAM_SPACE, PARAMS_DEFAULT
from .predictor import ResultPredictor
from .dataset import create_dataloader
from .model import create_encoder
from ..study import BOOLEAN
import torch

PARAM_SPACE = {
    **PARAM_SPACE,
    "d_hid_encoder": ("int_exp_2", 32, 64),
    "n_layers_encoder": ("int", 1, 3),
    "activation_encoder": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "d_hid_tf": ("int_exp_2", 32, 64),
    "activation_tf": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "n_heads_tf": ("int", 4, 16),
    "n_layers_tf": ("int", 1, 2),
    "d_hid_final": ("int_exp_2", 32, 128),
    "n_layers_final": ("int", 4, 16),
    "activation_final": ("activation", ["relu", "leakyrelu"]),
    "n_layers_head": ("int", 2, 4),
    "dropout": ("float", 0.0, 0.15),
    "pos_encoder": BOOLEAN,
    "bidirectional": ("categorical", ["none", "concat", "diff", "mean", "prod", "max"]),
    "optimizer": ("optimizer", ["adam", "adamw"]),
    "grad_clipping": ("float", 0.2, 1.0),
    "d_hid_expander": ("int_exp_2", 32, 64),
    "n_layers_expander": ("int", 2, 4),
    "activation_expander": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "d_hid_reducer": ("int_exp_2", 16, 64),
    "n_layers_reducer": ("int", 1, 2),
    "activation_reducer": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "dropout_reducer": ("float", 0.0, 0.15),
    "bias_reducer": BOOLEAN,
}
PARAM_SPACE.pop("s_embed")

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
    d_hid_expander=32,
    n_layers_expander=2,
    activation_expander=torch.nn.ReLU,
    bias_expander=False,
    n_heads_tf=8,
    pos_encoder=True,
    bias_encoder=False,
    bias_final=True,
    bias_reducer=False,
    pooling="global_max",
    lrs=LRS[0],
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
    return _create_predictor(
        d_input, 
        n_layers_tf=n_layers_tf, 
        predictor=predictor, 
        expander_kwargs=expander_kwargs, 
        n_heads_tf=n_heads_tf, 
        pos_encoder=pos_encoder, 
        bias_encoder=bias_encoder,
        bias_final=bias_final,
        bias_reducer=bias_reducer,
        pooling=pooling,
        lrs=lrs,
        **kwargs
    )
