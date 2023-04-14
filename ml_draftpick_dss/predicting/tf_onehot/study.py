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
    #"bias_encoder": BOOLEAN,
    "d_hid_tf": ("int_exp_2", 32, 64),
    "activation_tf": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "n_heads_tf": ("int", 10, 16, 2),
    "n_layers_tf": ("int", 2, 4),
    "d_hid_final": ("int_exp_2", 64, 128),
    "n_layers_final": ("int", 6, 14),
    "activation_final": ("activation", ["relu", "leakyrelu"]),
    "n_layers_head": ("int", 3, 4),
    "dropout": ("float", 0.0, 0.1),
    "bidirectional": ("categorical", ["none", "concat", "diff_left", "diff_right"]),
    "optimizer": ("optimizer", ["adam", "adamw"]),
    "grad_clipping": ("float", 0.2, 0.8),
    "d_hid_expander": ("int_exp_2", 32, 64),
    "n_layers_expander": ("int", 3, 4),
    "activation_expander": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_expander": BOOLEAN,
    #"use_multihead_linear_expander": BOOLEAN,
    "d_hid_reducer": ("int_exp_2", 16, 64),
    "n_layers_reducer": ("int", 1, 2),
    "activation_reducer": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "dropout_reducer": ("float", 0.04, 0.12),
    #"bias_reducer": BOOLEAN,
    #"lrs": ("lrs", list(range(len(LRS)))),
    "pooling": ("pooling", ["global_average", "global_max"]),
}
PARAM_SPACE.pop("s_embed")

PARAMS_DEFAULT = {
    #**PARAMS_DEFAULT,
    "lrs": LRS[0],
    "optimizer": torch.optim.Adam,
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
    d_hid_encoder=64,
    bias_encoder=False,
    n_layers_tf=2, 
    d_hid_expander=32,
    n_layers_expander=2,
    activation_expander=torch.nn.LeakyReLU,
    bias_expander=True,
    use_multihead_linear_expander=True,
    pos_encoder=True,
    n_heads_tf=8,
    d_hid_tf=32,
    activation_tf=torch.nn.LeakyReLU,
    d_hid_reducer=32,
    n_layers_reducer=1,
    activation_reducer=torch.nn.Identity,
    bias_reducer=False,
    activation_final=torch.nn.ReLU,
    bias_final=True,
    pooling="global_max",
    predictor=ResultPredictor,
    dropout=0.1,
    **kwargs
):
    if not isinstance(d_input, int):
        d_input = d_input.dim
    expander_kwargs={
        "d_hid": d_hid_expander,
        "n_layers": n_layers_expander,
        "activation": activation_expander,
        "bias": bias_expander,
        "dropout": dropout,
        "use_multihead_linear": use_multihead_linear_expander,
    }
    return _create_predictor(
        d_input, 
        d_hid_encoder=d_hid_encoder,
        bias_encoder=bias_encoder,
        expander_kwargs=expander_kwargs, 
        pos_encoder=pos_encoder, 
        n_layers_tf=n_layers_tf, 
        n_heads_tf=n_heads_tf, 
        d_hid_tf=d_hid_tf,
        activation_tf=activation_tf,
        d_hid_reducer=d_hid_reducer,
        n_layers_reducer=n_layers_reducer,
        activation_reducer=activation_reducer,
        bias_reducer=bias_reducer,
        activation_final=activation_final,
        bias_final=bias_final,
        pooling=pooling,
        dropout=dropout,
        predictor=predictor, 
        **kwargs
    )
