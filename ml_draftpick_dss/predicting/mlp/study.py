import torch
from .dataset import create_dataloader
from .predictor import ResultPredictor
from ..study import BOOLEAN, get_metric, LRS, EPOCHS, SCHEDULER_CONFIGS
import optuna


PARAM_SPACE = {
    "d_final": ("int_exp_2", 32, 256),
    "d_hid_encoder": ("int_exp_2", 16, 128),
    "n_layers_encoder": ("int", 3, 8),
    "activation_encoder": ("activation", ["relu", "tanh", "elu"]),
    "bias_encoder": BOOLEAN,
    "d_hid_final": ("int_exp_2", 16, 256),
    "n_layers_final": ("int_exp_2", 1, 8),
    "activation_final": ("activation", ["identity", "relu", "tanh", "leakyrelu", "elu"]),
    "bias_final": BOOLEAN,
    "n_layers_head": ("int", 1, 6),
    "dropout": ("float", 0.05, 0.25),
    "lrs": ("lrs", list(range(len(LRS)))),
    "optimizer": ("optimizer", ["adamw", "sgd"]),
    "grad_clipping": ("bool_float", 0.6, 2.0),
    "pooling": ("categorical", ["concat", "diff_left", "diff_right", "max"]),
}

PARAMS_DEFAULT = {

}

PARAM_MAP = {}
"""
LRS = [
    [1e-2]
]
EPOCHS = [
    [200]
]
"""
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
    d_hid_encoder=128,
    n_layers_encoder=2,
    activation_encoder=torch.nn.ReLU,
    bias_encoder=True,
    d_final=128,
    d_hid_final=128,
    n_layers_final=3,
    activation_final=torch.nn.ReLU,
    bias_final=True,
    n_layers_head=1,
    dropout=0.1,
    pooling="diff_left",
    n_heads=3,
    activation_final_head=torch.nn.Tanh,
    predictor=ResultPredictor,
    **kwargs
):
    if not isinstance(d_input, int):
        d_input = d_input.dim

    _predictor = predictor(
        d_input,
        d_final,
        encoder_kwargs={
            "d_hid": d_hid_encoder,
            "n_layers": n_layers_encoder,
            "activation": activation_encoder,
            "bias": bias_encoder,
            "dropout": dropout,
        }, 
        final_kwargs={
            "d_hid": d_hid_final,
            "n_layers": n_layers_final,
            "activation": activation_final,
            "bias": bias_final,
            "dropout": dropout,
            "pooling": pooling
        },
        head_kwargs={
            "n_heads": n_heads,
            "d_hid": d_hid_final,
            "n_layers": n_layers_head,
            "activation": activation_final,
            "activation_final": activation_final_head,
            "bias": bias_final,
            "dropout": dropout,
        },
        **kwargs
    )
    return _predictor