import torch
from ..mlp.study import LRS, EPOCHS, PARAM_SPACE, create_predictor as _create_predictor
from .dataset import create_dataloader
from .predictor import ResultPredictor
from ..study import BOOLEAN, get_metric, LRS, EPOCHS, SCHEDULER_CONFIGS
import optuna


PARAM_SPACE = {
    **PARAM_SPACE,
    "d_hid_gate": ("int_exp_2", 32, 256),
    "n_layers_self_gate": ("bool_int_exp_2", 0, 8),
    "n_layers_cross_gate": ("bool_int_exp_2", 0, 8),
    "activation_gate": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "activation_final_gate": ("activation", ["tanh", "sigmoid"]),
    "bias_gate": BOOLEAN,
    "cross_0": BOOLEAN,
    "self_residual": BOOLEAN,
    "cross_residual": BOOLEAN,
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
    d_hid_gate=128,
    n_layers_self_gate=2,
    n_layers_cross_gate=2,
    activation_gate=torch.nn.ReLU,
    activation_final_gate=torch.nn.Tanh,
    bias_gate=True,
    cross_0=False,
    self_residual=False,
    cross_residual=False,
    predictor=ResultPredictor,
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
    pooling="concat",
    #**kwargs
):
    if not isinstance(d_input, int):
        d_input = d_input.dim

    _predictor = predictor(
        d_input,
        d_final,
        self_gate_kwargs={
            "d_hid": d_hid_gate,
            "n_layers": n_layers_self_gate,
            "activation": activation_gate,
            "activation_final": activation_final_gate,
            "bias": bias_gate,
            "dropout": dropout,
        }, 
        cross_gate_kwargs={
            "d_hid": d_hid_gate,
            "n_layers": n_layers_cross_gate,
            "activation": activation_gate,
            "activation_final": activation_final_gate,
            "bias": bias_gate,
            "dropout": dropout,
        }, 
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
            "heads": ["victory", "score", "duration"],
            "d_hid": d_hid_final,
            "n_layers": n_layers_head,
            "activation": activation_final,
            "bias": bias_final,
            "dropout": dropout,
        },
        cross_0=cross_0,
        self_residual=self_residual,
        cross_residual=cross_residual,
    )
    return _predictor