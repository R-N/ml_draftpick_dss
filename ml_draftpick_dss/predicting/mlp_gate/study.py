import torch
from ..mlp.study import PARAM_SPACE, create_predictor as _create_predictor, PARAMS_DEFAULT
from .dataset import create_dataloader
from .predictor import ResultPredictor
from ..study import BOOLEAN, get_metric, SCHEDULER_CONFIGS
import optuna


PARAM_SPACE = {
    #**PARAM_SPACE,
    #"d_final": ("int_exp_2", 128, 256),
    #"d_hid_encoder": ("int_exp_2", 64, 128),
    #"n_layers_encoder": ("int", 5, 6),
    "activation_encoder": ("activation", ["identity", "elu"]),
    #"bias_encoder": BOOLEAN,
    "d_hid_final": ("int_exp_2", 64, 128),
    #"n_layers_final": ("int", 1, 2),
    #"activation_final": ("activation", ["identity", "elu"]),
    #"bias_final": BOOLEAN,
    #"n_layers_head": ("int", 4, 6),
    "dropout": ("float", 0.0, 0.1),
    #"optimizer": ("optimizer", ["adam", "adamw"]),
    "grad_clipping": ("float", 0.6, 1.1),
    #"pooling": ("categorical", ["concat", "diff_left", "diff_right"]),
    "d_hid_gate": ("int_exp_2", 32, 64),
    #"n_layers_self_gate": ("int", 3, 4),
    #"n_layers_cross_gate": ("int", 3, 5),
    #"activation_gate": ("activation", ["relu", "elu"]),
    #"activation_final_gate": ("activation", ["tanh", "sigmoid"]),
    #"bias_gate": BOOLEAN,
    #"cross_0": BOOLEAN,
    #"self_residual": BOOLEAN,
    #"cross_residual": BOOLEAN,
    "onecycle_lr": ("log_float", 1e-2, 1e-1),
    "onecycle_epochs": ("int", 80, 100),
    "lr": ("log_float", 1e-6, 1e-4),
    "min_epoch": ("int", 25, 80),
}

PARAMS_DEFAULT = {
    #**PARAMS_DEFAULT,
    #"lr": 1e-3,
    "optimizer": torch.optim.Adam,
    "grad_clipping": 0.7921847463607343,
    "lr": 6.567116884019413e-05,
    "min_epoch": 73,
    "onecycle_epochs": 98,
    "onecycle_lr": 0.010640154204407485
}

PARAM_MAP = {}

def create_predictor(
    d_input=171,
    d_hid_gate=32,
    n_layers_self_gate=3,
    n_layers_cross_gate=3,
    activation_gate=torch.nn.ReLU,
    activation_final_gate=torch.nn.Sigmoid,
    bias_gate=True,
    cross_0=False,
    self_residual=False,
    cross_residual=False,
    predictor=ResultPredictor,
    d_hid_encoder=64,
    n_layers_encoder=6,
    activation_encoder=torch.nn.Identity,
    bias_encoder=True,
    d_final=256,
    d_hid_final=128,
    n_layers_final=1,
    activation_final=torch.nn.Identity,
    activation_final_head=torch.nn.Sigmoid,
    bias_final=False,
    n_layers_head=5,
    n_heads=3,
    dropout=0.04122349236114184,
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
            "n_heads": n_heads,
            "d_hid": d_hid_final,
            "n_layers": n_layers_head,
            "activation": activation_final,
            "activation_final": activation_final_head,
            "bias": bias_final,
            "dropout": dropout,
        },
        cross_0=cross_0,
        self_residual=self_residual,
        cross_residual=cross_residual,
    )
    return _predictor