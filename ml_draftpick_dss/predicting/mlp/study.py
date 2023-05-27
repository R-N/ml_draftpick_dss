import torch
from .dataset import create_dataloader
from .predictor import ResultPredictor
from ..study import BOOLEAN, get_metric, SCHEDULER_CONFIGS
import optuna


PARAM_SPACE = {
    "d_final": ("int_exp_2", 128, 512),
    "d_hid_encoder": ("int_exp_2", 16, 32),
    #"n_layers_encoder": ("int", 6, 7),
    #"activation_encoder": ("activation", ["tanh", "elu"]),
    #"d_hid_final": ("int_exp_2", 64, 128),
    "n_layers_final": ("int", 1, 3),
    "activation_final": ("activation", ["relu", "leakyrelu"]),
    #"bias_final": BOOLEAN,
    #"n_layers_head": ("int", 1, 2),
    "dropout": ("float", 0.06, 0.12),
    #"optimizer": ("optimizer", ["adamw", "sgd"]),
    "grad_clipping": ("float", 1, 1.8),
    "pooling": ("categorical", ["concat", "diff_left"]),
    "onecycle_lr": ("log_float", 1e-4, 1e-2),
    "onecycle_epochs": ("int", 25, 60),
    "lr": ("log_float", 1e-6, 1e-4),
    "min_epoch": ("int", 25, 50),
}

PARAMS_DEFAULT = {
    "optimizer": torch.optim.AdamW,
}

PARAM_MAP = {}

def create_predictor(
    d_input=171,
    d_hid_encoder=32,
    n_layers_encoder=7,
    activation_encoder=torch.nn.ELU,
    bias_encoder=True,
    d_final=128,
    d_hid_final=64,
    n_layers_final=2,
    activation_final=torch.nn.LeakyReLU,
    bias_final=False,
    n_layers_head=2,
    dropout=0.1,
    pooling="concat",
    n_heads=3,
    activation_final_head=torch.nn.Sigmoid,
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