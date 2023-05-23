import torch
from .dataset import create_dataloader, ResultDataset
from .model import GlobalPooling1D
from .predictor import ResultPredictor
from ..embedding import scaled_sqrt_factory, create_embedding_sizes
from ..encoding import HeroLabelEncoder, HeroOneHotEncoder
from ..study import POOLINGS, LOSSES, OPTIMS, ACTIVATIONS, BOOLEAN, map_parameter, get_metric, SCHEDULER_CONFIGS
import optuna


PARAM_SPACE = {
    "s_embed": ("int", 3, 4),
    #"d_hid_encoder": ("int_exp_2", 32, 64),
    #"n_layers_encoder": ("int", 1, 2),
    #"activation_encoder": ("activation", ["relu", "elu"]),
    #"bias_encoder": BOOLEAN,
    #"n_heads_tf": ("int_exp_2", 2, 4),
    "d_hid_tf": ("int_exp_2", 8, 16),
    #"n_layers_tf": ("int", 2, 3),
    "activation_tf": ("activation", ["identity", "relu", "elu"]),
    #"d_hid_final": ("int_exp_2", 32, 64),
    "n_layers_final": ("int", 3, 4),
    "activation_final": ("activation", ["tanh", "elu"]),
    #"bias_final": BOOLEAN,
    "n_layers_head": ("int", 2, 3),
    "dropout": ("float", 0.1, 0.14),
    #"bidirectional": ("categorical", ["none", "none_right", "concat", "diff_left", "diff_right"]),
    "bidirectional": ("categorical", ["none", "concat", "diff_left", "diff_right"]),
    #"optimizer": ("optimizer", ["adam", "adamw"]),
    "grad_clipping": ("float", 0.4, 0.65),
    #"pooling": ("pooling", ["global_average", "global_max"]),
    "onecycle_lr": ("log_float", 1e-3, 1),
    "onecycle_epochs": ("int", 50, 100),
    "lr": ("log_float", 1e-5, 1e-1),
    "min_epoch": ("int", 25, 100),
}

PARAMS_DEFAULT = {
    "optimizer": torch.optim.AdamW,
}

PARAM_MAP = {}

def create_predictor(
    encoder,
    s_embed=4,
    d_hid_encoder=32,
    n_layers_encoder=2,
    activation_encoder=torch.nn.ELU,
    bias_encoder=False,
    n_heads_tf=4,
    d_hid_tf=8,
    n_layers_tf=2,
    activation_tf=torch.nn.ELU,
    d_hid_reducer=64,
    n_layers_reducer=1,
    activation_reducer=torch.nn.Identity,
    bias_reducer=False,
    d_hid_final=32,
    n_layers_final=3,
    activation_final=torch.nn.ELU,
    activation_final_head=torch.nn.Sigmoid,
    bias_final=True,
    n_layers_head=2,
    n_heads=3,
    dropout_reducer=0,
    dropout=0.1,
    pos_encoder=False,
    bidirectional="diff_right",
    pooling=GlobalPooling1D(),
    predictor=ResultPredictor,
    **kwargs,
):
    if isinstance(encoder, int):
        sizes = encoder
    elif isinstance(encoder, HeroLabelEncoder):
        sizes = create_embedding_sizes(
            encoder.x.columns[1:], 
            f=scaled_sqrt_factory(s_embed)
        )
    elif isinstance(encoder, torch.nn.Module):
        sizes = encoder
    elif hasattr(encoder, "dim"):
        sizes = encoder.dim
    else:
        raise ValueError(f"Unknown encoder type: {type(encoder)}")
    _predictor = predictor(
        sizes, 
        encoder_kwargs={
            "d_hid": d_hid_encoder,
            "n_layers": n_layers_encoder,
            "activation": activation_encoder,
            "bias": bias_encoder,
            "dropout": dropout,
        }, 
        tf_encoder_kwargs={
            "n_heads": n_heads_tf,
            "d_hid": d_hid_tf,
            "n_layers": n_layers_tf,
            "activation": activation_tf,
            "dropout": dropout,
        }, 
        tf_decoder_kwargs={
            "n_heads": n_heads_tf,
            "d_hid": d_hid_tf,
            "n_layers": n_layers_tf,
            "activation": activation_tf,
            "dropout": dropout,
        },
        reducer_kwargs={
            "d_hid": d_hid_reducer,
            "n_layers": n_layers_reducer,
            "activation": activation_reducer,
            "bias": bias_reducer,
            "dropout": dropout_reducer,
        },
        final_kwargs={
            "d_hid": d_hid_final,
            "n_layers": n_layers_final,
            "activation": activation_final,
            "bias": bias_final,
            "dropout": dropout,
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
        pos_encoder=pos_encoder,
        bidirectional=bidirectional,
        pooling=pooling,
        **kwargs
    )
    return _predictor
