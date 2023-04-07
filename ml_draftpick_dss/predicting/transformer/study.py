import torch
from .dataset import create_dataloader, ResultDataset
from .model import GlobalPooling1D
from .predictor import ResultPredictor
from ..embedding import scaled_sqrt_factory, create_embedding_sizes
from ..encoding import HeroLabelEncoder, HeroOneHotEncoder
from ..study import POOLINGS, LOSSES, OPTIMS, ACTIVATIONS, BOOLEAN, map_parameter, get_metric, LRS, EPOCHS, SCHEDULER_CONFIGS
import optuna


PARAM_SPACE = {
    "s_embed": ("int", 1, 4),
    "d_hid_encoder": ("int_exp_2", 32, 128),
    "n_layers_encoder": ("int", 1, 8),
    "activation_encoder": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_encoder": BOOLEAN,
    "n_heads": ("int_exp_2", 1, 16),
    "d_hid_tf": ("int_exp_2", 32, 128),
    "n_layers_tf": ("int", 1, 4),
    "activation_tf": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    #"d_hid_reducer": ("int", 32, 256, 32),
    #"n_layers_reducer": ("int", 1, 4),
    #"activation_reducer": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    #"bias_reducer": BOOLEAN,
    "d_hid_final": ("int_exp_2", 32, 128),
    "n_layers_final": ("int_exp_2", 1, 8),
    "activation_final": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_final": BOOLEAN,
    "n_layers_head": ("int_exp_2", 1, 8),
    "dropout_reducer": ("float", 0.0, 0.3),
    "dropout": ("float", 0.0, 0.3),
    "pos_encoder": BOOLEAN,
    "bidirectional": BOOLEAN,
    "pooling": ("pooling", ["global_average", "global_product", "global_max"]),
    "lrs": ("lrs", list(range(len(LRS)))),
    "epochs": ("epochs", list(range(len(EPOCHS)))),
    "scheduler_config": ("scheduler_config", list(range(len(SCHEDULER_CONFIGS)))),
    #"norm_crit": ("loss", ["mse"]),
    "optimizer": ("optimizer", ["adam", "adamw", "sgd"]),
    "grad_clipping": ("bool_float", 0.0, 1.0),
    "batch_size": ("int_exp_2", 32, 128),
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
    encoder,
    s_embed=2,
    d_hid_encoder=128,
    n_layers_encoder=2,
    activation_encoder=torch.nn.ReLU,
    bias_encoder=True,
    n_heads=2,
    d_hid_tf=128,
    n_layers_tf=2,
    activation_tf=torch.nn.ReLU,
    d_hid_reducer=128,
    n_layers_reducer=1,
    activation_reducer=torch.nn.Identity,
    bias_reducer=False,
    d_hid_final=128,
    n_layers_final=3,
    activation_final=torch.nn.ReLU,
    bias_final=True,
    n_layers_head=1,
    dropout_reducer=0,
    dropout=0.1,
    pos_encoder=False,
    bidirectional=False,
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
            "n_heads": n_heads,
            "d_hid": d_hid_tf,
            "n_layers": n_layers_tf,
            "activation": activation_tf,
            "dropout": dropout,
        }, 
        tf_decoder_kwargs={
            "n_heads": n_heads,
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
            "heads": ["victory", "score", "duration"],
            "d_hid": d_hid_final,
            "n_layers": n_layers_head,
            "activation": activation_final,
            "bias": bias_final,
            "dropout": dropout,
        },
        pos_encoder=pos_encoder,
        bidirectional=bidirectional,
        pooling=pooling,
        **kwargs
    )
    return _predictor
