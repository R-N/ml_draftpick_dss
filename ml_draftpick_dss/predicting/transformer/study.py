import torch
from .dataset import create_dataloader, ResultDataset
from .model import GlobalPooling1D
from .predictor import ResultPredictor
from ..embedding import scaled_sqrt_factory, create_embedding_sizes
from ..encoding import HeroLabelEncoder, HeroOneHotEncoder
from ..study import POOLINGS, LOSSES, OPTIMS, ACTIVATIONS, BOOLEAN, map_parameter

LRS = [
    [1e-3, 1e-4, 1e-5]
]
EPOCHS = [
    [100, 100, 100]
]

PARAM_SPACE = {
    "n_heads": ("int", 2, 8),
    "d_hid_tf": ("int", 32, 512, 32),
    "n_layers_tf": ("int", 1, 8),
    "activation_tf": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "d_hid_reducer": ("int", 32, 512, 32),
    "n_layers_reducer": ("int", 1, 4),
    "activation_reducer": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_reducer": BOOLEAN,
    "d_hid_final": ("int", 32, 512, 32),
    "n_layers_final": ("int", 1, 16),
    "activation_final": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_final": BOOLEAN,
    "dropout_reducer": ("float", 0.0, 0.3),
    "dropout": ("float", 0.0, 0.3),
    "pos_encoder": BOOLEAN,
    "bidirectional": BOOLEAN,
    "pooling": ("pooling", ["global_average", "global_product", "global_max"]),
    #"lrs": ("lrs", list(range(len(LRS)))),
    #"epochs": ("epochs", list(range(len(EPOCHS)))),
    #"norm_crit": ("loss", ["mse"]),
    "optimizer": ("optimizer", ["adam", "adamw", "sgd"]),
    "grad_clipping": ("bool_float", 0.0, 1.0),
    "batch_size": ("int", 32, 128, 32),
}

PARAM_MAP = {
    "lrs": LRS,
    "epochs": EPOCHS,
}

def objective(
    datasets,
    encoder,
    id=1,
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
    dropout_reducer=0,
    dropout=0.1,
    pos_encoder=False,
    bidirectional=False,
    pooling=GlobalPooling1D(),
    lrs=[1e-3, 1e-4, 1e-5],
    epochs=[100, 100, 100],
    norm_crit=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam,
    grad_clipping=0,
    batch_size=128,
    metric="val_loss",
    predictor=ResultPredictor,
):
    train_set, val_set, test_set = datasets
    train_loader = create_dataloader(train_set, batch_size=batch_size)
    val_loader = create_dataloader(val_set, batch_size=batch_size)
    test_loader = create_dataloader(test_set, batch_size=batch_size)

    if isinstance(encoder, int):
        sizes = encoder
    elif isinstance(encoder, HeroOneHotEncoder):
        sizes = encoder.dim
    else:
        sizes = create_embedding_sizes(
            encoder.x.columns[1:], 
            #f=scaled_sqrt_factory(2)
        )
    predictor = predictor(
        sizes, 
        encoder_kwargs={
            "n_heads": n_heads,
            "d_hid": d_hid_tf,
            "n_layers": n_layers_tf,
            "activation": activation_tf,
            "dropout": dropout,
        }, 
        decoder_kwargs={
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
            "bias": bias_final,
            "dropout": dropout,
        },
        pos_encoder=pos_encoder,
        bidirectional=bidirectional,
        pooling=pooling,
    )
    predictor.prepare_training(
        train_loader,
        val_loader,
        lr=lrs[0],
        norm_crit=norm_crit,
        optimizer=optimizer,
        grad_clipping=grad_clipping,
        checkpoint_dir=f"studies/{id}/checkpoints",
        log_dir=f"studies/{id}/logs",
    )
    for lr, epoch in zip(lrs, epochs):
        predictor.set_lr(lr)
        for i in range(epoch):
            print(predictor.train())
            print(predictor.train(val=True))
    #last_metrics = predictor.train(val=True)[1]
    best_metrics = predictor.best_metrics
    if isinstance(metric, str):
        return best_metrics[metric]
    return [best_metrics[m] for m in metric]