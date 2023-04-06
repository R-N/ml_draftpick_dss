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
    "n_heads": ("int", 1, 4),
    "d_hid_tf": ("int", 32, 256, 32),
    "n_layers_tf": ("int", 1, 4),
    "activation_tf": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "d_hid_reducer": ("int", 32, 256, 32),
    "n_layers_reducer": ("int", 1, 4),
    "activation_reducer": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_reducer": BOOLEAN,
    "d_hid_final": ("int", 32, 256, 32),
    "n_layers_final": ("int", 1, 16),
    "activation_final": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_final": BOOLEAN,
    "n_layers_head": ("int", 1, 16),
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
    "batch_size": ("int", 32, 128, 32),
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

def objective(
    datasets,
    encoder,
    s_embed=2,
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
    lrs=LRS[0],
    epochs=EPOCHS[0],
    norm_crit=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam,
    grad_clipping=0,
    batch_size=128,
    metric="val_loss",
    checkpoint_dir=f"checkpoints",
    log_dir=f"logs",
    autosave=False,
    trial=None,
    scheduler_config=["plateau", False],
    predictor=ResultPredictor,
):
    train_set, val_set, test_set = datasets
    train_loader = create_dataloader(train_set, batch_size=batch_size)
    val_loader = create_dataloader(val_set, batch_size=batch_size)
    test_loader = create_dataloader(test_set, batch_size=batch_size)

    batch_count = len(train_loader)
    scheduler_type, early_stopping = scheduler_config
    scheduler_kwargs = {"steps": batch_count} if scheduler_type == "onecycle" else {}

    if isinstance(encoder, int):
        sizes = encoder
    elif isinstance(encoder, HeroLabelEncoder):
        sizes = create_embedding_sizes(
            encoder.x.columns[1:], 
            f=scaled_sqrt_factory(s_embed)
        )
    elif hasattr(encoder, "dim"):
        sizes = encoder.dim
    else:
        raise ValueError(f"Unknown encoder type: {type(encoder)}")
    predictor = predictor(
        sizes, 
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
    )
    predictor.prepare_training(
        train_loader,
        val_loader,
        lr=lrs[0],
        norm_crit=norm_crit,
        optimizer=optimizer,
        grad_clipping=grad_clipping,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        scheduler_type=scheduler_type,
        scheduler_kwargs=scheduler_kwargs,
    )
    print(predictor.summary())
    for lr, (min_epoch, max_epoch) in zip(lrs, epochs):
        if lr is None:
            lr = predictor.find_lr(min_epoch=min_epoch).best_lr
        predictor.set_lr(lr)
        if early_stopping:
            early_stopping = predictor.create_early_stopping_1(min_epoch, max_epoch)
        for i in range(max_epoch):
            train_results = predictor.train(autosave=autosave)
            print(train_results)
            val_results = predictor.train(autosave=autosave, val=True)
            print(val_results)
            intermediate_value = get_metric({**train_results[1], **val_results[1]}, metric)
            train_metric = metric[4:] if metric.startswith("val_") else metric
            train_metric = train_results[1][train_metric]
            if early_stopping:
                early_stopping(train_metric, intermediate_value)
            if trial:
                trial.report(intermediate_value, predictor.epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    #last_metrics = predictor.train(val=True)[1]
    best_metrics = predictor.best_metrics
    final_value = get_metric(best_metrics, metric)
    return final_value