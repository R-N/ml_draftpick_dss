import torch
from .dataset import create_dataloader, ResultDataset
from .predictor import ResultPredictor
from ..study import BOOLEAN, get_metric, LRS, EPOCHS, SCHEDULER_CONFIGS
import optuna


PARAM_SPACE = {
    "d_final": ("int_exp_2", 32, 512),
    "d_hid_encoder": ("int_exp_2", 32, 512),
    "n_layers_encoder": ("int_exp_2", 1, 16),
    "activation_encoder": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_encoder": BOOLEAN,
    "d_hid_final": ("int_exp_2", 32, 512),
    "n_layers_final": ("int_exp_2", 1, 16),
    "activation_final": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_final": BOOLEAN,
    "n_layers_head": ("int_exp_2", 1, 16),
    "dropout": ("float", 0.0, 0.3),
    "lrs": ("lrs", list(range(len(LRS)))),
    "epochs": ("epochs", list(range(len(EPOCHS)))),
    "scheduler_config": ("scheduler_config", list(range(len(SCHEDULER_CONFIGS)))),
    #"norm_crit": ("loss", ["mse"]),
    "optimizer": ("optimizer", ["adam", "adamw", "sgd"]),
    "grad_clipping": ("bool_float", 0.0, 1.0),
    "batch_size": ("int_exp_2", 32, 128),
    "pooling": ("categorical", ["concat", "diff", "mean", "prod"])
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
    pooling="concat",
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
):
    train_set, val_set, test_set = datasets
    train_loader = create_dataloader(train_set, batch_size=batch_size)
    val_loader = create_dataloader(val_set, batch_size=batch_size)
    test_loader = create_dataloader(test_set, batch_size=batch_size)

    batch_count = len(train_loader)
    scheduler_type, early_stopping = scheduler_config
    scheduler_kwargs = {"steps": batch_count} if scheduler_type == "onecycle" else {}

    if not isinstance(d_input, int):
        d_input = d_input.dim

    predictor = ResultPredictor(
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
            "heads": ["victory", "score", "duration"],
            "d_hid": d_hid_final,
            "n_layers": n_layers_head,
            "activation": activation_final,
            "bias": bias_final,
            "dropout": dropout,
        },
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
            predictor.inc_epoch()
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