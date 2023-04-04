import torch
from .dataset import create_dataloader, ResultDataset
from .predictor import ResultPredictor
from ..study import BOOLEAN, get_metric
import optuna

LRS = [
    [1e-3, 1e-4, 1e-5]
]
EPOCHS = [
    [100, 100, 100]
]

PARAM_SPACE = {
    "d_final": ("int", 32, 512, 32),
    "d_hid_encoder": ("int", 32, 512, 32),
    "n_layers_encoder": ("int", 1, 16),
    "activation_encoder": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_encoder": BOOLEAN,
    "d_hid_final": ("int", 32, 512, 32),
    "n_layers_final": ("int", 1, 16),
    "activation_final": ("activation", ["identity", "relu", "tanh", "sigmoid", "leakyrelu", "elu"]),
    "bias_final": BOOLEAN,
    "n_layers_head": ("int", 1, 16),
    "dropout": ("float", 0.0, 0.3),
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
    d_input=171,
    d_final=128,
    d_hid_encoder=128,
    n_layers_encoder=2,
    activation_encoder=torch.nn.ReLU,
    bias_encoder=True,
    d_hid_final=128,
    n_layers_final=3,
    activation_final=torch.nn.ReLU,
    bias_final=True,
    n_layers_head=1,
    dropout=0.1,
    lrs=[1e-3, 1e-4, 1e-5],
    epochs=[100, 100, 100],
    norm_crit=torch.nn.MSELoss(),
    optimizer=torch.optim.Adam,
    grad_clipping=0,
    batch_size=128,
    metric="val_loss",
    checkpoint_dir=f"checkpoints",
    log_dir=f"logs",
    autosave=False,
    trial=None,
):
    train_set, val_set, test_set = datasets
    train_loader = create_dataloader(train_set, batch_size=batch_size)
    val_loader = create_dataloader(val_set, batch_size=batch_size)
    test_loader = create_dataloader(test_set, batch_size=batch_size)

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
    )
    print(predictor.summary())
    for lr, epoch in zip(lrs, epochs):
        predictor.set_lr(lr)
        for i in range(epoch):
            train_results = predictor.train(autosave=autosave)
            print(train_results)
            val_results = predictor.train(autosave=autosave, val=True)
            print(val_results)
            if trial:
                intermediate_value = get_metric({**train_results[1], **val_results[1]}, metric)
                trial.report(intermediate_value, predictor.epoch)
                if trial.should_prune():
                    raise optuna.TrialPruned()

    #last_metrics = predictor.train(val=True)[1]
    best_metrics = predictor.best_metrics
    final_value = get_metric(best_metrics, metric)
    return final_value