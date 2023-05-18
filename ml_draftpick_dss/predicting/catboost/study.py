from .predictor import ResultPredictor
from .dataset import create_dataloader
from ..study import get_metric, LRS, EPOCHS


PARAM_SPACE = {
    "objective": ("categorical", ["Logloss", "CrossEntropy"]),
    "colsample_bylevel": ("log_float", 0.01, 0.1),
    "depth": ("int", 1, 12),
    "boosting_type": ("categorical", ["Ordered", "Plain"]),
    "bootstrap_type": (
        "categorical", ["Bayesian", "Bernoulli", "MVS"]
    ),
    'l2_leaf_reg': ('qloguniform', 0, 2, 1),
    'lr': ('float', 1e-3, 1e-1),
}

PARAMS_DEFAULT = {
}

PARAM_MAP = {}

def create_predictor(
    encoder=None,
    create_dataloader=None,
    objective="CrossEntropy",
    colsample_bylevel=1,
    depth=6,
    boosting_type="Plain",
    bootstrap_type="Bayesian",
    l2_leaf_reg=3.0,
    lr=0.1,
    predictor=ResultPredictor,
    **kwargs
):

    _predictor = predictor(
        objective=objective,
        colsample_bylevel=colsample_bylevel,
        depth=depth,
        boosting_type=boosting_type,
        bootstrap_type=bootstrap_type,
        l2_leaf_reg=l2_leaf_reg,
        lr=lr,
        **kwargs
    )
    return _predictor

def objective(
    datasets,
    create_predictor,
    #lrs=LRS[0],
    #epochs=EPOCHS[0],
    metric="val_loss",
    checkpoint_dir=f"checkpoints",
    log_dir=f"logs",
    autosave=False,
    trial=None,
    **predictor_kwargs
):
    if trial:
        print(f"Begin trial {trial.number}")
    train_loader, val_loader, test_loader = datasets

    """
    assert len(epochs) == 1
    if "od_wait" not in predictor_kwargs:
        predictor_kwargs["od_wait"] = max(x[0] for x in epochs)
    if "epochs" not in predictor_kwargs:
        predictor_kwargs["epochs"] = sum(x[1] for x in epochs)
    if "lr" not in predictor_kwargs:
        predictor_kwargs["lr"] = lrs[0]
    """

    predictor = create_predictor(**predictor_kwargs)
    
    predictor.prepare_training(
        train_loader,
        val_loader,
        checkpoint_dir=checkpoint_dir,
    )

    train_results = predictor.train(autosave=autosave)
    print(train_results)
    val_results = predictor.train(autosave=autosave, val=True)
    print(val_results)
    intermediate_value = get_metric({**train_results[1], **val_results[1]}, metric)
    train_metric = metric[4:] if metric.startswith("val_") else metric
    train_metric = train_results[1][train_metric]
    if trial:
        trial.report(intermediate_value, predictor.epoch)

    final_value = get_metric(train_metric, metric)
    return final_value
