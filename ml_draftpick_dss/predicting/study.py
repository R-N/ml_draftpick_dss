from .modules import GlobalPooling1D, MEAN, PROD, SUM, MAX
import torch
import json
from ..util import mkdir
import math
import optuna

POOLINGS = {
    "global_average": GlobalPooling1D(MEAN),
    "global_product": GlobalPooling1D(PROD),
    "global_sum": GlobalPooling1D(SUM),
    "global_max": GlobalPooling1D(MAX),
}
LOSSES = {
    "mse": torch.nn.MSELoss(reduction="none"),
}
OPTIMS = {
    "adam": torch.optim.Adam,
    "adamw": torch.optim.AdamW,
    "sgd": torch.optim.SGD,
}
SCHEDULER_CONFIGS = [
    ("plateau", False),
    ("plateau", True),
    ("onecycle", True)
]
ACTIVATIONS = {
    "identity": torch.nn.Identity,
    "relu": torch.nn.ReLU,
    "tanh": torch.nn.Tanh,
    "sigmoid": torch.nn.Sigmoid,
    "leakyrelu": torch.nn.LeakyReLU,
    "elu": torch.nn.ELU,
    "selu": torch.nn.SELU,
    #"gelu": torch.nn.GELU,
}
LRS = [
    [1e-2],
    [1e-3],
]
EPOCHS = [
    [(50, 200)],
    [(50, 200)],
]
PARAM_MAP = {
    "pooling": POOLINGS,
    "loss": LOSSES,
    "optimizer": OPTIMS,
    "activation": ACTIVATIONS,
    "lrs": LRS,
    "epochs": EPOCHS,
    "scheduler_config": SCHEDULER_CONFIGS
}
BOOLEAN = ("categorical", [True, False])

def get_metric(best_metrics, metric):
    if isinstance(metric, str):
        return best_metrics[metric]
    return [best_metrics[m] for m in metric]

def map_parameter(param, source):
    try:
        return source[param]
    except Exception as ex:
        return param

def sample_parameter(trial, name, type, args):
    return getattr(trial, f"suggest_{type}")(name, *args)

def sample_parameters(trial, param_space, param_map={}):
    param_map = {
        **PARAM_MAP,
        **param_map,
    }
    params = {}
    params_raw = {}
    for k, v in param_space.items():
        type_0, *args = v
        type_1 = type_0
        if type_0.startswith("bool_"):
            sample = trial.suggest_categorical(f"{k}_bool", [True, False])
            if not sample:
                type_1 = None
                param = 0
                params_raw[k] = param
                params[k] = param
                continue
            type_0 = type_0[5:]
            type_1 = type_0[5:]
        if type_0 == "int_exp_2":
            low, high = args
            mul = high/low
            power = math.log(mul, 2)
            assert power % 1 == 0
            type_1 = "int"
            param = int(low * math.pow(2, trial.suggest_int(f"{k}_exp_2", 0, power)))
            params_raw[k] = param
            params[k] = param
            continue
        if type_0 in {"bool", "boolean"}:
            type_1, *args = BOOLEAN
        if type_0 in param_map:
            type_1 = "categorical"

        if type_1:
            param = sample_parameter(trial, k, type_1, args)
        params_raw[k] = param
        if type_0 in param_map:
            param = map_parameter(param, param_map[type_0])
        params[k] = param
    #params["id"] = trial.number
    return params, params_raw

def create_objective(
    objective, sampler=sample_parameters, 
    objective_kwargs={}, sampler_kwargs={}, 
    checkpoint_dir="checkpoints", log_dir="logs"
):
    def f(trial):
        id = trial.number
        study_dir = f"studies/{id}"
        mkdir(study_dir)

        params, params_raw = sampler(trial, **sampler_kwargs)
        param_path = f"{study_dir}/params.json"
        with open(param_path, 'w') as f:
            json.dump(params_raw, f, indent=4)
        print(json.dumps(params_raw, indent=4))
        if checkpoint_dir:
            _checkpoint_dir = f"{study_dir}/{checkpoint_dir}"
        if log_dir:
            _log_dir = f"{study_dir}/{log_dir}"
        _objective_kwargs = {
            **params, 
            **objective_kwargs,
        }
        return objective(
            **_objective_kwargs,
            checkpoint_dir=_checkpoint_dir,
            log_dir=_log_dir,
        )
    return f

def calc_basket(min_resource, max_resource, reduction_factor):
    basket = math.log(max_resource/min_resource, reduction_factor)
    return basket

def calc_reduction(min_resource, max_resource, basket=4):
    reduction_factor = math.pow(max_resource/min_resource, basket)
    return reduction_factor

def calc_min_resource(max_resource, basket=4, reduction_factor=3):
    min_resource = max_resource / math.pow(reduction_factor, basket)
    return min_resource

def objective(
    datasets,
    create_predictor,
    create_dataloader,
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
    **predictor_kwargs
):
    train_set, val_set, test_set = datasets
    train_loader = create_dataloader(train_set, batch_size=batch_size)
    val_loader = create_dataloader(val_set, batch_size=batch_size)
    test_loader = create_dataloader(test_set, batch_size=batch_size)

    batch_count = len(train_loader)
    scheduler_type, early_stopping = scheduler_config
    scheduler_kwargs = {"steps": batch_count} if scheduler_type == "onecycle" else {}

    predictor = create_predictor(**predictor_kwargs)
    
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
        _early_stopping = None
        if early_stopping:
            _early_stopping = predictor.create_early_stopping_1(min_epoch, max_epoch)
        for i in range(max_epoch):
            try:
                train_results = predictor.train(autosave=autosave)
                print(train_results)
                val_results = predictor.train(autosave=autosave, val=True)
                print(val_results)
                predictor.inc_epoch()
                intermediate_value = get_metric({**train_results[1], **val_results[1]}, metric)
                train_metric = metric[4:] if metric.startswith("val_") else metric
                train_metric = train_results[1][train_metric]
                if _early_stopping:
                    _early_stopping(train_metric, intermediate_value)
                if trial:
                    trial.report(intermediate_value, predictor.epoch)
                    if trial.should_prune():
                        raise optuna.TrialPruned()
            except optuna.TrialPruned as ex:
                print(str(ex))
                break

    #last_metrics = predictor.train(val=True)[1]
    best_metrics = predictor.best_metrics
    final_value = get_metric(best_metrics, metric)
    return final_value
