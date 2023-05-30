from .modules import GlobalPooling1D, MEAN, PROD, SUM, MAX
import torch
import json
from ..util import mkdir
import math
import optuna

POOLINGS = {
    "global_average": GlobalPooling1D(MEAN),
    "global_avg": GlobalPooling1D(MEAN),
    "global_mean": GlobalPooling1D(MEAN),
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
PARAM_MAP = {
    "pooling": POOLINGS,
    "loss": LOSSES,
    "optimizer": OPTIMS,
    "activation": ACTIVATIONS,
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

def sample_parameter(trial, name, type, args, kwargs):
    return getattr(trial, f"suggest_{type}")(name, *args, **kwargs)

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
        kwargs = {}
        if type_0.startswith("bool_"):
            sample = trial.suggest_categorical(f"{k}_bool", [True, False])
            if not sample:
                type_1 = None
                param = 0
                params_raw[k] = param
                params[k] = param
                continue
            type_0 = type_0[5:]
            type_1 = type_0
        if type_0.startswith("log_"):
            type_1 = type_0[4:]
            kwargs["log"] = True
        if type_0 == "qloguniform":
            low, high, q = args
            type_1 = "float"
            param = round(math.exp(
                trial.suggest_float(f"{k}_qloguniform", low, high)
            ) / q) * q
            params_raw[k] = param
            params[k] = param
            continue
        if type_0 == "int_exp_2":
            low, high = args
            low = max(low, 1)
            low = math.log(low, 2)
            high = math.log(high, 2)
            assert low % 1 == 0
            assert high % 1 == 0
            type_1 = "int"
            param = int(math.pow(2, trial.suggest_int(f"{k}_exp_2", low, high)))
            params_raw[k] = param
            params[k] = param
            continue
        if type_0 in {"bool", "boolean"}:
            type_1, *args = BOOLEAN
        if type_0 in param_map:
            type_1 = "categorical"

        if type_1:
            param = sample_parameter(trial, k, type_1, args, kwargs)
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
        print(f"Begin trial {trial.number}")
        study_dir = f"studies/{id}"
        mkdir(study_dir)

        params, params_raw = sampler(trial, **sampler_kwargs)
        param_path = f"{study_dir}/params.json"
        with open(param_path, 'w') as f:
            try:
                json.dump(params_raw, f, indent=4)
            except TypeError as ex:
                print(params_raw)
                raise
        print(json.dumps(params_raw, indent=4))
        if checkpoint_dir:
            _checkpoint_dir = f"{study_dir}/{checkpoint_dir}"
        if log_dir:
            _log_dir = f"{study_dir}/{log_dir}"
        return objective(
            **objective_kwargs,
            **params, 
            checkpoint_dir=_checkpoint_dir,
            log_dir=_log_dir,
            trial=trial,
        )
    return f


def map_parameters(params_raw, param_map={}):
    param_map = {**PARAM_MAP, **param_map}
    ret = {}
    for k, v in params_raw.items():
        if k.endswith("_exp_2"):
            v = int(math.pow(2, v))
            k = k[:-6]
        else:
            for k0, v0 in param_map.items():
                if k0 in k:
                    v = v0[v]
        ret[k] = v
    return ret

def create_eval(
    eval, eval_kwargs={}, 
    params={}, params_raw={}, 
    mapping_kwargs={}, eval_id=1,
    checkpoint_dir="checkpoints", log_dir="logs"
):
    def f():
        id = eval_id
        print(f"Begin eval {eval_id}")
        study_dir = f"evals/{id}"
        mkdir(study_dir)

        _params = {
            **params,
            **map_parameters(params_raw, **mapping_kwargs)
        }
        param_path = f"{study_dir}/params.json"
        with open(param_path, 'w') as f:
            try:
                json.dump(params_raw, f, indent=4)
            except TypeError as ex:
                print(params_raw)
                raise
        print(json.dumps(params_raw, indent=4))
        if checkpoint_dir:
            _checkpoint_dir = f"{study_dir}/{checkpoint_dir}"
        if log_dir:
            _log_dir = f"{study_dir}/{log_dir}"
        return eval(
            **eval_kwargs,
            **_params, 
            checkpoint_dir=_checkpoint_dir,
            log_dir=_log_dir,
            eval_id=eval_id
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
    norm_crit=None,
    optimizer=torch.optim.Adam,
    grad_clipping=0,
    batch_size=64,
    metric="val_loss",
    checkpoint_dir=f"checkpoints",
    log_dir=f"logs",
    autosave="val_loss",
    trial=None,
    scheduler_config=SCHEDULER_CONFIGS[2],
    bin_crit=torch.nn.BCELoss(reduction="none"),
    onecycle_lr=1e-3,
    onecycle_epochs=50,
    onecycle_save="val_loss",
    lr=1e-5,
    min_epoch=50,
    max_epoch=200,
    wait=25,
    **predictor_kwargs
):
    if trial:
        print(f"Begin trial {trial.number}")
    print("Metric: ", metric)
    train_set, val_set, test_set = datasets
    train_loader = create_dataloader(train_set, batch_size=batch_size)
    val_loader = create_dataloader(val_set, batch_size=batch_size)
    test_loader = create_dataloader(test_set, batch_size=batch_size)

    batch_count = len(train_loader)
    scheduler_type, early_stopping = scheduler_config

    scheduler_kwargs = {}
    if scheduler_type == "onecycle":
        if not autosave:
            autosave = onecycle_save
        wait = onecycle_epochs//2
        scheduler_kwargs = {
            "steps": batch_count, 
            "epochs": onecycle_epochs,
            "lr": onecycle_lr
        }

    predictor = create_predictor(**predictor_kwargs)

    if lr is None:
        lr = predictor.find_lr(min_epoch=min_epoch).best_lr
    
    predictor.prepare_training(
        train_loader,
        val_loader,
        lr=lr,
        norm_crit=norm_crit,
        optimizer=optimizer,
        grad_clipping=grad_clipping,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        scheduler_type=scheduler_type,
        scheduler_kwargs=scheduler_kwargs,
        bin_crit=bin_crit,
    )
    print(predictor.summary())

    def _train(lr, min_epoch, max_epoch, prune=True, wait=wait, early_stopping_1=None):
        if lr is None:
            lr = predictor.find_lr(min_epoch=min_epoch).best_lr
        predictor.set_lr(lr)
        _early_stopping = None
        if early_stopping:
            if not early_stopping_1:
                _early_stopping = predictor.create_early_stopping_1(wait, max_epoch)
            else:
                _early_stopping = predictor.create_early_stopping_2(early_stopping_1, max_epoch)
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
                if prune and trial:
                    trial.report(intermediate_value, predictor.epoch)
            except optuna.TrialPruned as ex:
                print(str(ex))
                break
            finally:
                if trial.should_prune():
                    raise optuna.TrialPruned()
        return _early_stopping

    _early_stopping_1 = None
    if scheduler_type == "onecycle":
        print("Initial onecycle run")
        _early_stopping_1 = _train(lr, min(onecycle_epochs, min_epoch), onecycle_epochs, prune=False, wait=wait)
        print("Done onecycle run")
        predictor.load_checkpoint(onecycle_save)
        predictor.scheduler_type = "plateau"
        print("Continue with plateau")
    _train(lr, min_epoch, max_epoch, prune=True, wait=0, early_stopping_1=_early_stopping_1)
        
    #last_metrics = predictor.train(val=True)[1]
    best_metrics = predictor.best_metrics
    final_value = get_metric(best_metrics, metric)
    return final_value

def eval(
    datasets,
    create_predictor,
    create_dataloader,
    norm_crit=None,
    optimizer=torch.optim.Adam,
    grad_clipping=0,
    batch_size=64,
    metric="val_loss",
    checkpoint_dir=f"checkpoints",
    log_dir=f"logs",
    autosave="val_loss",
    eval_id=1,
    scheduler_config=SCHEDULER_CONFIGS[2],
    bin_crit=torch.nn.BCELoss(reduction="none"),
    onecycle_lr=10,
    onecycle_epochs=50,
    onecycle_save="val_loss",
    lr=1e-3,
    min_epoch=50,
    max_epoch=200,
    wait=25,
    **predictor_kwargs
):
    if eval_id:
        print(f"Begin eval {eval_id}")
    print("Metric: ", metric)
    train_set, val_set, test_set = datasets
    train_loader = create_dataloader(train_set, batch_size=batch_size)
    val_loader = create_dataloader(val_set, batch_size=batch_size)
    test_loader = create_dataloader(test_set, batch_size=batch_size)

    batch_count = len(train_loader)
    scheduler_type, early_stopping = scheduler_config

    scheduler_kwargs = {}
    if scheduler_type == "onecycle":
        if not autosave:
            autosave = onecycle_save
        wait = onecycle_epochs//2
        scheduler_kwargs = {
            "steps": batch_count, 
            "epochs": onecycle_epochs,
            "lr": onecycle_lr
        }

    predictor = create_predictor(**predictor_kwargs)

    if lr is None:
        lr = predictor.find_lr(min_epoch=min_epoch).best_lr
    
    predictor.prepare_training(
        train_loader,
        val_loader,
        lr=lr,
        norm_crit=norm_crit,
        optimizer=optimizer,
        grad_clipping=grad_clipping,
        checkpoint_dir=checkpoint_dir,
        log_dir=log_dir,
        scheduler_type=scheduler_type,
        scheduler_kwargs=scheduler_kwargs,
        bin_crit=bin_crit,
    )
    print(predictor.summary())

    if not hasattr(autosave, "__iter__"):
        autosave = [autosave]
    if not hasattr(onecycle_save, "__iter__"):
        onecycle_save = [onecycle_save]
    autosave = tuple(set([*autosave, *onecycle_save, metric]))

    _autosave = autosave
    def _train(lr, min_epoch, max_epoch, prune=True, wait=wait, early_stopping_1=None, autosave=_autosave):
        if lr is None:
            lr = predictor.find_lr(min_epoch=min_epoch).best_lr
        predictor.set_lr(lr)
        _early_stopping = None
        if early_stopping:
            if not early_stopping_1:
                _early_stopping = predictor.create_early_stopping_1(wait, max_epoch)
            else:
                _early_stopping = predictor.create_early_stopping_2(early_stopping_1, max_epoch)
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
            except optuna.TrialPruned as ex:
                print(str(ex))
                break
            finally:
                pass
        return _early_stopping

    _early_stopping_1 = None
    if scheduler_type == "onecycle":
        print("Initial onecycle run")
        _early_stopping_1 = _train(lr, min(onecycle_epochs, min_epoch), onecycle_epochs, prune=False, wait=wait, autosave=autosave)
        print("Done onecycle run")
        predictor.load_checkpoint(onecycle_save)
        predictor.scheduler_type = "plateau"
        print("Continue with plateau")
    _train(lr, min_epoch, max_epoch, prune=True, wait=0, early_stopping_1=_early_stopping_1, autosave=autosave)

    predictor.load_checkpoint(metric)

    #last_metrics = predictor.train(val=True)[1]
    best_metrics_train = predictor.best_metrics
    
    victory_preds, bin_pred, eval_metrics = predictor.eval(test_loader)

    return best_metrics_train, eval_metrics
