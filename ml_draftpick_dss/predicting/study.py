from .modules import GlobalPooling1D, MEAN, PROD, SUM, MAX
import torch

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
    [1e-3, 1e-4, 1e-5]
]
EPOCHS = [
    [100, 100, 100]
]
PARAM_MAP = {
    "pooling": POOLINGS,
    "loss": LOSSES,
    "optimizer": OPTIMS,
    "activation": ACTIVATIONS,
    "lrs": LRS,
    "epochs": EPOCHS
}
BOOLEAN = ("categorical", [True, False])

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
    for k, v in param_space.items():
        type_0, *args = v
        type_1 = type_0
        if type_0.startswith("bool_"):
            sample = trial.suggest_categorical(f"{k}_bool", [True, False])
            if not sample:
                params[k] = 0
                continue
            type_1 = type_0[5:]
        elif type_0 in {"bool", "boolean"}:
            type_1, *args = BOOLEAN
        elif type_0 in param_map:
            type_1 = "categorical"
        param = sample_parameter(trial, k, type_1, args)
        if type_0 in param_map:
            param = map_parameter(param, param_map[type_0])
        params[k] = param
    params["id"] = trial.number
    return params

def create_objective(objective, sampler=sample_parameters, objective_kwargs={}, sampler_kwargs={}):
    def f(trial):
        return objective(**sampler(trial, **sampler_kwargs), **objective_kwargs)
    return f
