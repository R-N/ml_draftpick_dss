from .predictor import ResultPredictor
from .dataset import create_dataloader


PARAM_SPACE = {
    "objective": ("categorical", ["Logloss", "CrossEntropy"]),
    "colsample_bylevel": ("log_float", 0.01, 0.1),
    "depth": ("int", 1, 12),
    "boosting_type": ("categorical", ["Ordered", "Plain"]),
    "bootstrap_type": (
        "categorical", ["Bayesian", "Bernoulli", "MVS"]
    ),
    'l2_leaf_reg': ('log_float', 0, 2, 1),
    'learning_rate': ('float', 1e-3, 1e-1),
}

PARAMS_DEFAULT = {
}

PARAM_MAP = {}

def create_predictor(
    objective="CrossEntropy",
    colsample_bylevel=1,
    depth=6,
    boosting_type="Plain",
    bootstrap_type="Bayesian",
    l2_leaf_reg=3.0,
    learning_rate=0.1,
    predictor=ResultPredictor,
    **kwargs
):
    if not isinstance(d_input, int):
        d_input = d_input.dim

    _predictor = predictor(
        objective=objective,
        colsample_bylevel=colsample_bylevel,
        depth=depth,
        boosting_type=boosting_type,
        bootstrap_type=bootstrap_type,
        l2_leaf_reg=l2_leaf_reg,
        learning_rate=learning_rate,
        **kwargs
    )
    return _predictor

