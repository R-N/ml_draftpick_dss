from ..transformer.study import objective as _objective, LRS as _LRS, EPOCHS as _EPOCHS, PARAM_SPACE as _PARAM_SPACE
from .predictor import ResultPredictor

LRS = _LRS
EPOCHS = _EPOCHS

PARAM_SPACE = {
    **_PARAM_SPACE,
    "lrs": ("categorical", list(range(len(LRS)))),
    "epochs": ("categorical", list(range(len(EPOCHS)))),
}
PARAM_SPACE.pop("s_embed")

PARAM_MAP = {
    "lrs": LRS,
    "epochs": EPOCHS,
}

def objective(datasets, d_input, *args, predictor=ResultPredictor, **kwargs):
    return _objective(datasets, d_input, *args, predictor=predictor, **kwargs)
