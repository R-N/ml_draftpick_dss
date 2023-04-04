from ..transformer.study import objective as _objective, LRS, EPOCHS, PARAM_SPACE
from .predictor import ResultPredictor


PARAM_SPACE = {
    **PARAM_SPACE,
    "lrs": ("categorical", list(range(len(LRS)))),
    "epochs": ("categorical", list(range(len(EPOCHS)))),
}
PARAM_SPACE.pop("s_embed")

PARAM_MAP = {}
"""
LRS = LRS
EPOCHS = EPOCHS
PARAM_MAP = {
    "lrs": LRS,
    "epochs": EPOCHS,
}
"""

def objective(datasets, d_input, *args, predictor=ResultPredictor, **kwargs):
    return _objective(datasets, d_input, *args, predictor=predictor, **kwargs)
