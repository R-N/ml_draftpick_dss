from ..transformer.study import objective as _objective, LRS, EPOCHS, PARAM_SPACE
from .predictor import ResultPredictor


PARAM_SPACE = {
    **PARAM_SPACE,
    "n_heads": ("int", 1, 2),
    "n_heads": ("int", 1, 2),
    "n_layers_tf": ("int", 1, 4),
    #"lrs": ("categorical", list(range(len(LRS)))),
    #"epochs": ("categorical", list(range(len(EPOCHS)))),
}
PARAM_SPACE.pop("s_embed")
PARAM_SPACE.pop("pos_encoder")

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
