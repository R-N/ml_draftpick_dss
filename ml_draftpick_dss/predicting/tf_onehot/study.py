from ..transformer.study import study as _study
from .predictor import ResultPredictor

def study(datasets, d_input, *args, predictor=ResultPredictor, **kwargs):
    return _study(datasets, d_input, *args, predictor=predictor, **kwargs)
