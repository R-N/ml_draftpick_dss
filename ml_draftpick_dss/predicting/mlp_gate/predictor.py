
from .model import ResultPredictorModel
from ..mlp.predictor import ResultPredictor as _ResultPredictor

class ResultPredictor(_ResultPredictor):
    def __init__(self, *args, model=ResultPredictorModel, **kwargs):
        super().__init__(*args, model=model, **kwargs)