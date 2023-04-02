
from .model import ResultPredictorModel
from ..transformer.predictor import ResultPredictor as _ResultPredictor

class ResultPredictor(_ResultPredictor):
    def __init__(self, *args, model=ResultPredictorModel, **kwargs):
        super().__init__(*args, model=model, **kwargs)