from ..transformer.predictor import ResultPredictor as _ResultPredictor
from .model import ResultPredictorModel

class ResultPredictor(_ResultPredictor):
    def __init__(self, *args, model=ResultPredictorModel, **kwargs):
        super().__init__(*args, model=model, **kwargs)