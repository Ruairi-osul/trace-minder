from .preprocess import AllTimeDecodePreprocesser, EventPreprocesser
from .trainers import Trainer
from sklearn.base import BaseEstimator
from typing import Callable


class AllTimeDecodeDispatcher:
    ...


class EventDispatcher(AllTimeDecodeDispatcher):
    def __init__(
        self,
        preprocesser: EventPreprocesser,
        decoder_factory: Callable,
        trainer: Trainer,
    ):
        self.preprocesser = preprocesser
        self.decoder = decoder_factory()
        self.trainer = trainer

    def __call__(self, rotate: bool = False):
        X, y = self.preprocesser.get_X_y(rotate=rotate)
        score = self.trainer(self.decoder, X, y)
        return score
