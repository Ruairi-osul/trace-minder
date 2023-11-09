from .preprocess import AllTimeDecodePreprocesser, EventPreprocesserSingleSession
from .trainers import Trainer
from .dispatcher import EventDispatcher
from sklearn.base import BaseEstimator
from typing import Any, Callable, Tuple
from joblib import Parallel, delayed
import numpy as np


class AllTimeBoostrapper:
    ...


class EventBootstrapper:
    def __init__(
        self,
        dispatcher: EventDispatcher,
        n_boot: int = 75,
        n_jobs: int = -1,
    ):
        self.dispatcher = dispatcher
        self.n_boot = n_boot
        self.n_jobs = n_jobs

    def get_observed(self):
        return self.dispatcher(rotate=False)

    def get_reps(self):
        reps = Parallel(n_jobs=self.n_jobs)(
            delayed(self.dispatcher)(rotate=True) for _ in range(self.n_boot)
        )
        reps = np.asarray(reps)
        return reps

    def get_p(self, observed: float, reps: np.ndarray):
        return (reps >= observed).mean()

    def __call__(self) -> Tuple[float, float]:
        observed = self.get_observed()
        reps = self.get_reps()
        p = self.get_p(observed, reps)
        return observed, p
