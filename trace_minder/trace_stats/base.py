import numpy as np


class BaseStat:
    """
    A base class for statistics to summarize traces (both time and values).
    """

    def __call__(self, time: np.ndarray, values: np.ndarray) -> float:
        raise NotImplementedError
