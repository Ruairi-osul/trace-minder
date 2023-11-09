from observational_fear_repeated.stats import auc
from .base import BaseStat
import numpy as np


class AucPostSubPre(BaseStat):
    """
    Calculates the area under the curve of the post-conditioning period minus the area under the curve of the
    pre-conditioning period. If absolute is True, the absolute value of the difference is returned.

    Parameters
    ----------
    to_1 : bool
        Whether to normalize the data to 1 before calculating the area under the curve.
    absolute : bool
        Whether to return the absolute value of the difference.

    Returns
    -------
    float
        The difference between the area under the curve of the post-conditioning period and the area under the curve
    """

    def __init__(self, to_1: bool = True, absolute: bool = False):
        self.to_1 = to_1
        self.absolute = absolute

    def __call__(self, time: np.ndarray, values: np.ndarray) -> float:
        pre_data = values[time < 0]
        post_data = values[time > 0]

        auc_pre = auc(pre_data, to_1=self.to_1)
        auc_post = auc(post_data, to_1=self.to_1)

        diff = auc_post - auc_pre

        if self.absolute:
            diff = np.abs(diff)

        return diff
