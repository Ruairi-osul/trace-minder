from .base import BaseStat
from .auc import AucPostSubPre
from typing import Callable
import numpy as np


def get_trace_stat(name: str) -> BaseStat:
    if name == "auc":
        return AucPostSubPre()
    elif name == "auc-absolute":
        return AucPostSubPre(absolute=True)
    else:
        raise ValueError(f"Unknown trace stat: {name}")
