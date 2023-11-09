from typing import Any
from typing import Callable, Union
import pandas as pd


class BaseAligner:
    def __init__(
        self,
        t_before: float,
        t_after: float,
        time_col: str = "time",
        created_event_index_col: str = "event_idx",
        created_aligned_time_col: str = "aligned_time",
        created_neuron_col: str = "neuron",
        created_value_col: str = "value",
        round_precision: int = 1,
        drop_non_aligned: bool = True,
        average_trace_agg_func: Union[str, Callable] = "mean",
        copy: bool = True,
    ):
        self.t_before = t_before
        self.t_after = t_after
        self.time_col = time_col
        self.created_event_index_col = created_event_index_col
        self.created_aligned_time_col = created_aligned_time_col
        self.round_precision = round_precision
        self.drop_non_aligned = drop_non_aligned
        self.created_neuron_col = created_neuron_col
        self.created_value_col = created_value_col
        self.average_trace_agg_func = average_trace_agg_func
        self.copy = copy

    def align(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        ...

    def align_long(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        ...

    def average_trace(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        ...

    def average_trace_long(self, *args: Any, **kwargs: Any) -> pd.DataFrame:
        ...
