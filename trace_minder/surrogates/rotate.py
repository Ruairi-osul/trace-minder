from calcium_clear.surrogates import rotate_traces
from typing import Optional
import pandas as pd
from .base import SurrogateGenerator


class Rotater(SurrogateGenerator):
    def __init__(
        self, time_col: str, increment: Optional[int] = None, copy: bool = False
    ):
        self.time_col = time_col
        self.increment = increment
        self.copy = copy

    def __call__(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        return rotate_traces(
            df_traces, time_col=self.time_col, increment=self.increment
        )
