from calcium_clear.surrogates import rotate_traces
from typing import Optional
import pandas as pd


class SurrogateGenerator:
    """
    A base class for callables that take df_traces and return surrogate df_traces.
    """

    def __call__(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        raise NotImplementedError