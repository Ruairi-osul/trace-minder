from .base import SurrogateGenerator
import pandas as pd


class Identity(SurrogateGenerator):
    """
    A callable that returns the input df_traces.
    """

    def __call__(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        return df_traces