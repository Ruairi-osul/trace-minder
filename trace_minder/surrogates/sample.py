from calcium_clear.surrogates import sample_traces
import pandas as pd
from .base import SurrogateGenerator


class TraceSampler(SurrogateGenerator):
    """
    A class used to sample traces from a DataFrame.

    This class is a callable that wraps the `sample_traces` function. When an instance of this class is called with a DataFrame, it samples traces from the DataFrame using the parameters specified during the instantiation of the class.

    Attributes:
        time_col (str): The name of the time column. Defaults to "time".
        n_retained (int | None): The number of columns to retain. If None, all columns are retained. Defaults to None.
        frac_retained (float | None): The fraction of columns to retain. If None, all columns are retained. Defaults to None.
        with_replacement (bool): Whether to sample with replacement. Defaults to False.
        other_cols (list | None): List of other column names to retain. If None, no other columns are retained. Defaults to None.

    Methods:
        __call__(df_traces: pd.DataFrame) -> pd.DataFrame:
            Sample traces from `df_traces` using the parameters specified during the instantiation of the class.

    Examples:
        >>> sampler = Sampler("time", 2, 0.5, True, ["col1", "col2"])
        >>> sampled_df = sampler(df)
    """

    def __init__(
        self,
        time_col: str = "time",
        n_retained: int | None = None,
        frac_retained: float | None = None,
        with_replacement: bool = False,
        other_cols: list | None = None,
    ):
        self.time_col = time_col
        self.n_retained = n_retained
        self.frac_retained = frac_retained
        self.with_replacement = with_replacement
        self.other_cols = other_cols

    def __call__(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        return sample_traces(
            df_traces,
            time_col=self.time_col,
            n_retained=self.n_retained,
            frac_retained=self.frac_retained,
            with_replacement=self.with_replacement,
            other_cols=self.other_cols,
        )
