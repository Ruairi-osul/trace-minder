from calcium_clear.surrogates import rotate_traces
from typing import Optional
import pandas as pd
from .base import SurrogateGenerator


class Rotater(SurrogateGenerator):
    """
    A class used to rotate traces in a DataFrame.

    This class is a callable that wraps the `rotate_traces` calcium_clear function. When an instance of this class is called with a DataFrame, it rotates the traces in the DataFrame using the parameters specified during the instantiation of the class.

    Attributes:
        time_col (str): The name of the time column.
        increment (int | None): The number of positions to rotate the traces. If None, a random increment is chosen. Defaults to None.
        copy (bool): Whether to copy the DataFrame before rotating the traces. Defaults to False.

    Methods:
        __call__(df_traces: pd.DataFrame) -> pd.DataFrame:
            Rotate the traces in `df_traces` using the parameters specified during the instantiation of the class.

    Examples:
        >>> rotater = Rotater("time", 2, True)
        >>> rotated_df = rotater(df)
    """

    def __init__(
        self, time_col: str, increment: Optional[int] = None, copy: bool = False
    ):
        self.time_col = time_col
        self.increment = increment
        self.copy = copy

    def __call__(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        return rotate_traces(
            df_traces, time_col=self.time_col, increment=self.increment, copy=self.copy
        )
