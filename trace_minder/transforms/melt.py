import pandas as pd
from typing import Any, Optional
from typing import Union, List


class Melter:
    def __init__(
        self,
        id_vars: Union[str, List[str]],
        created_cell_col: str = "cell_id",
        created_value_col: str = "value",
        cast_cell_col_type: Optional[str] = None,
        drop_cols: Optional[List[str]] = None,
    ):
        self.id_vars = id_vars
        self.created_cell_col = created_cell_col
        self.created_value_col = created_value_col
        self.cast_cell_col_type = cast_cell_col_type
        self.drop_cols = drop_cols

    def _cast_cell_col(self, df: pd.DataFrame) -> pd.DataFrame:
        if self.cast_cell_col_type is not None:
            df[self.created_cell_col] = df[self.created_cell_col].astype(
                self.cast_cell_col_type
            )
        return df

    def __call__(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        if self.drop_cols is not None:
            df_traces = df_traces.drop(columns=self.drop_cols)
        df_traces = pd.melt(
            df_traces,
            id_vars=self.id_vars,
            var_name=self.created_cell_col,
            value_name=self.created_value_col,
        )
        df_traces = self._cast_cell_col(df_traces)
        return df_traces
