from .rotate import Rotater
from .sample import TraceSampler
import pandas as pd


class SurrogateTemplate:
    def __init__(
        self,
        rotater: Rotater | None = None,
        sampler: TraceSampler | None = None,
        copy: bool = True,
    ):
        self.rotater = rotater
        self.sampler = sampler
        self.copy = copy

    def __call__(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        if self.copy:
            df_traces = df_traces.copy()
        if self.rotater is not None:
            df_traces = self.rotater(df_traces)
        if self.sampler is not None:
            df_traces = self.sampler(df_traces)
        return df_traces
