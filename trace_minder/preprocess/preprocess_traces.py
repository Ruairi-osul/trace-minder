from typing import Any, Optional, Union, Sequence, Callable
from calcium_clear.resample.resample_pd import resample_traces
import numpy as np
import pandas as pd
from scipy.signal import medfilt
from scipy.ndimage import gaussian_filter1d


class TracePreprocessor:
    def __init__(
        self,
        max_time: Optional[float] = None,
        min_time: Optional[float] = None,
        time_selectors: Optional[
            Sequence[Callable[[Union[pd.Series, np.ndarray]], np.ndarray]]
        ] = None,
        standardize: bool = False,
        min_max: bool = False,
        resample_frequency: Optional[float] = None,
        resample_strategy: str = "ffill",
        time_col: str = "time",
        gaussian_sigma: Optional[float] = None,
        medfilt_kernel_size: Optional[int] = None,
    ):
        self.max_time = max_time
        self.min_time = min_time
        self.time_selectors = time_selectors
        self.standardize = standardize
        self.min_max = min_max
        self.resample_frequency = resample_frequency
        self.resample_strategy = resample_strategy
        self.time_col = time_col
        self.gaussian_sigma = gaussian_sigma
        self.medfilt_kernel_size = medfilt_kernel_size

    def subset_max_time(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        return df_traces[df_traces[self.time_col] <= self.max_time]

    def subset_min_time(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        return df_traces[df_traces[self.time_col] >= self.min_time]

    def subset_time_selectors(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        assert self.time_selectors is not None, "time_selectors must be provided"
        for selector in self.time_selectors:
            df_traces = df_traces[selector(df_traces[self.time_col])]
        return df_traces

    def resample_traces(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        return resample_traces(
            df_wide=df_traces,
            time_col=self.time_col,
            resample_frequency=self.resample_frequency,
            resample_strategy=self.resample_strategy,
        )

    def z_score_traces(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        df_traces.set_index(self.time_col, inplace=True)
        df_traces = (df_traces - df_traces.mean()) / df_traces.std()
        df_traces.reset_index(inplace=True)
        return df_traces

    def min_max_traces(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        df_traces.set_index(self.time_col, inplace=True)
        df_traces = (df_traces - df_traces.min()) / (df_traces.max() - df_traces.min())
        df_traces.reset_index(inplace=True)
        return df_traces

    def gaussian_filter_traces(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        df_traces.set_index(self.time_col, inplace=True)
        df_traces = df_traces.apply(gaussian_filter1d, sigma=self.gaussian_sigma)
        df_traces.reset_index(inplace=True)
        return df_traces

    def medfilt_traces(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        df_traces.set_index(self.time_col, inplace=True)
        df_traces = df_traces.apply(medfilt, kernel_size=self.medfilt_kernel_size)
        df_traces.reset_index(inplace=True)
        return df_traces

    def __call__(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        if self.max_time is not None:
            df_traces = self.subset_max_time(df_traces)
        if self.min_time is not None:
            df_traces = self.subset_min_time(df_traces)
        if self.time_selectors is not None:
            df_traces = self.subset_time_selectors(df_traces)
        if self.resample_frequency is not None:
            df_traces = self.resample_traces(df_traces)
        if self.standardize:
            df_traces = self.z_score_traces(df_traces)
        if self.min_max:
            df_traces = self.min_max_traces(df_traces)
        if self.gaussian_sigma is not None:
            df_traces = self.gaussian_filter_traces(df_traces)
        if self.medfilt_kernel_size is not None:
            df_traces = self.medfilt_traces(df_traces)
        return df_traces
