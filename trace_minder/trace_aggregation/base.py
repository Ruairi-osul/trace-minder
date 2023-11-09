from calcium_clear.trace_aggregation import (
    prepost_agg,
    prepost_agg_long,
    prepost_diff,
    event_agg,
    event_agg_long,
)
from typing import Optional, Union, Callable, Any
import pandas as pd
import numpy as np


class TraceAggregator:
    def __init__(
        self,
        aligned_time_col: str = "aligned_time",
        event_idx_col: Optional[str] = "event_idx",
        neuron_col: str = "neuron",
        value_col: str = "value",
        agg_func: str = "auc",
        created_pre_post_col: str = "pre_post",
        pre_indicator: str = "pre",
        post_indicator: str = "post",
        time_sep: float = 0,
        created_diff_col: str = "post_sub_pre",
        drop_pre_post_cols: bool = False,
    ):
        self.aligned_time_col = aligned_time_col
        self.event_idx_col = event_idx_col
        self.neuron_col = neuron_col
        self.value_col = value_col
        self.agg_func = agg_func
        self.created_pre_post_col = created_pre_post_col
        self.pre_indicator = pre_indicator
        self.post_indicator = post_indicator
        self.time_sep = time_sep
        self.created_diff_col = created_diff_col
        self.drop_pre_post_cols = drop_pre_post_cols

    def aggregate(self, df_aligned: pd.DataFrame) -> pd.DataFrame:
        ...


class PrePostAggregator(TraceAggregator):
    """Aggregate pre and post separately."""

    def aggregate(self, df_aligned: pd.DataFrame) -> pd.DataFrame:
        """One one row for event with columns of aggregated pre/post"""
        return prepost_agg(
            df_aligned_long=df_aligned,
            aligned_time_col=self.aligned_time_col,
            event_idx_col=self.event_idx_col,
            neuron_col=self.neuron_col,
            value_col=self.value_col,
            agg_func=self.agg_func,
            created_pre_post_col=self.created_pre_post_col,
            pre_indicator=self.pre_indicator,
            post_indicator=self.post_indicator,
            time_sep=self.time_sep,
        )

    def aggregate_long(self, df_aligned: pd.DataFrame) -> pd.DataFrame:
        """One one row for each pre/post of each event."""
        return prepost_agg_long(
            df_aligned_long=df_aligned,
            aligned_time_col=self.aligned_time_col,
            event_idx_col=self.event_idx_col,
            neuron_col=self.neuron_col,
            value_col=self.value_col,
            agg_func=self.agg_func,
            created_pre_post_col=self.created_pre_post_col,
            pre_indicator=self.pre_indicator,
            post_indicator=self.post_indicator,
            time_sep=self.time_sep,
        )

    def prepost_diff(self, df_aligned: pd.DataFrame) -> pd.DataFrame:
        """Aggregate each event's pre and post separately and take the difference."""
        return prepost_diff(
            df_aligned_long=df_aligned,
            aligned_time_col=self.aligned_time_col,
            event_idx_col=self.event_idx_col,
            neuron_col=self.neuron_col,
            value_col=self.value_col,
            prepost_agg_func=self.agg_func,
            pre_indicator=self.pre_indicator,
            post_indicator=self.post_indicator,
            time_sep=self.time_sep,
            created_diff_col=self.created_diff_col,
            drop_pre_post_cols=self.drop_pre_post_cols,
        )


class WholeTraceAggregator:
    """
    Takes a function that aggregates the whole trace.

    The function is passed a pd.DataFrame for each session with columns
      - aligned_time
      - pre_post
      - value
    """

    def __init__(
        self,
        aligned_time_col: str = "aligned_time",
        event_idx_col: Optional[str] = "event_idx",
        neuron_col: str = "neuron",
        value_col: str = "value",
        agg_func: Union[str, Callable[[pd.DataFrame], Any]] = "auc_post_minus_pre",
        created_pre_post_col: str = "pre_post",
        pre_indicator: str = "pre",
        post_indicator: str = "post",
        time_sep: float = 0,
        created_diff_col: str = "post_sub_pre",
        drop_pre_post_cols: bool = False,
    ):
        self.aligned_time_col = aligned_time_col
        self.event_idx_col = event_idx_col
        self.neuron_col = neuron_col
        self.value_col = value_col
        self.agg_func = agg_func
        self.created_pre_post_col = created_pre_post_col
        self.pre_indicator = pre_indicator
        self.post_indicator = post_indicator
        self.time_sep = time_sep
        self.created_diff_col = created_diff_col
        self.drop_pre_post_cols = drop_pre_post_cols

    def aggregate(self, df_aligned: pd.DataFrame) -> pd.DataFrame:
        return event_agg(
            df_aligned_long=df_aligned,
            aligned_time_col=self.aligned_time_col,
            event_idx_col=self.event_idx_col,
            neuron_col=self.neuron_col,
            value_col=self.value_col,
            agg_func=self.agg_func,
            created_pre_post_col=self.created_pre_post_col,
            pre_indicator=self.pre_indicator,
            post_indicator=self.post_indicator,
            time_sep=self.time_sep,
        )

    def aggregate_long(self, df_aligned: pd.DataFrame) -> pd.DataFrame:
        return event_agg_long(
            df_aligned_long=df_aligned,
            aligned_time_col=self.aligned_time_col,
            event_idx_col=self.event_idx_col,
            neuron_col=self.neuron_col,
            value_col=self.value_col,
            agg_func=self.agg_func,
            created_pre_post_col=self.created_pre_post_col,
            pre_indicator=self.pre_indicator,
            post_indicator=self.post_indicator,
            time_sep=self.time_sep,
        )
