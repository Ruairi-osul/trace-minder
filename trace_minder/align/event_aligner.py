import pandas as pd
import numpy as np
from typing import Optional
from calcium_clear.surrogates import rotate_traces
from calcium_clear.align import (
    align_to_events,
    align_to_events_long,
    average_trace,
    average_trace_long,
)
from .base import BaseAligner


class EventAligner(BaseAligner):
    def align(self, df_traces: pd.DataFrame, event_starts: np.ndarray) -> pd.DataFrame:
        if self.copy:
            df_traces = df_traces.copy()

        df_traces = align_to_events(
            df_wide=df_traces,
            events=event_starts,
            t_before=self.t_before,
            t_after=self.t_after,
            time_col=self.time_col,
            created_event_index_col=self.created_event_index_col,
            created_aligned_time_col=self.created_aligned_time_col,
            round_precision=self.round_precision,
            drop_non_aligned=self.drop_non_aligned,
        )
        return df_traces

    def align_long(
        self, df_traces: pd.DataFrame, event_starts: np.ndarray
    ) -> pd.DataFrame:
        if self.copy:
            df_traces = df_traces.copy()

        df_traces = align_to_events_long(
            df_wide=df_traces,
            events=event_starts,
            t_before=self.t_before,
            t_after=self.t_after,
            time_col=self.time_col,
            created_event_index_col=self.created_event_index_col,
            created_aligned_time_col=self.created_aligned_time_col,
            round_precision=self.round_precision,
            drop_non_aligned=self.drop_non_aligned,
        )
        return df_traces

    def average_trace(self, df_traces: pd.DataFrame, event_starts: np.ndarray):
        if self.copy:
            df_traces = df_traces.copy()

        df_average_trace = average_trace(
            df_wide=df_traces,
            events=event_starts,
            t_before=self.t_before,
            t_after=self.t_after,
            time_col=self.time_col,
            created_aligned_time_col=self.created_aligned_time_col,
            round_precision=self.round_precision,
            agg_func=self.average_trace_agg_func,
        )
        return df_average_trace

    def average_trace_long(self, df_traces: pd.DataFrame, event_starts: np.ndarray):
        if self.copy:
            df_traces = df_traces.copy()

        df_average_trace = average_trace_long(
            df_wide=df_traces,
            events=event_starts,
            t_before=self.t_before,
            t_after=self.t_after,
            time_col=self.time_col,
            created_aligned_time_col=self.created_aligned_time_col,
            created_neuron_col=self.created_neuron_col,
            created_value_col=self.created_value_col,
            round_precision=self.round_precision,
            agg_func=self.average_trace_agg_func,
        )
        return df_average_trace
