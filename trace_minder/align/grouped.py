import pandas as pd
import numpy as np
from calcium_clear.align.align_events import (
    align_to_events_grouped,
    align_to_events_grouped_long,
)
from calcium_clear.align.average_trace import (
    average_trace_grouped,
    average_trace_grouped_long,
)
from .base import BaseAligner
from typing import Dict, List, Any


class GroupedAligner(BaseAligner):
    def __init__(
        self,
        t_before: float,
        t_after: float,
        df_wide_group_mapper: Dict[str, List[Any]],
        round_precision: int = 1,
        n_jobs: int = -1,
        time_col: str = "time",
        df_events_event_time_col: str = "event_time",
        df_events_group_col: str = "group",
        created_event_index_col: str = "event_idx",
        created_aligned_time_col: str = "aligned_time",
        created_neuron_col: str = "neuron",
        created_value_col: str = "value",
        drop_non_aligned: bool = True,
        copy: bool = True,
    ):
        self.t_before = t_before
        self.t_after = t_after
        self.df_wide_group_mapper = df_wide_group_mapper
        self.round_precision = round_precision
        self.n_jobs = n_jobs
        self.time_col = time_col
        self.df_events_event_time_col = df_events_event_time_col
        self.df_events_group_col = df_events_group_col
        self.created_event_index_col = created_event_index_col
        self.created_aligned_time_col = created_aligned_time_col
        self.created_neuron_col = created_neuron_col
        self.created_value_col = created_value_col
        self.drop_non_aligned = drop_non_aligned
        self.copy = copy

    def align(
        self,
        df_traces: pd.DataFrame,
        event_starts: np.ndarray,
    ) -> pd.DataFrame:
        if self.copy:
            df_traces = df_traces.copy()

        df_traces = align_to_events_grouped(
            df_wide=df_traces,
            df_events=event_starts,
            t_before=self.t_before,
            t_after=self.t_after,
            df_wide_group_mapper=self.df_wide_group_mapper,
            round_precision=self.round_precision,
            n_jobs=self.n_jobs,
            df_wide_time_col=self.time_col,
            df_events_event_time_col=self.df_events_event_time_col,
            df_events_group_col=self.df_events_group_col,
            created_event_index_col=self.created_event_index_col,
            created_aligned_time_col=self.created_aligned_time_col,
            created_neuron_col=self.created_neuron_col,
            created_value_col=self.created_value_col,
            drop_non_aligned=self.drop_non_aligned,
        )
        return df_traces

    def align_long(
        self,
        df_traces: pd.DataFrame,
        event_starts: pd.DataFrame,
    ) -> pd.DataFrame:
        if self.copy:
            df_traces = df_traces.copy()

        df_traces = align_to_events_grouped_long(
            df_wide=df_traces,
            df_events=event_starts,
            t_before=self.t_before,
            t_after=self.t_after,
            df_wide_group_mapper=self.df_wide_group_mapper,
            round_precision=self.round_precision,
            n_jobs=self.n_jobs,
            df_wide_time_col=self.time_col,
            df_events_event_time_col=self.df_events_event_time_col,
            df_events_group_col=self.df_events_group_col,
            created_event_index_col=self.created_event_index_col,
            created_aligned_time_col=self.created_aligned_time_col,
            created_neuron_col=self.created_neuron_col,
            created_value_col=self.created_value_col,
            drop_non_aligned=self.drop_non_aligned,
        )
        return df_traces

    def average_trace(
        self,
        df_traces: pd.DataFrame,
        event_starts: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates an average trace


        Columns of returned df cell_ids + created_aligned_time_col

        """
        if self.copy:
            df_traces = df_traces.copy()

        df_average_trace = average_trace_grouped(
            df_wide=df_traces,
            df_events=event_starts,
            t_before=self.t_before,
            t_after=self.t_after,
            df_wide_group_mapper=self.df_wide_group_mapper,
            round_precision=self.round_precision,
            n_jobs=self.n_jobs,
            df_wide_time_col=self.time_col,
            df_events_event_time_col=self.df_events_event_time_col,
            df_events_group_col=self.df_events_group_col,
            created_aligned_time_col=self.created_aligned_time_col,
        )
        # df_average_trace = df_average_trace.reset_index()
        return df_average_trace

    def average_trace_long(
        self,
        df_traces: pd.DataFrame,
        event_starts: pd.DataFrame,
    ) -> pd.DataFrame:
        """
        Creates an average trace


        Columns of returned df cell_ids + created_aligned_time_col

        """
        if self.copy:
            df_traces = df_traces.copy()

        df_average_trace = average_trace_grouped_long(
            df_wide=df_traces,
            df_events=event_starts,
            t_before=self.t_before,
            t_after=self.t_after,
            df_wide_group_mapper=self.df_wide_group_mapper,
            round_precision=self.round_precision,
            n_jobs=self.n_jobs,
            df_wide_time_col=self.time_col,
            df_events_event_time_col=self.df_events_event_time_col,
            df_events_group_col=self.df_events_group_col,
            created_aligned_time_col=self.created_aligned_time_col,
            created_neuron_col=self.created_neuron_col,
            created_value_col=self.created_value_col,
        )
        # df_average_trace = df_average_trace.reset_index()
        return df_average_trace
