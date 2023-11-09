import pandas as pd
import numpy as np
from typing import Optional
from trace_minder.surrogates import SurrogateGenerator, Rotater
from sklearn.preprocessing import LabelEncoder
from .event_preprocess import EventTimeseriesMaker


class AllTimeDecodePreprocesser:
    def get_X_y(self, rotate: bool = False, encode_y: bool = True):
        raise NotImplementedError


class EventPreprocesser(AllTimeDecodePreprocesser):
    ...


class EventPreprocesserSingleSession(EventPreprocesser):
    def __init__(
        self,
        df_traces: pd.DataFrame,
        event_starts: np.ndarray,
        event_timeseries_maker: EventTimeseriesMaker,
        time_col: str = "time",
        shuffler: Optional[SurrogateGenerator] = None,
    ):
        self.df_traces = df_traces
        self.event_starts = np.asarray(event_starts)
        self.event_timeseries_maker = event_timeseries_maker
        self.time_col = time_col
        self.shuffler = (
            Rotater(time_col=self.time_col, copy=True) if shuffler is None else shuffler
        )
        self.le = LabelEncoder()

    def _get_y(self, encode: bool = True):
        y = self.event_timeseries_maker(
            time_arr=self.df_traces[self.time_col].values,
            event_starts=self.event_starts,
        )

        if encode:
            y = self.le.fit_transform(y)
        return y

    def _get_X(self, rotate: bool = False):
        df = self.df_traces.copy() if not rotate else self.shuffler(self.df_traces)
        return df.values

    def get_X_y(self, rotate: bool = False, encode_y: bool = True):
        X = self._get_X(rotate=rotate)
        y = self._get_y(encode=encode_y)
        return X, y


class EventPreprocesserTwoSession(EventPreprocesser):
    # sort out coregistration
    def __init__(
        self,
        session1_preprocesser: EventPreprocesserSingleSession,
        session2_preprocesser: EventPreprocesserSingleSession,
    ):
        self.session1_preprocesser = session1_preprocesser
        self.session2_preprocesser = session2_preprocesser
        self._coregister()
        self.le = LabelEncoder()

    def _coregister(self):
        # Extract columns excluding time columns
        neurons1 = set(
            c
            for c in self.session1_preprocesser.df_traces.columns
            if c != self.session1_preprocesser.time_col
        )
        neurons2 = set(
            c
            for c in self.session2_preprocesser.df_traces.columns
            if c != self.session2_preprocesser.time_col
        )

        # Find common neurons between the two sessions
        common_neurons = sorted(neurons1.intersection(neurons2))

        # Update dataframes to keep only the common neurons and time column in the same order
        self.session1_preprocesser.df_traces = self.session1_preprocesser.df_traces[
            common_neurons + [self.session1_preprocesser.time_col]
        ]

        self.session2_preprocesser.df_traces = self.session2_preprocesser.df_traces[
            common_neurons + [self.session2_preprocesser.time_col]
        ]

    def get_X_y(self, rotate: bool = False, encode_y: bool = True):
        X1, y1 = self.session1_preprocesser.get_X_y(rotate=rotate, encode_y=False)
        X2, y2 = self.session2_preprocesser.get_X_y(rotate=rotate, encode_y=False)

        X = np.concatenate([X1, X2], axis=0)
        y = np.concatenate([y1, y2], axis=0)

        if encode_y:
            y = self.le.fit_transform(y)

        return X, y
