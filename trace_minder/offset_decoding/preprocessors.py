import pandas as pd
import numpy as np
from typing import Optional, Dict
from calcium_clear.align.align_events import align_to_events
from calcium_clear.surrogates import rotate_traces


class OffsetPreprocesser:
    """
    A base class for preprocessers that yeild X and y data at a set of offsets.

    Methods
    -------
    stack_dfs(dfs)
        Stacks a dictionary of dataframes into a single dataframe.
    get_X_y_at_offset(offset)
        Returns X and y data at a given offset.

    Attributes
    ----------
    offsets : np.ndarray
        The offsets that the preprocesser will yeild data at.

    """

    @property
    def offsets(
        self,
    ):
        ...

    def get_X_y_at_offset(
        self,
        offset: float,
    ):
        ...

    @staticmethod
    def stack_dfs(dfs: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        return pd.concat([df.assign(group=group) for group, df in dfs.items()])


class RandomizedPreprocessor(OffsetPreprocesser):
    """
    A base class for preprocessers that yeild X and y data at a set of offsets where a
    random process is involves in data generation.

    Methods
    -------
    reroll()
        Rerolls the random process used to generate data. Returns a new instance.

    Parameters
    ----------
    rotation_increment : Optional[int], optional
        The amount to rotate the data by, by default None.

    """

    def __init__(
        self,
        time_col: str,
        rotation_increment: Optional[int] = None,
    ):
        self.time_col = time_col
        self.rotation_increment = rotation_increment

    def reroll(self, rotation_increment: Optional[int] = None):
        ...


class BlockPreprocessor(RandomizedPreprocessor):
    def __init__(
        self,
        created_aligned_time_col: str,
        created_event_index_col: str,
        time_col: str,
    ):
        self.created_aligned_time_col = created_aligned_time_col
        self.created_event_index_col = created_event_index_col
        self.time_col = time_col

        self.df = self._make_df()

    def _make_df(self) -> pd.DataFrame:
        ...

    @property
    def offsets(self):
        return self.df[self.created_aligned_time_col].unique()

    def get_X_y_at_offset(self, offset: float):
        df_at_offset = self.df.loc[self.df[self.created_aligned_time_col] == offset, :]
        X = df_at_offset.drop(
            columns=[
                self.created_aligned_time_col,
                self.created_event_index_col,
                "group",
                self.time_col,
            ]
        ).values
        y = df_at_offset["group"].values
        return X, y


class BlockRotatePreprocessor(BlockPreprocessor):
    """
    A preprocesser that yeilds X and y data at a set of offsets where observed and rotated
    data are aligned to block starts.

    Parameters
    ----------
    df_traces : pd.DataFrame
        A dataframe of traces.
    block_starts : np.ndarray
        The start times of blocks (floats).
    t_before : float
        The amount of time before the block start to include in the data.
    t_after : float
        The amount of time after the block start to include in the data.
    time_col : str, optional
        The name of the time column in df_traces, by default "time".
    round_precision : int, optional
        The number of decimal places to round the time column to, by default 2.
    created_aligned_time_col : str, optional
        The name of the column in the returned dataframe that contains the aligned
        time, by default "aligned_time".
    created_event_index_col : str, optional
        The name of the column in the returned dataframe that contains the index of the
        event, by default "event_idx".
    max_trials : Optional[int], optional
        The maximum number of trials to include in the data, by default None. If none, all.
    rotation_increment : Optional[int], optional
        The amount to rotate the data by, by default None.
    reroll_both : bool, optional
        Whether to reroll both the observed and rotated data, by default False.

    Methods
    -------
    get_X_y_at_offset(offset)
        Returns X and y data at a given offset.
    reroll(rotation_increment)
        Rerolls the random process used to generate data. Returns a new instance.
    """

    def __init__(
        self,
        df_traces: pd.DataFrame,
        block_starts: np.ndarray,
        t_before: float,
        t_after: float,
        time_col: str = "time",
        round_precision: int = 2,
        created_aligned_time_col: str = "aligned_time",
        created_event_index_col: str = "event_idx",
        max_trials: Optional[int] = None,
        rotation_increment: Optional[int] = None,
        reroll_both: bool = False,
    ):
        self.block_starts = block_starts
        self.t_before = t_before
        self.t_after = t_after
        self.time_col = time_col
        self.round_precision = round_precision
        self.created_aligned_time_col = created_aligned_time_col
        self.created_event_index_col = created_event_index_col
        self.max_trials = max_trials
        self.rotation_increment = rotation_increment
        self.reroll_both = reroll_both

        self.df_traces = df_traces
        self.df = self._make_df()

    def _make_df(
        self,
    ) -> pd.DataFrame:
        df_traces_rotated = rotate_traces(
            self.df_traces.copy(),
            increment=self.rotation_increment,
            time_col=self.time_col,
        )
        df_observed = self._align_traces_to_block_starts(self.df_traces)
        df_rotated = self._align_traces_to_block_starts(df_traces=df_traces_rotated)
        df = self.stack_dfs(dict(observed=df_observed, rotated=df_rotated))
        return df

    def _align_traces_to_block_starts(self, df_traces: pd.DataFrame) -> pd.DataFrame:
        df_aligned = align_to_events(
            df_wide=df_traces.copy(),
            events=self.block_starts,
            t_before=self.t_before,
            t_after=self.t_after,
            time_col=self.time_col,
            round_precision=self.round_precision,
        )
        return df_aligned

    def reroll(self, rotation_increment: Optional[int] = None):
        """
        Create a new instance of the class with rerolled randomized data,
        while keeping all other parameters the same.
        """
        if self.reroll_both:
            df_traces = rotate_traces(
                self.df_traces.copy(),
                increment=rotation_increment,
                time_col=self.time_col,
            )
        else:
            df_traces = self.df_traces

        return type(self)(
            df_traces=df_traces,
            block_starts=self.block_starts,
            t_before=self.t_before,
            t_after=self.t_after,
            time_col=self.time_col,
            round_precision=self.round_precision,
            created_aligned_time_col=self.created_aligned_time_col,
            created_event_index_col=self.created_event_index_col,
            max_trials=self.max_trials,
            rotation_increment=rotation_increment or self.rotation_increment,
            reroll_both=self.reroll_both,
        )


class TwoSessionBlockRotatePreprocessor(BlockPreprocessor):
    """ """

    def __init__(
        self,
        session1_df_traces: pd.DataFrame,
        session1_block_starts: np.ndarray,
        session2_df_traces: pd.DataFrame,
        session2_block_starts: np.ndarray,
        t_before: float,
        t_after: float,
        session1_name: str = "session1",
        session2_name: str = "session2",
        time_col: str = "time",
        round_precision: int = 2,
        created_aligned_time_col: str = "aligned_time",
        created_event_index_col: str = "event_idx",
        max_trials: Optional[int] = None,
        rotation_increment: Optional[int] = None,
        reroll_both: bool = False,
    ):
        self.session1_df_traces = session1_df_traces
        self.session2_df_traces = session2_df_traces
        self.session1_block_starts = session1_block_starts
        self.session2_block_starts = session2_block_starts
        self.t_before = t_before
        self.t_after = t_after
        self.time_col = time_col
        self.round_precision = round_precision
        self.created_aligned_time_col = created_aligned_time_col
        self.created_event_index_col = created_event_index_col
        self.max_trials = max_trials
        self.rotation_increment = rotation_increment
        self.reroll_both = reroll_both
        self.session1_name = session1_name
        self.session2_name = session2_name

        self.session1_preprocessor = self._make_session_preprocessor(
            self.session1_df_traces, self.session1_block_starts
        )
        self.session2_preprocessor = self._make_session_preprocessor(
            self.session2_df_traces, self.session2_block_starts
        )

        self.df = self._make_df()
        self.df = self.coregister(self.df)

    def _make_df(self):
        return pd.concat(
            [
                df.assign(group=lambda x: x.group.map(lambda x: f"{group_name}_{x}"))
                for df, group_name in zip(
                    [self.session1_preprocessor.df, self.session2_preprocessor.df],
                    [self.session1_name, self.session2_name],
                )
            ]
        )

    def _make_session_preprocessor(self, df_traces, events):
        return BlockRotatePreprocessor(
            df_traces=df_traces,
            block_starts=events,
            t_before=self.t_before,
            t_after=self.t_after,
            time_col=self.time_col,
            round_precision=self.round_precision,
            created_aligned_time_col=self.created_aligned_time_col,
            created_event_index_col=self.created_event_index_col,
            max_trials=self.max_trials,
            rotation_increment=self.rotation_increment,
            reroll_both=self.reroll_both,
        )

    def coregister(self, df: pd.DataFrame) -> pd.DataFrame:
        return df.dropna(axis=1)

    def reroll(self):
        self.session1_preprocessor = self.session1_preprocessor.reroll()
        self.session2_preprocessor = self.session2_preprocessor.reroll()
        self.df = self._make_df()
        self.df = self.coregister(self.df)
        return self
