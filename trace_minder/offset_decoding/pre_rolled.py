import pandas as pd
import numpy as np
from typing import Optional
from tqdm import tqdm
from sklearn.model_selection import KFold
from sklearn.base import BaseEstimator
from sklearn.model_selection._split import BaseCrossValidator
from joblib import delayed, Parallel
from calcium_clear.surrogates import rotate_traces
from .preprocessors import BlockRotatePreprocessor, TwoSessionBlockRotatePreprocessor
from .offset_dispatchers import NestedRandomizedDispatcher


def block_decode(
    df_traces: pd.DataFrame,
    events: np.ndarray,
    decoder: BaseEstimator,
    t_before: float = 5,
    t_after: float = 5,
    num_runs: int = 75,
    cross_validator=None,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
    reroll_both: bool = False,
    verbose: bool = False,
):
    """
    Functional interface for block decoding.

    Parameters
    ----------
    df_traces : pd.DataFrame
        A dataframe containing calcium traces.
    events : np.ndarray
        An array of event times. floats
    decoder : BaseEstimator
        A scikit-learn estimator that can be fit and predict.
    t_before : float, optional
        Time before event to include in the trace, by default 5
    t_after : float, optional
        Time after event to include in the trace, by default 5
    num_runs : int, optional
        Number of times to reroll the preprocessor, by default 75.
    cross_validator : Optional[BaseEstimator], optional
        A scikit-learn cross validator, by default None (uses KFold with 5 folds).
    scoring : str, optional
        A scikit-learn scoring metric, by default "f1_macro".
    n_jobs : int, optional
        Number of jobs to run in parallel, by default -1.
    reroll_both : bool, optional
        Whether to reroll both the observed and surrogate traces afer each run, by default False.
    """
    if cross_validator is None:
        cross_validator = KFold(n_splits=5, shuffle=True, random_state=42)

    preprocessor = BlockRotatePreprocessor(
        df_traces=df_traces.copy(),
        block_starts=events,
        t_before=t_before,
        t_after=t_after,
        reroll_both=reroll_both,
    )

    dispatcher = NestedRandomizedDispatcher(
        offset_decoder=decoder,
        trace_preprocessor=preprocessor,
        cross_validator=cross_validator,
        scoring=scoring,
        num_runs=num_runs,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    df_results = dispatcher.run()
    return df_results


def block_decode_two_sessions(
    session1_df_traces: pd.DataFrame,
    session1_block_starts: np.ndarray,
    session2_df_traces: pd.DataFrame,
    session2_block_starts: np.ndarray,
    decoder: BaseEstimator,
    t_before: float = 5,
    t_after: float = 5,
    num_runs: int = 75,
    cross_validator=None,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
    reroll_both: bool = False,
    verbose: bool = False,
):
    if cross_validator is None:
        cross_validator = KFold(n_splits=5, shuffle=True, random_state=42)

    preprocessor = TwoSessionBlockRotatePreprocessor(
        session1_df_traces=session1_df_traces.copy(),
        session1_block_starts=session1_block_starts,
        session2_df_traces=session2_df_traces.copy(),
        session2_block_starts=session2_block_starts,
        t_before=t_before,
        t_after=t_after,
        reroll_both=reroll_both,
    )

    dispatcher = NestedRandomizedDispatcher(
        offset_decoder=decoder,
        trace_preprocessor=preprocessor,
        cross_validator=cross_validator,
        scoring=scoring,
        num_runs=num_runs,
        verbose=verbose,
        n_jobs=n_jobs,
    )
    df_results = dispatcher.run()
    return df_results


def block_decode_with_shuffle_double_reroll(
    df_traces: pd.DataFrame,
    events: np.ndarray,
    decoder: BaseEstimator,
    time_col: str = "time",
    rotation_increment: Optional[int] = None,
    t_before: float = 5,
    t_after: float = 5,
    num_runs: int = 75,
    cross_validator: Optional[BaseCrossValidator] = None,
    scoring: str = "f1_macro",
    n_jobs: int = -1,
    verbose: bool = False,
):
    """
    Functional interface for block decoding and comparison with shuffle v shuffle decoding

    Parameters
    ----------
    df_traces : pd.DataFrame
        A dataframe containing calcium traces.
    events : np.ndarray
        An array of event times. floats
    decoder : BaseEstimator
        A scikit-learn estimator that can be fit and predict.
    time_col : str, optional
        The name of the time column in df_traces, by default "time".
    rotation_increment : Optional[int], optional
        The number of timepoints to rotate the traces by, by default None.
    t_before : float, optional
        Time before event to include in the trace, by default 5
    t_after : float, optional
        Time after event to include in the trace, by default 5
    num_runs : int, optional
        Number of times to reroll the preprocessor, by default 75.
    cross_validator : Optional[BaseEstimator], optional
        A scikit-learn cross validator, by default None (uses KFold with 5 folds).
    scoring : str, optional
        A scikit-learn scoring metric, by default "f1_macro".
    n_jobs : int, optional
        Number of jobs to run in parallel, by default -1.
    verbose : bool, optional
        Whether to print progress, by default False.
    """
    df_res_obs = block_decode(
        df_traces=df_traces.copy(),
        events=events,
        decoder=decoder,
        t_before=t_before,
        t_after=t_after,
        num_runs=num_runs,
        cross_validator=cross_validator,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
    )
    df_res_obs["condition"] = "observed"
    df_res_shuffle = block_decode(
        df_traces=rotate_traces(
            df_traces.copy(), increment=rotation_increment, time_col=time_col
        ),
        events=events,
        decoder=decoder,
        t_before=t_before,
        t_after=t_after,
        num_runs=num_runs,
        cross_validator=cross_validator,
        scoring=scoring,
        n_jobs=n_jobs,
        verbose=verbose,
        reroll_both=True,
    )
    df_res_shuffle["condition"] = "shuffled"

    df_res = pd.concat([df_res_obs, df_res_shuffle])
    return df_res


class EventOffsetDecoderRobust:
    """
    Decode observed/rotated traces at offsets relative to a single set of events.

    Overall method:
        - Decode observed traces at offsets relative to events NUM_INNER times
        - For each NUM_OUTER iteration:
            - Rotate the "observed" traces
            - Decode "observed" traces at offsets relative to events NUM_INNER times


    Methods
    -------
    run
        Run the decoding procedure.

    """

    def __init__(
        self,
        decoder: BaseEstimator,
        time_col: str = "time",
        t_before: float = 5,
        t_after: float = 5,
        num_runs_inner: int = 75,
        num_runs_outer: int = 75,
        cross_validator: Optional[BaseCrossValidator] = None,
        rotation_increment: Optional[int] = None,
        scoring: str = "f1_macro",
        n_jobs_outer: int = -1,
        n_jobs_inner: int = 1,
        verbose_inner: bool = False,
        verbose_outer: bool = False,
    ):
        self.decoder = decoder
        self.time_col = time_col
        self.rotation_increment = rotation_increment
        self.t_before = t_before
        self.t_after = t_after
        self.num_runs_inner = num_runs_inner
        self.num_runs_outer = num_runs_outer
        self.scoring = scoring
        self.n_jobs_outer = n_jobs_outer
        self.n_jobs_inner = n_jobs_inner
        self.verbose_inner = verbose_inner
        self.verbose_outer = verbose_outer

        if cross_validator is None:
            self.cross_validator = KFold(n_splits=5, shuffle=True, random_state=42)
        else:
            self.cross_validator = cross_validator

    def _single_iteration(self, run_index, df_traces, events):
        """
        A single iteration of the outer loop.

        Parameters
        ----------
        run_index : int
            The index of the current iteration.
        df_traces : pd.DataFrame
            The traces to be decoded.
        events : np.ndarray
            The events to be used as offsets.

        Returns
        -------
        df_res_shuffle : pd.DataFrame
            The results of the shuffled decoding. Columns are:
                - "condition" : "observed" or "shuffled"
                - "outer_run" : the index of the outer loop iteration
                - "inner_run" : the index of the inner loop iteration
                - "score" : the score of the decoding
                - "offset" : the offset used for the decoding
        """
        df_res_shuffle = self._block_decode(
            df_traces=self._rotate_traces(df_traces.copy()),
            events=events,
            n_jobs=self.n_jobs_inner,
        )
        df_res_shuffle["outer_run"] = run_index
        return df_res_shuffle

    def _rotate_traces(self, df_traces):
        return rotate_traces(
            df_traces, increment=self.rotation_increment, time_col=self.time_col
        )

    def _block_decode(self, df_traces, events, n_jobs=1):
        """
        A single iteration of the inner loop.

        Parameters
        ----------
        df_traces : pd.DataFrame
            A dataframe of traces to decode.
        events : np.ndarray
            The event times to decode relative to.
        n_jobs : int
            The number of jobs to run in parallel.

        Returns
        -------
        df_results : pd.DataFrame
            The results of the decoding. Columns are:
                - "condition" : "observed" or "shuffled"
                - "outer_run" : the index of the outer loop iteration
                - "inner_run" : the index of the inner loop iteration
                - "score" : the score of the decoding
                - "offset" : the offset used for the decoding
        """
        preprocessor = BlockRotatePreprocessor(
            df_traces=df_traces.copy(),
            block_starts=events,
            t_before=self.t_before,
            t_after=self.t_after,
            reroll_both=False,
        )

        dispatcher = NestedRandomizedDispatcher(
            offset_decoder=self.decoder,
            trace_preprocessor=preprocessor,
            cross_validator=self.cross_validator,
            scoring=self.scoring,
            num_runs=self.num_runs_inner,
            verbose=self.verbose_inner,
            n_jobs=n_jobs,
        )
        df_results = dispatcher.run()
        return df_results

    def run(self, df_traces: pd.DataFrame, events: np.ndarray):
        """
        Run the decoding procedure.

        Parameters
        ----------
        df_traces : pd.DataFrame
            A dataframe of traces to decode.
        events : np.ndarray
            The event times to decode relative to.

        Returns
        -------
        df_res : pd.DataFrame
            The results of the decoding. Columns are:
                - "condition" : "observed" or "shuffled"
                - "outer_run" : the index of the outer loop iteration
                - "inner_run" : the index of the inner loop iteration
                - "score" : the score of the decoding
                - "offset" : the offset used for the decoding
        """
        if self.verbose_outer:
            print("Running observed")
            outer_iterater = tqdm(range(self.num_runs_outer))
        else:
            outer_iterater = range(self.num_runs_outer)

        df_res_obs = self._block_decode(df_traces=df_traces.copy(), events=events)
        df_res_obs["condition"] = "observed"
        df_res_obs["outer_run"] = np.nan

        if self.verbose_outer:
            print("Running Shuffle")

        outter_shuffle_dfs = Parallel(n_jobs=self.n_jobs_outer)(
            delayed(self._single_iteration)(run_index, df_traces, events)
            for run_index in outer_iterater
        )

        outter_shuffle_df = pd.concat(outter_shuffle_dfs)
        outter_shuffle_df["condition"] = "shuffled"

        df_res = pd.concat([df_res_obs, outter_shuffle_df])
        return df_res
