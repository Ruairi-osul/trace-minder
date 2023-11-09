import numpy as np
import pandas as pd
from trace_minder.surrogates import SurrogateGenerator, Rotater
from trace_minder.align.event_aligner import EventAligner

# from trace_minder.trace_stats import get_trace_stat, BaseStat
from trace_minder.transforms import Melter

from typing import Optional, Union, Callable
from joblib import Parallel, delayed

from trace_minder.align.base import BaseAligner
from trace_minder.trace_aggregation import PrePostAggregator
from trace_minder.preprocess import TracePreprocessor
from typing import Union, Any
from calcium_clear.stats import p_adjust


class RotatedAverageTraceResponders:
    def __init__(
        self,
        aligner: BaseAligner,
        aggregator: PrePostAggregator,
        rotator: Optional[SurrogateGenerator] = None,
        average_trace_preprocessor: Optional[TracePreprocessor] = None,
        n_boot: int = 200,
        adjust_pval: bool = False,
        sided: str = "two_sided",
        created_pval_col: str = "pval",
        created_stat_col: str = "stat",
        created_sig_col: str = "sig",
        n_jobs: int = -1,
    ):
        self.aligner = aligner
        self.aggregator = aggregator
        self.average_trace_preprocessor = average_trace_preprocessor
        self.pre_col = self.aggregator.pre_indicator
        self.post_col = self.aggregator.post_indicator
        self.adjust_pval = adjust_pval
        self.created_pval_col = created_pval_col
        self.created_stat_col = created_stat_col
        self.created_sig_col = created_sig_col

        self.time_col_raw = self.aligner.time_col
        self.neuron_col = self.aggregator.neuron_col
        self.stat_col = self.aggregator.created_diff_col
        self.aggregator.event_idx_col = None

        self.n_boot = n_boot
        self.n_jobs = n_jobs

        if rotator is not None:
            if not isinstance(rotator, SurrogateGenerator):
                raise TypeError(
                    f"rotator must be a subclass of SurrogateGenerator, not {type(rotator)}"
                )
            self.rotator = rotator
        else:
            self.rotator = Rotater(time_col=self.time_col_raw, copy=True)

        assert sided in [
            "two_sided",
            "lower",
            "higher",
        ], "sided must be one of 'two_sided', 'lower', or 'higher'"
        self.sided = sided

    def _single_run(
        self, df: pd.DataFrame, events: Any, rotate: bool = False
    ) -> pd.DataFrame:
        df = df.copy()
        if rotate:
            df = self.rotator(df)
        df = self.aligner.average_trace_long(df, events)
        if self.average_trace_preprocessor is not None:
            df = self.average_trace_preprocessor(df)

        df = self.aggregator.prepost_diff(df)
        df = df.rename(
            columns={self.aggregator.created_diff_col: self.created_stat_col}
        )
        df = df.loc[:, [self.neuron_col, self.created_stat_col]]
        return df

    def _get_reps(self, df: pd.DataFrame, events: Any) -> pd.DataFrame:
        frames = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_run)(df, events=events, rotate=True)
            for _ in range(self.n_boot)
        )
        df_res = pd.concat(frames, axis=0)
        return df_res

    def _get_obs(self, df: pd.DataFrame, events: Any) -> pd.DataFrame:
        return self._single_run(df, events=events, rotate=False)

    def _get_pvals(self, df_obs: pd.DataFrame, df_reps: pd.DataFrame):
        pvals = []
        for _, row in df_obs.iterrows():
            cell_id = row[self.neuron_col]
            obs_stat = row[self.created_stat_col]

            reps_stat = df_reps[df_reps[self.neuron_col] == cell_id][
                self.created_stat_col
            ].values

            if self.sided == "two_sided":
                pval = np.mean(np.abs(reps_stat) >= np.abs(obs_stat)) * 2
            elif self.sided == "lower":
                pval = np.mean(reps_stat <= obs_stat)
            elif self.sided == "higher":
                pval = np.mean(reps_stat >= obs_stat)
            pvals.append((cell_id, pval))

        df_pval = pd.DataFrame(pvals, columns=[self.neuron_col, self.created_pval_col])
        return df_pval

    def _adjust_pvals(self, df_res: pd.DataFrame) -> pd.DataFrame:
        df_res[self.created_pval_col] = p_adjust(df_res[self.created_pval_col])
        return df_res

    def get_responders(self, df_traces: pd.DataFrame, events: Any) -> pd.DataFrame:
        df_obs = self._get_obs(df_traces, events)
        df_reps = self._get_reps(df_traces, events)
        df_pval = self._get_pvals(df_obs, df_reps)
        df_res = pd.merge(df_obs, df_pval, on=self.neuron_col)
        if self.adjust_pval:
            df_res = self._adjust_pvals(df_res)
        df_res[self.created_sig_col] = df_res[self.created_pval_col] < 0.05

        return df_res


# class RotatedAverageTraceResponders2:
#     def __init__(
#         self,
#         df_traces: pd.DataFrame,
#         time_col: str,
#         events: np.ndarray,
#         aligner: EventAligner,
#         shuffler: Optional[SurrogateGenerator] = None,
#         stat: Union[str, Callable[[np.ndarray, np.ndarray], float], BaseStat] = "auc",
#         n_boot: int = 200,
#         sided: str = "two_sided",
#         created_cell_col: str = "cell_id",
#         n_jobs: int = -1,
#     ):
#         self.df_traces = df_traces
#         self.events = events
#         self.time_col = time_col
#         if shuffler is not None:
#             if not isinstance(shuffler, SurrogateGenerator):
#                 raise TypeError(
#                     f"shuffler must be a subclass of SurrogateGenerator, not {type(shuffler)}"
#                 )
#             self.shuffler = shuffler
#         else:
#             self.shuffler = Rotater(time_col=self.time_col, copy=True)
#         self.shuffler

#         self.aligner = aligner
#         self.n_boot = n_boot
#         self.n_jobs = n_jobs

#         assert sided in [
#             "two_sided",
#             "lower",
#             "higher",
#         ], "sided must be one of 'two_sided', 'lower', or 'higher'"
#         self.sided = sided
#         self.created_cell_col = created_cell_col

#         self.melter = Melter(
#             id_vars=[aligner.created_aligned_time_col],
#             created_cell_col="cell_id",
#             created_value_col="value",
#         )

#         self.stat = get_trace_stat(stat) if isinstance(stat, str) else stat

#     def _apply_stat_single_cell(
#         self,
#         df: pd.DataFrame,
#         stat: Callable[[np.ndarray, np.ndarray], float],
#         value_col: str,
#         time_col: str,
#     ) -> float:
#         time = df[time_col].values
#         values = df[value_col].values

#         return stat(time, values)

#     def _single_run(self, rotate: bool = False):
#         df = self.df_traces.copy()
#         # rotate if nessessary
#         if rotate:
#             df = self.shuffler(df)

#         # align
#         df = self.aligner.average_trace(df_traces=df, event_starts=self.events)
#         # # melt
#         df = self.melter(df)

#         # compute stat
#         ser_res = df.groupby(self.melter.created_cell_col).apply(
#             self._apply_stat_single_cell,
#             stat=self.stat,
#             value_col=self.melter.created_value_col,
#             time_col=self.aligner.created_aligned_time_col,
#         )
#         df_res = ser_res.to_frame(name="stat")
#         df_res.reset_index(inplace=True)
#         df_res.rename(columns={"level_1": self.created_cell_col}, inplace=True)
#         return df_res

#     def _get_reps(self):
#         # Using joblib's Parallel and delayed to parallelize the generation of reps
#         frames = Parallel(n_jobs=self.n_jobs)(
#             delayed(self._single_run)(rotate=True) for _ in range(self.n_boot)
#         )
#         df_res = pd.concat(frames, axis=0)
#         return df_res

#     def _get_obs(self):
#         return self._single_run(rotate=False)

#     def _get_pval(self, df_obs: pd.DataFrame, df_reps: pd.DataFrame):
#         pvals = []
#         for _, row in df_obs.iterrows():
#             cell_id = row[self.created_cell_col]
#             obs_stat = row["stat"]

#             reps_stat = df_reps[df_reps[self.created_cell_col] == cell_id][
#                 "stat"
#             ].values

#             if self.sided == "two_sided":
#                 pval = np.mean(np.abs(reps_stat) >= np.abs(obs_stat)) * 2
#             elif self.sided == "lower":
#                 pval = np.mean(reps_stat <= obs_stat)
#             elif self.sided == "higher":
#                 pval = np.mean(reps_stat >= obs_stat)

#             pvals.append((cell_id, pval))

#         df_pval = pd.DataFrame(pvals, columns=[self.created_cell_col, "pval"])
#         return df_pval

#     def __call__(self):
#         df_obs = self._get_obs()
#         df_reps = self._get_reps()
#         df_pval = self._get_pval(df_obs, df_reps)
#         df_res = pd.merge(df_obs, df_pval, on=self.created_cell_col)
#         return df_res
