import numpy as np
import pandas as pd
from trace_minder.surrogates import SurrogateGenerator, Rotater

from typing import Optional
from joblib import Parallel, delayed

from trace_minder.align.base import BaseAligner
from trace_minder.trace_aggregation import PrePostAggregator
from trace_minder.preprocess import TracePreprocessor
from typing import Any
from calcium_clear.stats import p_adjust


class AUCDiff:
    def __init__(
        self,
        aligner: BaseAligner,
        average_trace_preprocessor: Optional[TracePreprocessor],
        aggregator: PrePostAggregator,
        created_stat_col: str = "auc_diff",
        created_neuron_col: str = "cell_id",
    ):
        self.aligner = aligner
        self.average_trace_preprocessor = average_trace_preprocessor
        self.aggregator = aggregator

        self.pre_col = self.aggregator.pre_indicator
        self.post_col = self.aggregator.post_indicator

        self.time_col_raw = self.aligner.time_col
        self.neuron_col = self.aggregator.neuron_col
        self.created_stat_col = created_stat_col
        self.created_neuron_col = created_neuron_col

        self.df_aligned_ = None
        self.df_diff_ = None

    def get_stat(self, df_traces: pd.DataFrame, events: Any) -> pd.DataFrame:
        self.df_aligned_ = self.aligner.average_trace_long(df_traces, events)
        if self.average_trace_preprocessor is not None:
            self.df_aligned_ = self.average_trace_preprocessor(self.df_aligned_)
        self.df_diff_ = self.aggregator.prepost_diff(self.df_aligned_)

        self.df_diff_ = self.df_diff_.rename(
            columns={
                self.aggregator.created_diff_col: self.created_stat_col,
                self.aggregator.neuron_col: self.created_neuron_col,
            }
        )

        return self.df_diff_


class AUCDiffResponders:
    def __init__(
        self,
        aligner: BaseAligner,
        aggregator: PrePostAggregator,
        average_trace_preprocessor: Optional[TracePreprocessor] = None,
        rotator: Optional[SurrogateGenerator] = None,
        n_boot: int = 200,
        adjust_pval: bool = False,
        sided: str = "two_sided",
        created_pval_col: str = "pval",
        created_stat_col: str = "stat",
        created_neuron_col: str = "cell_id",
        created_sig_col: str = "sig",
        n_jobs: int = -1,
        _store_reps: bool = False,
    ):
        self.aucdiff_calculater = AUCDiff(
            aligner=aligner,
            average_trace_preprocessor=average_trace_preprocessor,
            aggregator=aggregator,
            created_stat_col=created_stat_col,
            created_neuron_col=created_neuron_col,
        )
        self.create_stat_col = created_stat_col
        self.created_neuron_col = created_neuron_col
        self.rotator = (
            rotator if rotator is not None else Rotater(time_col="time", copy=True)
        )
        self.n_boot = n_boot
        self.adjust_pval = adjust_pval
        self.sided = sided
        self.created_pval_col = created_pval_col
        self.created_sig_col = created_sig_col
        self.n_jobs = n_jobs
        self._store_reps = _store_reps

        self.df_obs_ = None
        self.df_bootreps_ = None
        self.df_pval_ = None

    def _single_run(
        self, df: pd.DataFrame, events: Any, rotate: bool = False
    ) -> pd.DataFrame:
        df = df.copy()
        if rotate:
            df = self.rotator(df)
        df = self.aucdiff_calculater.get_stat(df, events)
        df = df.loc[:, [self.created_neuron_col, self.create_stat_col]]
        return df

    def _get_reps(self, df: pd.DataFrame, events: Any) -> pd.DataFrame:
        frames = Parallel(n_jobs=self.n_jobs)(
            delayed(self._single_run)(df, events=events, rotate=True)
            for _ in range(self.n_boot)
        )
        df_res = pd.concat(frames, axis=0)
        df_res = df_res.assign(
            sample=lambda x: x.groupby(self.created_neuron_col).cumcount()
        ).sort_values([self.created_neuron_col, "sample"])
        return df_res

    def _get_obs(self, df: pd.DataFrame, events: Any) -> pd.DataFrame:
        return self._single_run(df, events=events, rotate=False)

    def _get_pvals(self, df_obs: pd.DataFrame, df_reps: pd.DataFrame):
        pvals = []
        for _, row in df_obs.iterrows():
            cell_id = row[self.created_neuron_col]
            obs_stat = row[self.create_stat_col]

            reps_stat = df_reps[df_reps[self.created_neuron_col] == cell_id][
                self.create_stat_col
            ].values

            if self.sided == "two_sided":
                pval = np.mean(np.abs(reps_stat) >= np.abs(obs_stat)) * 2
            elif self.sided == "lower":
                pval = np.mean(reps_stat <= obs_stat)
            elif self.sided == "higher":
                pval = np.mean(reps_stat >= obs_stat)
            pvals.append((cell_id, pval))

        df_pval = pd.DataFrame(
            pvals, columns=[self.created_neuron_col, self.created_pval_col]
        )
        return df_pval

    def _adjust_pvals(self, df_res: pd.DataFrame) -> pd.DataFrame:
        df_res[self.created_pval_col] = p_adjust(df_res[self.created_pval_col])
        return df_res

    def get_responders(self, df_traces: pd.DataFrame, events: Any) -> pd.DataFrame:
        self.df_obs_ = self._get_obs(df_traces, events)
        self.df_bootreps_ = self._get_reps(df_traces, events)
        self.df_pval_ = self._get_pvals(self.df_obs_, self.df_bootreps_)
        self.df_responders_ = pd.merge(
            self.df_obs_, self.df_pval_, on=self.created_neuron_col
        )
        if self.adjust_pval:
            self.df_responders_ = self._adjust_pvals(self.df_responders_)
        self.df_responders_[self.created_sig_col] = (
            self.df_responders_[self.created_pval_col] < 0.05
        )
        if not self._store_reps:
            del self.df_bootreps_
        return self.df_responders_
