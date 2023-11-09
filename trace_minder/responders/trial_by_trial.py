import pandas as pd
import numpy as np
from typing import Callable
from calcium_clear.align.align_events import align_to_events
from scipy.stats import wilcoxon, ttest_rel
from calcium_clear.stats import auc, p_adjust
from trace_minder.align.base import BaseAligner
from trace_minder.trace_aggregation import PrePostAggregator
from typing import Union


class TrialPairedResponders:
    def __init__(
        self,
        aligner: BaseAligner,
        aggregator: PrePostAggregator,
        adjust_pval: bool = False,
    ):
        self.aligner = aligner
        self.aggregator = aggregator
        self.pre_col = self.aggregator.pre_indicator
        self.post_col = self.aggregator.post_indicator
        self.neuron_col = self.aggregator.neuron_col
        self.adjust_pval = adjust_pval

    def compare_agg(self, df: pd.DataFrame) -> pd.Series:
        raise NotImplementedError

    def get_responders(
        self, df_traces: pd.DataFrame, events: Union[np.ndarray, pd.DataFrame]
    ) -> pd.DataFrame:
        df_traces = df_traces.copy()
        df_aligned = self.aligner.align_long(df_traces, events)
        df_agg = self.aggregator.aggregate(df_aligned)
        df_responders = (
            df_agg.groupby(self.neuron_col).apply(self.compare_agg).reset_index()
        )
        if self.adjust_pval:
            df_responders["p_adj"] = p_adjust(df_responders["p"])
        return df_responders


class WilcoxonResponders(TrialPairedResponders):
    def compare_agg(self, df: pd.DataFrame) -> pd.Series:
        out = {}
        out["diff_of_means"] = (df[self.post_col] - df[self.pre_col]).mean()
        out["W"], out["p"] = wilcoxon(df[self.post_col], df[self.pre_col])
        out["direction"] = np.where(
            out["diff_of_means"] > 0, "activation", "inhibition"
        )
        return pd.Series(out)


class TtestResponders(TrialPairedResponders):
    def compare_agg(self, df: pd.DataFrame) -> pd.Series:
        out = {}
        out["diff_of_means"] = (df[self.post_col] - df[self.pre_col]).mean()
        out["t"], out["p"] = ttest_rel(df[self.post_col], df[self.pre_col])
        out["direction"] = np.where(
            out["diff_of_means"] > 0, "activation", "inhibition"
        )
        return pd.Series(out)
