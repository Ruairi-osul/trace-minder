from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.decomposition import PCA, KernelPCA
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import cross_val_score, KFold
from typing import Optional
from sklearn.base import BaseEstimator
from joblib import parallel, delayed
from sklearn.feature_selection import SelectKBest, f_classif
from sklearn.base import BaseEstimator, clone
from copy import deepcopy
from .preprocessors import OffsetPreprocesser, RandomizedPreprocessor
import pandas as pd
from tqdm import tqdm


class OffsetDecoderDispatcher:
    """
    Takes a decoder and preprocessers and runs the decoder on each offset.

    Parameters
    ----------
    offset_decoder : BaseEstimator
        A scikit-learn estimator that can be fit and predict.
    trace_preprocessor : OffsetPreprocesser
        A preprocessor that can be used to get X and y data at a given offset.
    cross_validator : Optional[BaseEstimator], optional
        A scikit-learn cross validator, by default None (uses KFold with 5 folds).
    scoring : str, optional
        A scikit-learn scoring metric, by default "f1_macro".
    n_jobs : int, optional
        Number of jobs to run in parallel, by default -1.

    Methods
    -------
    fit_single_offset(offset)
        Fits the decoder on the data at a single offset.
    fit_all_offsets()
        Fits the decoder on the data at all offsets.
    """

    def __init__(
        self,
        offset_decoder: BaseEstimator,
        trace_preprocessor: OffsetPreprocesser,
        cross_validator: Optional[BaseEstimator] = None,
        scoring: str = "f1_macro",
        n_jobs: int = -1,
    ):
        self.offset_decoder = offset_decoder
        self.trace_preprocessor = trace_preprocessor
        self.label_encoder = LabelEncoder()
        self.cross_validator = cross_validator
        self.scoring = scoring
        self.n_jobs = n_jobs

        self.offsets = self.trace_preprocessor.offsets

    @staticmethod
    def _fit_single_offset_static(X, y, offset_decoder, cross_validator, scoring):
        scores = cross_val_score(
            offset_decoder, X, y, cv=cross_validator, scoring=scoring
        )
        return scores.mean()

    def fit_single_offset(self, offset):
        X, y = self.trace_preprocessor.get_X_y_at_offset(offset)
        y = self.label_encoder.fit_transform(y)
        scores = cross_val_score(
            self.offset_decoder, X, y, cv=self.cross_validator, scoring=self.scoring
        )
        return scores.mean()

    def fit_all_offsets(self):
        data = [
            {
                "X": X,
                "y": y,
                "offset_decoder": self.offset_decoder,
                "cross_validator": self.cross_validator,
                "scoring": self.scoring,
            }
            for X, y in [
                self.trace_preprocessor.get_X_y_at_offset(offset)
                for offset in self.offsets
            ]
        ]
        scores = parallel.Parallel(n_jobs=self.n_jobs)(
            delayed(self._fit_single_offset_static)(**kwargs) for kwargs in data
        )

        return pd.DataFrame(dict(offsets=self.offsets, scores=scores))


class NestedRandomizedDispatcher:
    """
    Runs an offset decoder on a randomized preprocessor multiple times.

    A randomized preprocessor is a preprocessor that at generates data from
    at least one random source. It must have the reroll method.

    Parameters
    ----------
    offset_decoder : BaseEstimator
        A scikit-learn estimator that can be fit and predict.
    trace_preprocessor : RandomizedPreprocessor
        A preprocessor that can be used to get X and y data at a given offset.
    cross_validator : Optional[BaseEstimator], optional
        A scikit-learn cross validator, by default None (uses KFold with 5 folds).
    scoring : str, optional
        A scikit-learn scoring metric, by default "f1_macro".
    num_runs : int, optional
        Number of times to reroll the preprocessor, by default 75.
    n_jobs : int, optional
        Number of jobs to run in parallel, by default -1.
    verbose : bool, optional
        Whether to show a progress bar indicating the current run, by default False.
    """

    def __init__(
        self,
        offset_decoder: BaseEstimator,
        trace_preprocessor: RandomizedPreprocessor,
        cross_validator: Optional[BaseEstimator] = None,
        scoring: str = "f1_macro",
        num_runs: int = 75,
        n_jobs: int = -1,
        verbose: bool = False,
    ):
        self.offset_decoder = offset_decoder
        self.trace_preprocessor = trace_preprocessor
        self.cross_validator = cross_validator
        self.scoring = scoring
        self.num_runs = num_runs
        self.n_jobs = n_jobs
        self.verbose = verbose

    def run(self):
        iterater = tqdm(range(self.num_runs)) if self.verbose else range(self.num_runs)

        result_dfs = []
        for i in iterater:
            dispatcher = OffsetDecoderDispatcher(
                offset_decoder=clone(self.offset_decoder),
                trace_preprocessor=self.trace_preprocessor.reroll(),
                cross_validator=self.cross_validator,
                scoring=self.scoring,
                n_jobs=self.n_jobs,
            )
            result_df = dispatcher.fit_all_offsets()
            result_df["run"] = i
            result_dfs.append(result_df)

        return pd.concat(result_dfs)
