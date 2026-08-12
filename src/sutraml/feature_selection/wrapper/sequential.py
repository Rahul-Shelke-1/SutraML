"""Sequential wrapper feature selection."""

from __future__ import annotations

from typing import Any, Optional

import numpy as np
from sklearn.feature_selection import SequentialFeatureSelector
from sklearn.model_selection import BaseCrossValidator

from ..base import BaseFeatureSelector


class SequentialWrapperSelector(BaseFeatureSelector):
    """Wrapper feature selector backed by sklearn's SequentialFeatureSelector.

    Forward selection starts with no features and progressively adds features.

    Backward elimination starts with all features and progressively removes
    features.

    Parameters
    ----------
    estimator:
        Estimator used to evaluate feature subsets.

    direction:
        Selection direction. Must be ``"forward"`` or ``"backward"``.

    n_features_to_select:
        Number of features to select. Can be an integer, a float in (0, 1),
        or ``"auto"``.

    scoring:
        Scoring strategy passed to sklearn.

    cv:
        Cross-validation strategy.

    tol:
        Optional tolerance passed to sklearn.

    n_jobs:
        Number of jobs used by sklearn.
    """

    def __init__(
        self,
        estimator: Any,
        direction: str = "forward",
        n_features_to_select: Optional[int | float | str] = "auto",
        *,
        scoring: Optional[str] = None,
        cv: int | BaseCrossValidator = 5,
        tol: Optional[float] = None,
        n_jobs: Optional[int] = None,
    ) -> None:
        self.estimator = estimator
        self.direction = direction
        self.n_features_to_select = n_features_to_select
        self.scoring = scoring
        self.cv = cv
        self.tol = tol
        self.n_jobs = n_jobs

    def _fit(self, X: Any, y: Any = None) -> np.ndarray:
        """Fit sklearn's sequential selector and return its support mask."""
        if y is None:
            raise ValueError(
                "SequentialWrapperSelector requires target values `y`."
            )

        self._validate_direction()

        self.selector_ = SequentialFeatureSelector(
            estimator=self.estimator,
            direction=self.direction,
            n_features_to_select=self.n_features_to_select,
            scoring=self.scoring,
            cv=self.cv,
            tol=self.tol,
            n_jobs=self.n_jobs,
        )

        self.selector_.fit(X, y)

        return self.selector_.get_support()

    def _validate_direction(self) -> None:
        """Validate the selection direction."""
        if self.direction not in {"forward", "backward"}:
            raise ValueError(
                "direction must be either 'forward' or 'backward'. "
                f"Got {self.direction!r}."
            )

    def get_feature_scores(self) -> np.ndarray:
        """Return sklearn's feature scores when available.

        SequentialFeatureSelector does not expose a general feature-importance
        score because selection is based on subset performance. Therefore this
        method intentionally raises rather than returning misleading scores.
        """
        self._check_is_fitted()

        raise NotImplementedError(
            "Sequential feature selection does not produce independent "
            "feature scores. Use get_support() or get_feature_names_out()."
        )