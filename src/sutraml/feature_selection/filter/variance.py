"""Variance-based feature selection.

This module contains unsupervised filtering based on feature variance.
Features whose variance is less than or equal to ``threshold`` are removed.
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd
from sklearn.feature_selection import VarianceThreshold

from ..base import BaseFeatureSelector


class VarianceSelector(BaseFeatureSelector):
    """Select features whose variance is greater than a threshold.

    Parameters
    ----------
    threshold : float, default=0.0
        Features with variance less than or equal to this value are removed.

        ``threshold=0.0`` removes constant features only.

    Notes
    -----
    Variance filtering is an unsupervised filter method, therefore ``y`` is
    ignored.
    """

    def __init__(self, threshold: float = 0.0) -> None:
        if not isinstance(
            threshold,
            (int, float, np.integer, np.floating),
        ):
            raise TypeError("threshold must be a numeric value.")

        if not np.isfinite(threshold):
            raise ValueError("threshold must be finite.")

        if threshold < 0:
            raise ValueError(
                "threshold must be greater than or equal to 0."
            )

        self.threshold = float(threshold)

    def _fit(self, X: Any, y: Any = None) -> np.ndarray:
        """Learn feature variances and return the support mask."""

        X_array = self._to_numeric_array(X)

        selector = VarianceThreshold(
            threshold=self.threshold
        )

        selector.fit(X_array)

        self.variances_ = np.asarray(
            selector.variances_,
            dtype=float,
        )

        self.threshold_ = float(selector.threshold)

        return self.variances_ > self.threshold_

    def get_feature_variances(self) -> np.ndarray:
        """Return variance for every input feature."""

        self._check_is_fitted()

        return self.variances_.copy()

    def get_feature_statistics(self) -> pd.DataFrame:
        """Return variance statistics for every input feature."""

        self._check_is_fitted()

        if self.feature_names_in_ is not None:
            feature_names = self.feature_names_in_
        else:
            feature_names = np.asarray(
                [
                    f"x{i}"
                    for i in range(self.n_features_in_)
                ],
                dtype=object,
            )

        return pd.DataFrame(
            {
                "Feature": feature_names,
                "Variance": self.variances_,
                "Selected": self.support_,
            }
        )

    @staticmethod
    def _to_numeric_array(X: Any) -> np.ndarray:
        """Convert X to a numeric 2-D numpy array."""

        if hasattr(X, "dtypes"):
            non_numeric = X.select_dtypes(
                exclude=np.number
            ).columns.tolist()

            if non_numeric:
                raise TypeError(
                    "VarianceSelector requires numeric features. "
                    f"Non-numeric features found: {non_numeric}"
                )

        try:
            X_array = np.asarray(X, dtype=float)

        except (TypeError, ValueError) as exc:
            raise TypeError(
                "VarianceSelector requires numeric features."
            ) from exc

        if X_array.ndim != 2:
            raise ValueError(
                "X must be a 2-dimensional array-like object."
            )

        return X_array