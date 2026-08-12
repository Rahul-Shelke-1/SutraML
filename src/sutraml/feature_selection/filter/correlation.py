from __future__ import annotations

from typing import Any, Literal

import numpy as np
import pandas as pd

from ..base import BaseFeatureSelector


class CorrelationSelector(BaseFeatureSelector):
    """
    Filter selector that removes highly correlated features.

    This selector performs unsupervised feature selection based on
    feature-to-feature correlation.

    Parameters
    ----------
    threshold : float, default=0.90
        Correlation threshold above which a feature is considered
        highly correlated with another feature.

    method : {"pearson", "spearman", "kendall"}, default="pearson"
        Correlation method used to calculate the feature correlation matrix.

    absolute : bool, default=True
        Whether to use absolute correlation values.

        If True:
            correlation = -0.95 and correlation = +0.95
            are both considered highly correlated.

        If False:
            only positive correlations above the threshold are considered.

    Notes
    -----
    The selector does not use ``y``.

    The selector is responsible only for:
        1. calculating feature correlations,
        2. identifying redundant features,
        3. returning the feature-selection support mask.

    Visualization and reporting should be handled by a higher-level layer.

    Attributes
    ----------
    correlation_matrix_ : pandas.DataFrame
        Correlation matrix learned during fitting.

    correlated_pairs_ : pandas.DataFrame
        Feature pairs whose correlation exceeds the threshold.

    removed_features_ : list
        Names of features removed by the selector.
    """

    def __init__(
        self,
        threshold: float = 0.90,
        method: Literal[
            "pearson",
            "spearman",
            "kendall",
        ] = "pearson",
        absolute: bool = True,
    ) -> None:
        super().__init__()

        self.threshold = threshold
        self.method = method
        self.absolute = absolute

    def _fit(
        self,
        X: Any,
        y: Any = None,
    ) -> np.ndarray:
        """
        Learn the feature-selection mask.

        Parameters
        ----------
        X : array-like
            Training feature matrix.

        y : array-like, optional
            Ignored. Correlation filtering is unsupervised.

        Returns
        -------
        numpy.ndarray
            Boolean support mask where:

                True  -> feature is selected
                False -> feature is removed
        """

        self._validate_parameters()

        # ---------------------------------------------------------
        # Convert X to DataFrame.
        #
        # This gives us a consistent interface for calculating
        # correlations while still allowing BaseFeatureSelector
        # to accept NumPy arrays and pandas DataFrames.
        # ---------------------------------------------------------
        X_df = self._to_dataframe(X)

        # ---------------------------------------------------------
        # Validate numeric features.
        # ---------------------------------------------------------
        non_numeric_features = X_df.select_dtypes(
            exclude=np.number
        ).columns.tolist()

        if non_numeric_features:
            raise TypeError(
                "CorrelationSelector only supports numeric features. "
                f"Non-numeric features found: {non_numeric_features}"
            )

        # ---------------------------------------------------------
        # Calculate feature-to-feature correlation.
        # ---------------------------------------------------------
        self.correlation_matrix_ = X_df.corr(
            method=self.method
        )

        # ---------------------------------------------------------
        # Work with only the upper triangle.
        #
        # This avoids evaluating:
        #
        #     age -> bmi
        #     bmi -> age
        #
        # twice.
        # ---------------------------------------------------------
        upper_triangle = self.correlation_matrix_.where(
            np.triu(
                np.ones(
                    self.correlation_matrix_.shape,
                    dtype=bool,
                ),
                k=1,
            )
        )

        if self.absolute:
            comparison_matrix = upper_triangle.abs()
        else:
            comparison_matrix = upper_triangle

        # ---------------------------------------------------------
        # Find highly correlated feature pairs.
        # ---------------------------------------------------------
        correlated_pairs = []

        for column in comparison_matrix.columns:

            correlated_rows = comparison_matrix.index[
                comparison_matrix[column] > self.threshold
            ]

            for row in correlated_rows:
                correlated_pairs.append(
                    {
                        "feature_1": row,
                        "feature_2": column,
                        "correlation": self.correlation_matrix_.loc[
                            row,
                            column,
                        ],
                    }
                )

        self.correlated_pairs_ = pd.DataFrame(
            correlated_pairs,
            columns=[
                "feature_1",
                "feature_2",
                "correlation",
            ],
        )

        # ---------------------------------------------------------
        # Determine which features to remove.
        #
        # Current deterministic strategy:
        #
        #     [A, B] highly correlated
        #
        # Keep A
        # Remove B
        #
        # Therefore feature ordering determines which member
        # of a correlated pair survives.
        # ---------------------------------------------------------
        removed_features = set()

        for column in comparison_matrix.columns:

            correlated_rows = comparison_matrix.index[
                comparison_matrix[column] > self.threshold
            ]

            if len(correlated_rows) > 0:
                removed_features.add(column)

        # ---------------------------------------------------------
        # Store diagnostic information.
        #
        # These are data artifacts, NOT visualization.
        # ---------------------------------------------------------
        self.removed_features_ = [
            feature
            for feature in X_df.columns
            if feature in removed_features
        ]

        self.selected_features_ = [
            feature
            for feature in X_df.columns
            if feature not in removed_features
        ]

        # ---------------------------------------------------------
        # Build support mask.
        #
        # BaseFeatureSelector.fit() will validate this mask and
        # store it as self.support_.
        # ---------------------------------------------------------
        support = np.asarray(
            [
                feature not in removed_features
                for feature in X_df.columns
            ],
            dtype=bool,
        )

        return support

    def _validate_parameters(self) -> None:
        """Validate selector parameters."""

        if not isinstance(
            self.threshold,
            (int, float),
        ):
            raise TypeError(
                "threshold must be a numeric value."
            )

        if not 0 <= self.threshold <= 1:
            raise ValueError(
                "threshold must be between 0 and 1."
            )

        valid_methods = {
            "pearson",
            "spearman",
            "kendall",
        }

        if self.method not in valid_methods:
            raise ValueError(
                f"method must be one of {valid_methods}. "
                f"Got '{self.method}'."
            )

        if not isinstance(
            self.absolute,
            bool,
        ):
            raise TypeError(
                "absolute must be a boolean."
            )

    def _to_dataframe(
        self,
        X: Any,
    ) -> pd.DataFrame:
        """
        Convert X into a DataFrame while preserving feature names.

        For a DataFrame:
            existing column names are preserved.

        For a NumPy array:
            generic names x0, x1, x2, ... are generated.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Returns
        -------
        pandas.DataFrame
            DataFrame representation of X.
        """

        if isinstance(X, pd.DataFrame):
            return X.copy()

        X_array = np.asarray(X)

        columns = [
            f"x{i}"
            for i in range(X_array.shape[1])
        ]

        return pd.DataFrame(
            X_array,
            columns=columns,
        )

    def get_correlated_pairs(self) -> pd.DataFrame:
        """
        Return feature pairs exceeding the correlation threshold.

        Returns
        -------
        pandas.DataFrame
            DataFrame with:

                feature_1
                feature_2
                correlation
        """

        self._check_is_fitted()

        return self.correlated_pairs_.copy()

    def get_correlation_matrix(self) -> pd.DataFrame:
        """
        Return the correlation matrix learned during fitting.

        This method exposes the analytical data for higher-level
        reporting or visualization layers.

        Returns
        -------
        pandas.DataFrame
            Feature correlation matrix.
        """

        self._check_is_fitted()

        return self.correlation_matrix_.copy()