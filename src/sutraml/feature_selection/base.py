"""
Base interface for feature seleciton.

Thi module defines the common contract that all fetaure selcitors must follow.

Suuported selector families:
    - Filter
    - Wrapper
    - Embedded

The selector is intentionally responsible only for:
    1. Learning which fetaures should be selected.
    2. Transforming the input dataset using the learned selection.

Model evaluation, cross-validation, scoring, and experiment comparision
belong to a higher-level experiment/evaliation layer.
"""

from __future__ import __annotations__

from abc import ABC, abstractmethod
from typing import Any, Optional

import numpy as np
from sklearn.base import BaseEstimator, TransformerMixin


class BaseFetaureSelector(BaseEstimator, TransformerMixin, ABC):
    """
    Abstratct base class for fetaure-selection algorithms.

    The class follows the scikit-learn transformer interface so that
    feature selectors can eventually be used inside sklearn Pipelines.

    Concrete selectors only need to implement `_fit()` and return a
    boolean support mask.

    Example
    -------
    >>> class MySelector(BaseFeatureSelector):
    ...     def _fit(self, X, y=None):
    ...         return np.array([True, False, True])
    """
    def fit(
            self,
            X: Any,
            y: Any = None,
    ) -> "BaseFeatureSelector":
        """
        Fit the feature selector.

        Parameters
        ----------
        X: array-like
           Training fetaure metrix.

        y: array-like, optional
           Target values.

        Returns
        -------
        BaseFetuareSelector
            Fitted selector.
        """
        self._validate_X(X)

        # Validate X/y sample count when target is provided.
        if y is not None and len(X) != len(y):
            raise ValueError(
                "X and y have inconsistent numbers of smaples."
                f"{len(X)} != {len(y)}."
            )

        # Store basic information about the input dataset.
        self.n_features_in_ = X.shape[1]
        self.feature_names_in_ = self._get_feature_names(X)

        # Let the concrete selector learn the feature subset.
        support = np.asarray(
            self._fit(X, y),
            dtype=bool,
        )

        # Validate the support mask returned by the implementation.
        if support.ndim != 1:
            raise ValueError(
                "Feature-selection support mask must be one-dimensional."
            )

        if support.shape[0] != self.n_features_in_:
            raise ValueError(
                "Feature-selection support mask must contain exactly one "
                "value for every input feature. "
                f"Expected {self.n_features_in_}, "
                f"got {support.shape[0]}."
            )

        # A selector should not silently return zero features.
        if not support.any():
            raise ValueError(
                "Feature selection produced an empty feature set."
            )

        # Store learned state.
        self.support_ = support
        self.n_features_selected_ = int(support.sum())
        self.is_fitted_ = True

        return self

    @abstractmethod
    def _fit(
        self,
        X: Any,
        y: Any = None,
    ) -> np.ndarray:
        """
        Learn the feature-selection mask.

        This method must be implemented by every concrete selector.

        Parameters
        ----------
        X : array-like
            Training feature matrix.

        y : array-like, optional
            Target values.

        Returns
        -------
        numpy.ndarray
            Boolean array with shape:

                (n_features,)

            where:

                True  -> feature is selected
                False -> feature is not selected
        """
        raise NotImplementedError

    def transform(self, X: Any) -> Any:
        """
        Transform X by retaining only selected features.

        Parameters
        ----------
        X : array-like
            Feature matrix to transform.

        Returns
        -------
        array-like
            Dataset containing only selected features.
        """

        self._check_is_fitted()
        self._validate_X(X)

        # Ensure the same number of features is supplied.
        if X.shape[1] != self.n_features_in_:
            raise ValueError(
                "X has a different number of features than the data used "
                "to fit the selector: "
                f"{X.shape[1]} != {self.n_features_in_}."
            )

        # Preserve pandas DataFrame when possible.
        if hasattr(X, "iloc"):
            return X.iloc[:, self.support_]

        # Otherwise return a NumPy array.
        X_array = np.asarray(X)

        return X_array[:, self.support_]

    def fit_transform(
        self,
        X: Any,
        y: Any = None,
        **fit_params: Any,
    ) -> Any:
        """
        Fit the selector and transform X.

        Parameters
        ----------
        X : array-like
            Training feature matrix.

        y : array-like, optional
            Target values.

        **fit_params
            Additional fit parameters.

        Returns
        -------
        array-like
            Dataset containing selected features.
        """

        return self.fit(X, y).transform(X)

    def get_support(
        self,
        indices: bool = False,
    ) -> np.ndarray:
        """
        Return the selected feature mask or feature indices.

        Parameters
        ----------
        indices : bool, default=False
            If False, return a boolean mask.

            If True, return integer indices of selected features.

        Returns
        -------
        numpy.ndarray
            Boolean mask or integer feature indices.

        Examples
        --------
        >>> selector.get_support()
        array([ True, False, True])

        >>> selector.get_support(indices=True)
        array([0, 2])
        """

        self._check_is_fitted()

        if indices:
            return np.flatnonzero(self.support_)

        return self.support_.copy()

    def get_feature_names_out(
        self,
        input_features: Optional[Any] = None,
    ) -> np.ndarray:
        """
        Return names of the selected features.

        Parameters
        ----------
        input_features : array-like, optional
            Feature names.

            If not provided, names learned during ``fit`` are used.

        Returns
        -------
        numpy.ndarray
            Names of selected features.
        """

        self._check_is_fitted()

        if input_features is None:
            input_features = self.feature_names_in_

        # NumPy arrays don't contain column names.
        if input_features is None:
            input_features = np.asarray(
                [
                    f"x{i}"
                    for i in range(self.n_features_in_)
                ],
                dtype=object,
            )

        else:
            input_features = np.asarray(
                input_features,
                dtype=object,
            )

            if input_features.ndim != 1:
                raise ValueError(
                    "input_features must be one-dimensional."
                )

            if len(input_features) != self.n_features_in_:
                raise ValueError(
                    "input_features must contain one name per "
                    "input feature: "
                    f"{len(input_features)} != "
                    f"{self.n_features_in_}."
                )

        return input_features[self.support_]

    @staticmethod
    def _validate_X(X: Any) -> None:
        """
        Validate the minimum X interface required by selectors.

        Parameters
        ----------
        X : array-like
            Feature matrix.
        """

        if X is None:
            raise TypeError(
                "X cannot be None."
            )

        if not hasattr(X, "shape"):
            raise TypeError(
                "X must be a 2-dimensional array-like object."
            )

        if len(X.shape) != 2:
            raise ValueError(
                "X must be a 2-dimensional array-like object."
            )

        if X.shape[1] == 0:
            raise ValueError(
                "X must contain at least one feature."
            )

    @staticmethod
    def _get_feature_names(
        X: Any,
    ) -> Optional[np.ndarray]:
        """
        Extract feature names from a pandas DataFrame.

        Parameters
        ----------
        X : array-like
            Input feature matrix.

        Returns
        -------
        numpy.ndarray or None
            Feature names if available.
        """

        if hasattr(X, "columns"):
            return np.asarray(
                X.columns,
                dtype=object,
            )

        return None

    def _check_is_fitted(self) -> None:
        """
        Check whether the selector has already been fitted.

        Raises
        ------
        RuntimeError
            If ``fit()`` has not been called.
        """

        if not getattr(
            self,
            "is_fitted_",
            False,
        ):
            raise RuntimeError(
                "This feature selector is not fitted yet. "
                "Call fit(X, y) before using this method."
            )