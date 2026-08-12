"""Unit tests for VarianceSelector."""

import numpy as np
import pandas as pd
import pytest

from heart_stroke_prediction.analyze.feature_selection import (
    VarianceSelector,
)


class TestVarianceSelector:
    """Tests for VarianceSelector."""

    def test_default_threshold_removes_constant_features(self):
        """Default threshold=0 should remove constant features."""

        X = pd.DataFrame(
            {
                "age": [20, 30, 40, 50, 60],
                "constant": [1, 1, 1, 1, 1],
                "glucose": [80, 100, 120, 140, 160],
            }
        )

        selector = VarianceSelector()

        X_selected = selector.fit_transform(X)

        assert selector.threshold == 0.0
        assert X_selected.shape == (5, 2)
        assert list(X_selected.columns) == [
            "age",
            "glucose",
        ]

    def test_threshold_removes_low_variance_features(self):
        """Features below or equal to threshold should be removed."""

        X = pd.DataFrame(
            {
                "high_variance": [1, 10, 20, 30, 40],
                "low_variance": [1.00, 1.01, 1.00, 1.01, 1.00],
                "constant": [5, 5, 5, 5, 5],
            }
        )

        selector = VarianceSelector(threshold=0.001)

        selector.fit(X)

        assert selector.get_support().tolist() == [
            True,
            False,
            False,
        ]

    def test_get_support_returns_boolean_mask(self):
        """get_support should return one boolean value per input feature."""

        X = pd.DataFrame(
            {
                "feature_a": [1, 2, 3, 4],
                "feature_b": [5, 5, 5, 5],
                "feature_c": [10, 20, 30, 40],
            }
        )

        selector = VarianceSelector()
        selector.fit(X)

        support = selector.get_support()

        assert support.dtype == bool
        assert support.shape == (3,)
        assert support.tolist() == [
            True,
            False,
            True,
        ]

    def test_get_support_indices(self):
        """get_support(indices=True) should return selected indices."""

        X = pd.DataFrame(
            {
                "a": [1, 2, 3],
                "b": [1, 1, 1],
                "c": [10, 20, 30],
                "d": [2, 2, 2],
            }
        )

        selector = VarianceSelector()
        selector.fit(X)

        indices = selector.get_support(indices=True)

        np.testing.assert_array_equal(
            indices,
            np.array([0, 2]),
        )

    def test_get_feature_names_out(self):
        """Selected feature names should be returned correctly."""

        X = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "constant": [1, 1, 1],
                "glucose": [80, 100, 120],
            }
        )

        selector = VarianceSelector()
        selector.fit(X)

        feature_names = selector.get_feature_names_out()

        np.testing.assert_array_equal(
            feature_names,
            np.array(["age", "glucose"]),
        )

    def test_transform_preserves_dataframe(self):
        """Transforming a DataFrame should return a DataFrame."""

        X = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "constant": [1, 1, 1],
                "glucose": [80, 100, 120],
            }
        )

        selector = VarianceSelector()
        selector.fit(X)

        transformed = selector.transform(X)

        assert isinstance(transformed, pd.DataFrame)
        assert list(transformed.columns) == [
            "age",
            "glucose",
        ]

    def test_variances_are_stored(self):
        """The selector should expose learned feature variances."""

        X = pd.DataFrame(
            {
                "a": [1, 2, 3, 4],
                "b": [5, 5, 5, 5],
            }
        )

        selector = VarianceSelector()
        selector.fit(X)

        variances = selector.get_feature_variances()

        assert variances.shape == (2,)

        np.testing.assert_allclose(
            variances,
            np.array([1.25, 0.0]),
        )

    def test_feature_statistics(self):
        """Feature statistics should contain all input features."""

        X = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "constant": [1, 1, 1],
                "glucose": [80, 100, 120],
            }
        )

        selector = VarianceSelector()
        selector.fit(X)

        report = selector.get_feature_statistics()

        assert list(report.columns) == [
            "Feature",
            "Variance",
            "Selected",
        ]

        assert len(report) == 3

        assert report["Feature"].tolist() == [
            "age",
            "constant",
            "glucose",
        ]

        assert report["Selected"].tolist() == [
            True,
            False,
            True,
        ]

    def test_numpy_array_is_supported(self):
        """VarianceSelector should support numpy arrays."""

        X = np.array(
            [
                [1, 10, 5],
                [2, 10, 5],
                [3, 10, 5],
                [4, 10, 5],
            ]
        )

        selector = VarianceSelector()

        transformed = selector.fit_transform(X)

        assert transformed.shape == (4, 1)

        np.testing.assert_array_equal(
            transformed.ravel(),
            np.array([1, 2, 3, 4]),
        )

    def test_non_numeric_features_raise_error(self):
        """Non-numeric features should raise TypeError."""

        X = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "gender": ["M", "F", "M"],
            }
        )

        selector = VarianceSelector()

        with pytest.raises(
            TypeError,
            match="requires numeric features",
        ):
            selector.fit(X)

    def test_negative_threshold_raises_error(self):
        """Negative threshold should not be accepted."""

        with pytest.raises(
            ValueError,
            match="greater than or equal to 0",
        ):
            VarianceSelector(threshold=-0.1)

    def test_non_numeric_threshold_raises_error(self):
        """Threshold must be numeric."""

        with pytest.raises(
            TypeError,
            match="threshold must be a numeric value",
        ):
            VarianceSelector(threshold="0.1")

    def test_infinite_threshold_raises_error(self):
        """Infinite threshold should not be accepted."""

        with pytest.raises(
            ValueError,
            match="threshold must be finite",
        ):
            VarianceSelector(threshold=np.inf)

    def test_transform_before_fit_raises_error(self):
        """Transform should fail when selector has not been fitted."""

        X = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "glucose": [80, 100, 120],
            }
        )

        selector = VarianceSelector()

        with pytest.raises(
            RuntimeError,
            match="not fitted yet",
        ):
            selector.transform(X)

    def test_get_support_before_fit_raises_error(self):
        """get_support should fail before fitting."""

        selector = VarianceSelector()

        with pytest.raises(
            RuntimeError,
            match="not fitted yet",
        ):
            selector.get_support()

    def test_all_features_removed_raises_error(self):
        """An empty selected feature set should raise ValueError."""

        X = pd.DataFrame(
            {
                "a": [1, 1, 1],
                "b": [2, 2, 2],
                "c": [3, 3, 3],
            }
        )

        selector = VarianceSelector()

        with pytest.raises(
            ValueError,
            match="empty feature set",
        ):
            selector.fit(X)

    def test_X_and_y_length_mismatch_raises_error(self):
        """Base selector should reject mismatched X and y."""

        X = pd.DataFrame(
            {
                "age": [20, 30, 40],
                "glucose": [80, 100, 120],
            }
        )

        y = np.array([0, 1])

        selector = VarianceSelector()

        with pytest.raises(
            ValueError,
            match="inconsistent numbers of samples",
        ):
            selector.fit(X, y)

    def test_custom_threshold_is_preserved(self):
        """The configured threshold should be preserved."""

        selector = VarianceSelector(threshold=2.5)

        assert selector.threshold == 2.5