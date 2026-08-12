import numpy as np
import pandas as pd
import pytest

from sutraml.feature_selection.filter import (
    CorrelationSelector,
)


# ---------------------------------------------------------------------------
# Fixtures
# ---------------------------------------------------------------------------


@pytest.fixture
def correlated_dataframe():
    """
    Dataset containing one highly correlated feature pair.

    feature_a and feature_b are perfectly positively correlated.
    feature_c is independent.
    """

    return pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [2, 4, 6, 8, 10],
            "feature_c": [5, 1, 4, 2, 8],
        }
    )


@pytest.fixture
def negatively_correlated_dataframe():
    """
    Dataset containing a perfectly negatively correlated pair.
    """

    return pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [5, 4, 3, 2, 1],
            "feature_c": [2, 7, 1, 8, 4],
        }
    )


@pytest.fixture
def independent_dataframe():
    """
    Dataset where no feature pair exceeds a high correlation threshold.
    """

    return pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [5, 1, 4, 2, 8],
            "feature_c": [3, 8, 1, 6, 4],
        }
    )


# ---------------------------------------------------------------------------
# Initialization
# ---------------------------------------------------------------------------


def test_default_initialization():
    """Selector should initialize with the expected defaults."""

    selector = CorrelationSelector()

    assert selector.threshold == 0.90
    assert selector.method == "pearson"
    assert selector.absolute is True


def test_custom_initialization():
    """Selector should preserve custom configuration."""

    selector = CorrelationSelector(
        threshold=0.80,
        method="spearman",
        absolute=False,
    )

    assert selector.threshold == 0.80
    assert selector.method == "spearman"
    assert selector.absolute is False


# ---------------------------------------------------------------------------
# Parameter validation
# ---------------------------------------------------------------------------


def test_threshold_must_be_numeric():
    """Non-numeric thresholds should raise TypeError."""

    selector = CorrelationSelector(
        threshold="0.90",
    )

    X = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [3, 2, 1],
        }
    )

    with pytest.raises(TypeError, match="threshold must be a numeric"):
        selector.fit(X)


@pytest.mark.parametrize(
    "threshold",
    [-0.1, 1.1, 2],
)
def test_threshold_must_be_between_zero_and_one(threshold):
    """Threshold must lie in [0, 1]."""

    selector = CorrelationSelector(
        threshold=threshold,
    )

    X = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [3, 2, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="threshold must be between 0 and 1",
    ):
        selector.fit(X)


@pytest.mark.parametrize(
    "method",
    [
        "invalid",
        "cosine",
        "euclidean",
    ],
)
def test_invalid_correlation_method(method):
    """Unsupported correlation methods should raise ValueError."""

    selector = CorrelationSelector(
        method=method,
    )

    X = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [3, 2, 1],
        }
    )

    with pytest.raises(
        ValueError,
        match="method must be one",
    ):
        selector.fit(X)


@pytest.mark.parametrize(
    "absolute",
    [1, 0, "True", None],
)
def test_absolute_must_be_boolean(absolute):
    """absolute must be a boolean."""

    selector = CorrelationSelector(
        absolute=absolute,
    )

    X = pd.DataFrame(
        {
            "a": [1, 2, 3],
            "b": [3, 2, 1],
        }
    )

    with pytest.raises(
        TypeError,
        match="absolute must be a boolean",
    ):
        selector.fit(X)


# ---------------------------------------------------------------------------
# Input validation
# ---------------------------------------------------------------------------


def test_non_numeric_features_are_rejected():
    """CorrelationSelector should reject categorical/string features."""

    X = pd.DataFrame(
        {
            "age": [20, 30, 40],
            "gender": ["M", "F", "M"],
        }
    )

    selector = CorrelationSelector()

    with pytest.raises(
        TypeError,
        match="only supports numeric features",
    ):
        selector.fit(X)


def test_numpy_array_is_supported():
    """Selector should accept NumPy arrays."""

    X = np.array(
        [
            [1, 2, 5],
            [2, 4, 1],
            [3, 6, 4],
            [4, 8, 2],
            [5, 10, 8],
        ]
    )

    selector = CorrelationSelector(
        threshold=0.90,
    )

    selector.fit(X)

    assert selector.n_features_in_ == 3
    assert selector.n_features_selected_ == 2


def test_empty_dataframe_is_rejected():
    """Empty input should be rejected by the base selector."""

    X = pd.DataFrame()

    selector = CorrelationSelector()

    with pytest.raises(ValueError, match="at least one feature"):
        selector.fit(X)


# ---------------------------------------------------------------------------
# Correlation detection
# ---------------------------------------------------------------------------


def test_highly_correlated_feature_is_removed(
    correlated_dataframe,
):
    """Highly correlated features should result in feature removal."""

    selector = CorrelationSelector(
        threshold=0.90,
    )

    selector.fit(correlated_dataframe)

    assert selector.removed_features_ == ["feature_b"]

    assert selector.selected_features_ == [
        "feature_a",
        "feature_c",
    ]


def test_no_features_removed_when_below_threshold(
    independent_dataframe,
):
    """No features should be removed when correlations are below threshold."""

    selector = CorrelationSelector(
        threshold=0.90,
    )

    selector.fit(independent_dataframe)

    assert selector.removed_features_ == []

    assert selector.selected_features_ == [
        "feature_a",
        "feature_b",
        "feature_c",
    ]


def test_threshold_controls_feature_removal(
    correlated_dataframe,
):
    """Changing the threshold should affect feature selection."""

    selector = CorrelationSelector(
        threshold=1.0,
    )

    selector.fit(correlated_dataframe)

    # Correlation is exactly 1.0 and the implementation uses
    # `>` rather than `>=`, so it should not be removed.
    assert selector.removed_features_ == []

    selector = CorrelationSelector(
        threshold=0.99,
    )

    selector.fit(correlated_dataframe)

    assert selector.removed_features_ == ["feature_b"]


# ---------------------------------------------------------------------------
# Absolute correlation
# ---------------------------------------------------------------------------


def test_negative_correlation_is_detected_when_absolute_is_true(
    negatively_correlated_dataframe,
):
    """Strong negative correlations should be detected by default."""

    selector = CorrelationSelector(
        threshold=0.90,
        absolute=True,
    )

    selector.fit(negatively_correlated_dataframe)

    assert selector.removed_features_ == ["feature_b"]


def test_negative_correlation_is_not_detected_when_absolute_is_false(
    negatively_correlated_dataframe,
):
    """
    Strong negative correlations should not be considered when
    absolute=False.
    """

    selector = CorrelationSelector(
        threshold=0.90,
        absolute=False,
    )

    selector.fit(negatively_correlated_dataframe)

    assert selector.removed_features_ == []


# ---------------------------------------------------------------------------
# Correlation matrix
# ---------------------------------------------------------------------------


def test_correlation_matrix_is_stored(
    correlated_dataframe,
):
    """Fitted selector should expose the correlation matrix."""

    selector = CorrelationSelector()

    selector.fit(correlated_dataframe)

    matrix = selector.get_correlation_matrix()

    assert isinstance(matrix, pd.DataFrame)

    assert matrix.shape == (3, 3)

    assert list(matrix.columns) == [
        "feature_a",
        "feature_b",
        "feature_c",
    ]

    assert matrix.loc[
        "feature_a",
        "feature_b",
    ] == pytest.approx(1.0)


def test_correlation_matrix_is_not_available_before_fit():
    """Accessing diagnostics before fit should fail."""

    selector = CorrelationSelector()

    with pytest.raises(
        RuntimeError,
        match="not fitted yet",
    ):
        selector.get_correlation_matrix()


# ---------------------------------------------------------------------------
# Correlated pairs
# ---------------------------------------------------------------------------


def test_correlated_pairs_are_returned(
    correlated_dataframe,
):
    """Selector should expose correlated feature pairs."""

    selector = CorrelationSelector(
        threshold=0.90,
    )

    selector.fit(correlated_dataframe)

    pairs = selector.get_correlated_pairs()

    assert isinstance(pairs, pd.DataFrame)

    assert list(pairs.columns) == [
        "feature_1",
        "feature_2",
        "correlation",
    ]

    assert len(pairs) == 1

    assert pairs.iloc[0]["feature_1"] == "feature_a"
    assert pairs.iloc[0]["feature_2"] == "feature_b"
    assert pairs.iloc[0]["correlation"] == pytest.approx(1.0)


def test_no_correlated_pairs_returns_empty_dataframe(
    independent_dataframe,
):
    """No correlated pairs should produce an empty DataFrame."""

    selector = CorrelationSelector(
        threshold=0.90,
    )

    selector.fit(independent_dataframe)

    pairs = selector.get_correlated_pairs()

    assert isinstance(pairs, pd.DataFrame)
    assert pairs.empty

    assert list(pairs.columns) == [
        "feature_1",
        "feature_2",
        "correlation",
    ]


# ---------------------------------------------------------------------------
# BaseFeatureSelector contract
# ---------------------------------------------------------------------------


def test_support_mask_matches_selected_features(
    correlated_dataframe,
):
    """support_ should correctly represent selected features."""

    selector = CorrelationSelector(
        threshold=0.90,
    )

    selector.fit(correlated_dataframe)

    expected = np.array(
        [
            True,
            False,
            True,
        ]
    )

    np.testing.assert_array_equal(
        selector.get_support(),
        expected,
    )


def test_support_indices_are_correct(
    correlated_dataframe,
):
    """get_support(indices=True) should return selected feature indices."""

    selector = CorrelationSelector(
        threshold=0.90,
    )

    selector.fit(correlated_dataframe)

    np.testing.assert_array_equal(
        selector.get_support(indices=True),
        np.array([0, 2]),
    )


def test_feature_names_out_are_correct(
    correlated_dataframe,
):
    """get_feature_names_out should return selected feature names."""

    selector = CorrelationSelector(
        threshold=0.90,
    )

    selector.fit(correlated_dataframe)

    np.testing.assert_array_equal(
        selector.get_feature_names_out(),
        np.array(
            [
                "feature_a",
                "feature_c",
            ],
            dtype=object,
        ),
    )


# ---------------------------------------------------------------------------
# Transform
# ---------------------------------------------------------------------------


def test_transform_returns_selected_features(
    correlated_dataframe,
):
    """transform should remove the correlated feature."""

    selector = CorrelationSelector(
        threshold=0.90,
    )

    selector.fit(correlated_dataframe)

    transformed = selector.transform(
        correlated_dataframe
    )

    assert isinstance(
        transformed,
        pd.DataFrame,
    )

    assert list(transformed.columns) == [
        "feature_a",
        "feature_c",
    ]

    assert transformed.shape == (5, 2)


def test_fit_transform_returns_selected_features(
    correlated_dataframe,
):
    """fit_transform should fit and transform in one operation."""

    selector = CorrelationSelector(
        threshold=0.90,
    )

    transformed = selector.fit_transform(
        correlated_dataframe
    )

    assert list(transformed.columns) == [
        "feature_a",
        "feature_c",
    ]


def test_transform_preserves_original_values(
    correlated_dataframe,
):
    """Selected feature values should remain unchanged."""

    selector = CorrelationSelector(
        threshold=0.90,
    )

    selector.fit(correlated_dataframe)

    transformed = selector.transform(
        correlated_dataframe
    )

    pd.testing.assert_series_equal(
        transformed["feature_a"],
        correlated_dataframe["feature_a"],
    )

    pd.testing.assert_series_equal(
        transformed["feature_c"],
        correlated_dataframe["feature_c"],
    )


# ---------------------------------------------------------------------------
# Fit / test separation
# ---------------------------------------------------------------------------


def test_transform_does_not_recalculate_correlation():
    """
    transform should use the feature-selection decision learned during fit.

    This is important for preventing test-set information from influencing
    feature selection.
    """

    X_train = pd.DataFrame(
        {
            "feature_a": [1, 2, 3, 4, 5],
            "feature_b": [2, 4, 6, 8, 10],
            "feature_c": [5, 1, 4, 2, 8],
        }
    )

    X_test = pd.DataFrame(
        {
            "feature_a": [10, 20],
            "feature_b": [100, 200],
            "feature_c": [3, 7],
        }
    )

    selector = CorrelationSelector(
        threshold=0.90,
    )

    selector.fit(X_train)

    transformed = selector.transform(X_test)

    assert list(transformed.columns) == [
        "feature_a",
        "feature_c",
    ]