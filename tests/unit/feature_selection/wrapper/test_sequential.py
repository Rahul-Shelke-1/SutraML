import numpy as np
import pandas as pd
import pytest
from sklearn.datasets import make_classification
from sklearn.linear_model import LogisticRegression

from sutraml.feature_selection.wrapper import (
    SequentialWrapperSelector,
)


@pytest.fixture
def classification_data():
    X, y = make_classification(
        n_samples=100,
        n_features=6,
        n_informative=3,
        n_redundant=1,
        random_state=42,
    )

    X = pd.DataFrame(
        X,
        columns=[
            "age",
            "bmi",
            "glucose",
            "hypertension",
            "heart_disease",
            "smoking",
        ],
    )

    return X, y


@pytest.fixture
def estimator():
    return LogisticRegression(
        max_iter=1000,
        random_state=42,
    )


def test_forward_selection_selects_requested_number_of_features(
    classification_data,
    estimator,
):
    X, y = classification_data

    selector = SequentialWrapperSelector(
        estimator=estimator,
        direction="forward",
        n_features_to_select=3,
        scoring="roc_auc",
        cv=3,
    )

    selector.fit(X, y)

    assert selector.n_features_in_ == 6
    assert selector.n_features_selected_ == 3
    assert selector.get_support().sum() == 3


def test_backward_elimination_selects_requested_number_of_features(
    classification_data,
    estimator,
):
    X, y = classification_data

    selector = SequentialWrapperSelector(
        estimator=estimator,
        direction="backward",
        n_features_to_select=3,
        scoring="roc_auc",
        cv=3,
    )

    selector.fit(X, y)

    assert selector.n_features_in_ == 6
    assert selector.n_features_selected_ == 3
    assert selector.get_support().sum() == 3


@pytest.mark.parametrize("direction", ["forward", "backward"])
def test_both_directions_produce_valid_support_mask(
    classification_data,
    estimator,
    direction,
):
    X, y = classification_data

    selector = SequentialWrapperSelector(
        estimator=estimator,
        direction=direction,
        n_features_to_select=3,
        scoring="roc_auc",
        cv=3,
    )

    selector.fit(X, y)

    support = selector.get_support()

    assert isinstance(support, np.ndarray)
    assert support.dtype == bool
    assert support.shape == (X.shape[1],)


@pytest.mark.parametrize(
    "direction",
    ["invalid", "", None, "Forward", "backward-elimination"],
)
def test_invalid_direction_is_rejected(
    classification_data,
    estimator,
    direction,
):
    X, y = classification_data

    selector = SequentialWrapperSelector(
        estimator=estimator,
        direction=direction,
        n_features_to_select=3,
        cv=3,
    )

    with pytest.raises(ValueError, match="direction"):
        selector.fit(X, y)


def test_target_is_required(classification_data, estimator):
    X, _ = classification_data

    selector = SequentialWrapperSelector(
        estimator=estimator,
        direction="forward",
        n_features_to_select=3,
        cv=3,
    )

    with pytest.raises(ValueError, match="requires target"):
        selector.fit(X)


def test_transform_preserves_selected_dataframe_columns(
    classification_data,
    estimator,
):
    X, y = classification_data

    selector = SequentialWrapperSelector(
        estimator=estimator,
        direction="forward",
        n_features_to_select=3,
        scoring="roc_auc",
        cv=3,
    )

    transformed = selector.fit_transform(X, y)

    expected_features = selector.get_feature_names_out()

    assert isinstance(transformed, pd.DataFrame)
    assert list(transformed.columns) == list(expected_features)
    assert transformed.shape == (X.shape[0], 3)


def test_get_support_indices_matches_feature_names(
    classification_data,
    estimator,
):
    X, y = classification_data

    selector = SequentialWrapperSelector(
        estimator=estimator,
        direction="forward",
        n_features_to_select=3,
        scoring="roc_auc",
        cv=3,
    )

    selector.fit(X, y)

    indices = selector.get_support(indices=True)
    names = selector.get_feature_names_out()

    assert len(indices) == 3
    assert list(names) == list(X.columns[indices])


def test_transform_before_fit_raises(classification_data, estimator):
    X, _ = classification_data

    selector = SequentialWrapperSelector(
        estimator=estimator,
        direction="forward",
        n_features_to_select=3,
        cv=3,
    )

    with pytest.raises(RuntimeError, match="not fitted"):
        selector.transform(X)