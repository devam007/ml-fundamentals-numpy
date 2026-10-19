import numpy as np
import pytest
from sklearn.cluster import KMeans as ReferenceKMeans
from sklearn.datasets import make_blobs, make_classification
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import adjusted_rand_score
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from ml_fundamentals import KMeans, KNNClassifier, LinearRegressionGD, LogisticRegressionGD


def test_linear_recovers_coefficients_and_matches_reference():
    features = np.random.default_rng(1).normal(size=(200, 3))
    target = features @ [2, -3, 0.5] + 4
    model = LinearRegressionGD().fit(features, target)
    reference = LinearRegression().fit(features, target)
    np.testing.assert_allclose(model.coef_, [2, -3, 0.5], atol=1e-7)
    np.testing.assert_allclose(model.predict(features), reference.predict(features), atol=1e-7)
    assert model.loss_history_[-1] < model.loss_history_[0]


def test_logistic_predictions_and_probability_normalization():
    features, target = make_classification(
        n_samples=300, n_features=4, n_redundant=0, random_state=4
    )
    features = StandardScaler().fit_transform(features)
    model = LogisticRegressionGD(n_iter=4000).fit(features, target)
    reference = LogisticRegression(C=1e8, max_iter=2000).fit(features, target)
    assert np.mean(model.predict(features) == reference.predict(features)) > 0.98
    np.testing.assert_allclose(model.predict_proba(features).sum(axis=1), 1)
    assert model.loss_history_[-1] < model.loss_history_[0]
    with np.errstate(over="raise", invalid="raise"):
        np.testing.assert_allclose(model._sigmoid(np.array([-1000.0, 0.0, 1000.0])), [0, 0.5, 1])


def test_knn_repeated_predictions_and_training_copy():
    features = np.array([[0.0], [1.0], [5.0], [6.0]])
    labels = np.array([0, 0, 1, 1])
    model = KNNClassifier(3).fit(features, labels)
    reference = KNeighborsClassifier(3).fit(features, labels)
    query = [[0.2], [5.8]]
    expected = reference.predict(query)
    features[:] = 999
    labels[:] = 9
    np.testing.assert_array_equal(model.predict(query), expected)
    np.testing.assert_array_equal(model.predict(query), expected)


def test_knn_ties_and_string_labels():
    model = KNNClassifier(2).fit([[0], [2]], ["z", "a"])
    assert model.predict([[1]])[0] == "a"


def test_kmeans_matches_separated_clusters_and_is_repeatable():
    features, target = make_blobs(n_samples=150, centers=3, cluster_std=0.3, random_state=7)
    model = KMeans().fit(features)
    reference = ReferenceKMeans(3, n_init=10, random_state=42).fit(features)
    assert adjusted_rand_score(target, model.labels_) == 1
    assert model.inertia_ == pytest.approx(reference.inertia_, rel=1e-6)
    np.testing.assert_array_equal(model.predict(features), model.labels_)
    np.testing.assert_allclose(KMeans().fit(features).cluster_centers_, model.cluster_centers_)


def test_kmeans_duplicate_points_and_iteration_limit():
    model = KMeans(3, max_iter=1).fit(np.ones((4, 2)))
    assert model.inertia_ == 0
    assert np.isfinite(model.cluster_centers_).all()
    np.testing.assert_array_equal(model.predict(np.ones((4, 2))), model.labels_)


@pytest.mark.parametrize(
    "model", [LinearRegressionGD(), LogisticRegressionGD(), KNNClassifier(), KMeans()]
)
def test_predict_before_fit(model):
    with pytest.raises(ValueError, match="fit"):
        model.predict([[1]])


@pytest.mark.parametrize("features", [[], [1, 2], [[np.nan]], [[np.inf]], [[]]])
def test_invalid_features(features):
    with pytest.raises(ValueError):
        LinearRegressionGD().fit(features, [1])


def test_invalid_labels_parameters_and_feature_count():
    for constructor, kwargs in [
        (KNNClassifier, {"n_neighbors": 0}),
        (KMeans, {"n_init": 0}),
        (LinearRegressionGD, {"learning_rate": -1}),
    ]:
        with pytest.raises(ValueError):
            constructor(**kwargs)
    with pytest.raises(ValueError):
        LogisticRegressionGD().fit([[0], [1]], [0, 2])
    with pytest.raises(ValueError):
        KNNClassifier(3).fit([[0], [1]], [0, 1])
    with pytest.raises(ValueError):
        KMeans(3).fit([[0], [1]])
    with pytest.raises(ValueError):
        LinearRegressionGD().fit([[0], [1]], [1])
    with pytest.raises(ValueError):
        LinearRegressionGD().fit([[0], [1]], [0, 1]).predict([[1, 2]])
