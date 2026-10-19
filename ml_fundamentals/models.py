"""Educational estimators. Inputs must already be scaled where appropriate."""

import numpy as np


def _features(values):
    features = np.asarray(values, dtype=float)
    if features.ndim != 2 or 0 in features.shape:
        raise ValueError("X must be a nonempty 2D matrix.")
    if not np.isfinite(features).all():
        raise ValueError("X must contain only finite numbers.")
    return features


def _target(values, size):
    target = np.asarray(values, dtype=float)
    if target.ndim != 1 or len(target) != size or not np.isfinite(target).all():
        raise ValueError("y must be a finite 1D array with one label per row.")
    return target


def _positive_integer(value, name):
    if isinstance(value, bool) or not isinstance(value, (int, np.integer)) or value < 1:
        raise ValueError(f"{name} must be a positive integer.")


class _Estimator:
    def _prediction_features(self, values):
        if not hasattr(self, "n_features_in_"):
            raise ValueError("Call fit before predict.")
        features = _features(values)
        if features.shape[1] != self.n_features_in_:
            raise ValueError("Prediction feature count must match training.")
        return features


class LinearRegressionGD(_Estimator):
    """Full-batch gradient descent on mean squared error, with an intercept."""

    def __init__(self, learning_rate=0.05, n_iter=2000):
        if not np.isfinite(learning_rate) or learning_rate <= 0:
            raise ValueError("learning_rate must be positive and finite.")
        _positive_integer(n_iter, "n_iter")
        self.learning_rate = learning_rate
        self.n_iter = n_iter

    def fit(self, X, y):
        features = _features(X)
        target = _target(y, len(features))
        self.n_features_in_ = features.shape[1]
        self.coef_ = np.zeros(self.n_features_in_)
        self.intercept_ = 0.0
        self.loss_history_ = []
        for _ in range(self.n_iter):
            error = features @ self.coef_ + self.intercept_ - target
            self.coef_ -= self.learning_rate * 2 * features.T @ error / len(features)
            self.intercept_ -= self.learning_rate * 2 * error.mean()
            residual = features @ self.coef_ + self.intercept_ - target
            loss = float(np.mean(residual**2))
            if not np.isfinite(loss):
                raise ValueError("Training diverged; scale X or reduce learning_rate.")
            self.loss_history_.append(loss)
        return self

    def predict(self, X):
        return self._prediction_features(X) @ self.coef_ + self.intercept_


class LogisticRegressionGD(LinearRegressionGD):
    """Binary logistic regression without regularization, labels 0 and 1."""

    @staticmethod
    def _sigmoid(logits):
        # exp(-logaddexp(0, -z)) is sigmoid(z) without exp overflow.
        return np.exp(-np.logaddexp(0, -logits))

    def fit(self, X, y):
        features = _features(X)
        target = _target(y, len(features))
        if not np.array_equal(np.unique(target), [0, 1]):
            raise ValueError("Training y must contain both binary classes 0 and 1.")
        self.n_features_in_ = features.shape[1]
        self.coef_ = np.zeros(self.n_features_in_)
        self.intercept_ = 0.0
        self.loss_history_ = []
        for _ in range(self.n_iter):
            error = self._sigmoid(features @ self.coef_ + self.intercept_) - target
            self.coef_ -= self.learning_rate * features.T @ error / len(features)
            self.intercept_ -= self.learning_rate * error.mean()
            logits = features @ self.coef_ + self.intercept_
            loss = float(np.mean(np.logaddexp(0, logits) - target * logits))
            if not np.isfinite(loss):
                raise ValueError("Training diverged; scale X or reduce learning_rate.")
            self.loss_history_.append(loss)
        return self

    def predict_proba(self, X):
        positive = self._sigmoid(self._prediction_features(X) @ self.coef_ + self.intercept_)
        return np.column_stack((1 - positive, positive))

    def predict(self, X):
        return (self.predict_proba(X)[:, 1] >= 0.5).astype(int)


class KNNClassifier(_Estimator):
    """Euclidean KNN; class ties choose the smallest label, distance ties row order."""

    def __init__(self, n_neighbors=3):
        _positive_integer(n_neighbors, "n_neighbors")
        self.n_neighbors = n_neighbors

    def fit(self, X, y):
        features = _features(X)
        target = np.asarray(y)
        if target.ndim != 1 or len(target) != len(features):
            raise ValueError("y must be 1D with one label per training row.")
        if target.dtype.kind not in "iuUS":
            raise ValueError("KNN labels must be integers or strings.")
        if self.n_neighbors > len(features):
            raise ValueError("n_neighbors cannot exceed the training sample count.")
        self.n_features_in_ = features.shape[1]
        self.X_ = features.copy()
        self.y_ = target.copy()
        return self

    def predict(self, X):
        predictions = []
        for row in self._prediction_features(X):
            distances = np.sum((self.X_ - row) ** 2, axis=1)
            nearest = np.argsort(distances, kind="stable")[: self.n_neighbors]
            labels, counts = np.unique(self.y_[nearest], return_counts=True)
            predictions.append(labels[np.argmax(counts)])
        return np.asarray(predictions)


class KMeans(_Estimator):
    """Lloyd's algorithm with seeded random initialization and multiple restarts.

    Empty clusters keep their previous center. Duplicate samples can therefore
    produce fewer than n_clusters distinct groups; this is intentional.
    """

    def __init__(self, n_clusters=3, max_iter=100, n_init=10, tol=1e-6, random_state=42):
        for name, value in [("n_clusters", n_clusters), ("max_iter", max_iter), ("n_init", n_init)]:
            _positive_integer(value, name)
        if not np.isfinite(tol) or tol < 0:
            raise ValueError("tol must be finite and nonnegative.")
        self.n_clusters, self.max_iter, self.n_init = n_clusters, max_iter, n_init
        self.tol, self.random_state = tol, random_state

    @staticmethod
    def _assign(features, centers):
        return np.argmin(np.sum((features[:, None] - centers) ** 2, axis=2), axis=1)

    def fit(self, X):
        features = _features(X)
        if self.n_clusters > len(features):
            raise ValueError("n_clusters cannot exceed the sample count.")
        rng = np.random.default_rng(self.random_state)
        best = None
        for _ in range(self.n_init):
            centers = features[rng.choice(len(features), self.n_clusters, replace=False)].copy()
            for iteration in range(self.max_iter):
                labels = self._assign(features, centers)
                updated = centers.copy()
                for cluster in range(self.n_clusters):
                    members = features[labels == cluster]
                    if len(members):
                        updated[cluster] = members.mean(axis=0)
                shift = np.linalg.norm(updated - centers)
                centers = updated
                if shift <= self.tol:
                    break
            labels = self._assign(features, centers)
            inertia = float(np.sum((features - centers[labels]) ** 2))
            if best is None or inertia < best[0]:
                best = (inertia, centers.copy(), labels.copy(), iteration + 1)
        self.inertia_, self.cluster_centers_, self.labels_, self.n_iter_ = best
        self.n_features_in_ = features.shape[1]
        return self

    def predict(self, X):
        return self._assign(self._prediction_features(X), self.cluster_centers_)
