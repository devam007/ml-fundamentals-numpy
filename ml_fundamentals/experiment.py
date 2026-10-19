"""Run seeded comparisons against scikit-learn and write plots and JSON."""

import argparse
import json
import platform
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import sklearn
from sklearn.cluster import KMeans as ReferenceKMeans
from sklearn.datasets import make_blobs, make_classification, make_regression
from sklearn.linear_model import LinearRegression, LogisticRegression
from sklearn.metrics import accuracy_score, adjusted_rand_score, mean_squared_error
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import StandardScaler

from .models import KMeans, KNNClassifier, LinearRegressionGD, LogisticRegressionGD


def run(output: Path, seed: int = 42) -> dict:
    output.mkdir(parents=True, exist_ok=True)
    report = {
        "seed": seed,
        "python": platform.python_version(),
        "numpy": np.__version__,
        "sklearn": sklearn.__version__,
        "results": {},
    }
    features, target = make_regression(n_samples=600, n_features=4, noise=5, random_state=seed)
    train, test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, random_state=seed
    )
    scaler = StandardScaler().fit(train)
    train, test = scaler.transform(train), scaler.transform(test)
    linear = LinearRegressionGD().fit(train, y_train)
    reference = LinearRegression().fit(train, y_train)
    report["results"]["linear_regression"] = {
        "numpy_test_mse": mean_squared_error(y_test, linear.predict(test)),
        "sklearn_test_mse": mean_squared_error(y_test, reference.predict(test)),
    }
    features, target = make_classification(
        n_samples=600, n_features=6, n_redundant=0, class_sep=1.3, random_state=seed
    )
    train, test, y_train, y_test = train_test_split(
        features, target, test_size=0.25, stratify=target, random_state=seed
    )
    scaler = StandardScaler().fit(train)
    train, test = scaler.transform(train), scaler.transform(test)
    logistic = LogisticRegressionGD(n_iter=4000).fit(train, y_train)
    reference = LogisticRegression(C=1e8, max_iter=2000).fit(train, y_train)
    report["results"]["logistic_regression"] = {
        "numpy_test_accuracy": accuracy_score(y_test, logistic.predict(test)),
        "sklearn_test_accuracy": accuracy_score(y_test, reference.predict(test)),
    }
    knn = KNNClassifier(5).fit(train, y_train)
    reference_knn = KNeighborsClassifier(5).fit(train, y_train)
    report["results"]["knn"] = {
        "numpy_test_accuracy": accuracy_score(y_test, knn.predict(test)),
        "sklearn_test_accuracy": accuracy_score(y_test, reference_knn.predict(test)),
    }
    features, target = make_blobs(n_samples=300, centers=3, cluster_std=0.7, random_state=seed)
    clusters = KMeans(random_state=seed).fit(features)
    reference_clusters = ReferenceKMeans(3, n_init=10, random_state=seed).fit(features)
    report["results"]["kmeans"] = {
        "numpy_inertia": clusters.inertia_,
        "sklearn_inertia": reference_clusters.inertia_,
        "agreement_ari": adjusted_rand_score(clusters.labels_, reference_clusters.labels_),
        "note": "Clustering fit comparison on the full synthetic dataset; not a held-out score.",
    }
    figure, axes = plt.subplots(1, 3, figsize=(13, 3.5))
    axes[0].semilogy(linear.loss_history_)
    axes[0].set(title="Linear regression: training MSE", xlabel="Iteration", ylabel="MSE")
    axes[1].plot(logistic.loss_history_)
    axes[1].set(
        title="Logistic regression: training loss", xlabel="Iteration", ylabel="Cross-entropy"
    )
    axes[2].scatter(features[:, 0], features[:, 1], c=clusters.labels_, s=12, cmap="viridis")
    axes[2].scatter(*clusters.cluster_centers_.T, marker="X", c="red", s=90)
    axes[2].set(title="K-means: synthetic clusters", xlabel="Feature 1", ylabel="Feature 2")
    figure.tight_layout()
    figure.savefig(output / "learning_curves.png", dpi=150)
    plt.close(figure)
    (output / "metrics.json").write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    return report


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=Path("reports"))
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    print(json.dumps(run(args.output, args.seed), indent=2))


if __name__ == "__main__":
    main()
