# Explain the algorithms

## Linear regression

Prediction: `y_hat = X @ w + b`. With `m` samples, the loss is
`mean((y_hat - y)**2)`. The gradients are `2 * X.T @ (y_hat - y) / m` and
`2 * mean(y_hat - y)`. Gradient descent subtracts learning rate times gradient.
Explain why doubling the learning rate can make the loss oscillate or diverge.

## Logistic regression

Start with `z = X @ w + b`, then apply sigmoid to obtain a class-1 probability.
The loss per example is `logaddexp(0, z) - y*z`. Its gradient with respect to z
is `sigmoid(z) - y`. Explain why accuracy alone does not reveal whether the
probabilities are calibrated, and why a threshold other than 0.5 may be useful.

## KNN

Training stores examples; prediction measures distances and votes. Scaling matters
because a feature with a large numeric range can dominate Euclidean distance.
Explain how k affects sensitivity to noise, and why repeated prediction must not
consume the stored training data.

## K-means

Alternate nearest-center assignment and mean-of-members updates. Multiple starts
reduce sensitivity to initialization. Cluster numbers are arbitrary, so compare
partitions with adjusted Rand score rather than raw equality of numeric labels.
Explain why empty clusters and duplicate points require a defined behavior.

## Interview practice

1. Derive the linear gradient on paper, then verify it with finite differences.
2. Change a seed and inspect the difference without choosing your favorite test score.
3. Explain the difference between model loss and an evaluation metric.
4. Identify which arrays are training data and which are held out.
5. Add a useful feature yourself and document the before/after behavior.

The initial code and documentation were prepared with Codex assistance. Record
your subsequent independent experiments and explanations here with dates.
