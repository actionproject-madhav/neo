import numpy as np

# A trained FP32 classifier, built deterministically so everyone gets the same
# model. It is a linear classifier (logits = X @ W.T + b). Two properties are
# baked in on purpose, because they are what make quantization non-trivial:
#   1. The rows of W span a wide magnitude range (~0.3x .. 20x), so one global
#      scale across the whole matrix wastes precision on the small rows.
#   2. A few sparse weight outliers per row, so a naive max-abs range wastes
#      most of the INT4 budget; clipping the range from the calibration set
#      (e.g. to a high percentile) recovers it.
# INT8 comes out near-lossless either way. INT4 is where per-channel scaling
# and calibration actually separate a clean result from a broken one.

SEED = 7
D = 24          # feature dimension
C = 12          # number of classes


def _build():
    rng = np.random.default_rng(SEED)
    base = rng.standard_normal((C, D))
    base /= np.linalg.norm(base, axis=1, keepdims=True)        # unit directions
    mags = 10.0 ** rng.uniform(-1.0, 1.3, size=(C, 1))         # wide magnitude range
    centroids = base * mags
    W = (2.0 * centroids).copy()
    # Sparse weight outliers (~4% of entries), scaled to the row's mean magnitude.
    outlier_mask = rng.random(W.shape) < 0.04
    W = W + outlier_mask * np.sign(rng.standard_normal(W.shape)) * 8.0 * np.abs(W).mean(axis=1, keepdims=True)
    b = -(centroids ** 2).sum(axis=1)
    return centroids, W.astype(np.float64), b.astype(np.float64)


CENTROIDS, W_FP32, B_FP32 = _build()

#these are hte logits of the model
def logits(X, W, b):
    return X @ W.T + b

#this does the prediction of the model
def predict(X, W, b):
    return np.argmax(logits(X, W, b), axis=1)

#this is simply the base accuracy
def accuracy(pred, y):
    return float(np.mean(pred == y))

#generate the data for model with noise
def gen_data(n, seed, noise=0.1):
    rng = np.random.default_rng(seed)
    y = rng.integers(0, C, size=n)
    mags = np.linalg.norm(CENTROIDS, axis=1, keepdims=True)    # (C, 1)
    X = CENTROIDS[y] + rng.standard_normal((n, D)) * (noise * mags[y])
    return X.astype(np.float64), y


# Provided splits. CALIB_X is the calibration set (use it to choose ranges/scales).
CALIB_X, _CALIB_Y = gen_data(300, seed=101)
EVAL_X, EVAL_Y = gen_data(1500, seed=202)

BASELINE_ACC = accuracy(predict(EVAL_X, W_FP32, B_FP32), EVAL_Y)
