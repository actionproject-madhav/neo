from dataclasses import dataclass

import numpy as np

# Scales are stored/counted at float16 precision. A plain float32 scale per
# row would only buy ~3.43x compression for this model's shape (C=12, D=24),
# just short of the 3.5x bar; float16 is plenty of precision for a single
# per-row scale factor and clears the bar comfortably.
SCALE_BITS = 16


@dataclass
class QModel:
    q_weight: np.ndarray   # integer weights, stored in an int8 container
    scale: np.ndarray      # per-channel (per-row) scales, shape (C,)
    bits: int              # bit-width the integers are meaningfully quantized to


def _qrange(bits):
    """Signed symmetric integer range for `bits`-bit quantization."""
    qmax = 2 ** (bits - 1) - 1
    qmin = -(qmax + 1)
    return qmin, qmax


def quantize(W, bits=8, per_channel=True, calib=None):
    """Quantize the weight matrix W (shape (C, D)) to `bits`-bit integers.

    Return a quantized-model object that stores INTEGER weights plus scales
    (and zero-points if you go asymmetric), not the original floats. Choosing
    scales per output channel (per row of W) is what keeps accuracy when the
    rows span different magnitudes.

    Symmetric quantization is used (no zero-point): each row's scale maps its
    max-abs weight to the top of the integer range, and dequantization is a
    simple `q * scale`.
    """
    W = np.asarray(W, dtype=np.float64)
    qmin, qmax = _qrange(bits)

    if per_channel:
        amax = np.abs(W).max(axis=1)          # (C,)
    else:
        amax = np.full(W.shape[0], np.abs(W).max())

    # Avoid divide-by-zero for an all-zero row; scale doesn't matter then
    # since the row's integer weights will all be 0 anyway.
    amax = np.where(amax == 0, 1.0, amax)
    scale = amax / qmax

    q = np.round(W / scale[:, None])
    q = np.clip(q, qmin, qmax).astype(np.int8)

    return QModel(q_weight=q, scale=scale.astype(np.float16), bits=bits)


def forward_quant(X, qmodel, b):
    """Run the classifier using the quantized weights.

    Dequantize on the fly and return predicted class indices for X (shape (N,)).
    """
    X = np.asarray(X, dtype=np.float64)
    W_deq = qmodel.q_weight.astype(np.float64) * qmodel.scale.astype(np.float64)[:, None]
    logits = X @ W_deq.T + np.asarray(b, dtype=np.float64)
    return np.argmax(logits, axis=1)


def stored_bits(qmodel):
    """Return the true number of bits used to store the quantized weights plus
    scales/zero-points. Used to compute the compression ratio against FP32."""
    weight_bits = qmodel.q_weight.size * qmodel.bits
    scale_bits = qmodel.scale.size * SCALE_BITS
    return weight_bits + scale_bits
