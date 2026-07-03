from dataclasses import dataclass

import numpy as np

# Scales are stored/counted at float16 precision. A plain float32 scale per
# row would only buy ~3.43x compression for this model's shape (C=12, D=24),
# just short of the 3.5x bar; float16 is plenty of precision for a single
# per-row scale factor and clears the bar comfortably.
SCALE_BITS = 16

# Below INT8, a few sparse per-row outliers would otherwise dictate the whole
# row's scale and crush the other weights into a handful of buckets. Clipping
# the calibration range to a percentile of |W| per row instead lets outliers
# saturate while giving the bulk of the weights the resolution they need.
# 90th percentile was picked by sweeping accuracy on the CALIBRATION split
# (CALIB_X/_CALIB_Y) only - never on EVAL_X/EVAL_Y, to avoid tuning a
# hyperparameter on the held-out set. See analysis/sweep_int4.py and
# analysis/int4_percentile_sweep.png: accuracy peaks around 90 and falls off
# sharply above ~95 as clipping starts eating real signal, not just outliers.
INT4_CLIP_PERCENTILE = 90


@dataclass
class QModel:
    q_weight: np.ndarray   # integer weights, stored in an int8 container
    scale: np.ndarray      # per-channel (per-row) scales, shape (C,)
    bits: int              # bit-width the integers are meaningfully quantized to


def _qrange(bits):
    """Signed symmetric integer range for `bits`-bit quantization."""
    if bits < 2:
        raise ValueError(f"bits must be >= 2 for signed symmetric quantization, got {bits}")
    qmax = 2 ** (bits - 1) - 1
    qmin = -(qmax + 1)
    return qmin, qmax


def _int_container_dtype(bits):
    """Smallest numpy signed integer dtype that can hold `bits`-bit values.

    quantize() supports bits=8 (primary) and bits=4 (extra credit); this just
    keeps larger bit-widths from silently overflowing/wrapping in a smaller
    container instead of failing loudly or storing correctly.
    """
    if bits <= 8:
        return np.int8
    if bits <= 16:
        return np.int16
    if bits <= 32:
        return np.int32
    return np.int64

#we use the percentile based approach to quantuze the weights as just using the max might lead to over fitting
#the precentile sweep is done on the calibration set and the result is tested on the evaluatuiin set
#what we are doing here is symmatric quantization and there's no zero point here
def quantize(W, bits=8, per_channel=True, calib=None):
    """Quantize the weight matrix W (shape (C, D)) to `bits`-bit integers.

    Return a quantized-model object that stores INTEGER weights plus scales
    (and zero-points if you go asymmetric), not the original floats. Choosing
    scales per output channel (per row of W) is what keeps accuracy when the
    rows span different magnitudes.

    Symmetric quantization is used (no zero-point): each row's scale maps a
    calibrated max-abs range to the top of the integer range, and
    dequantization is a simple `q * scale`.

    For low bit-widths (INT4), the calibration range is clipped to a
    percentile of |W| per row instead of the true max, so a few sparse
    outlier weights don't blow the whole row's precision budget (see
    INT4_CLIP_PERCENTILE). INT8 stays near-lossless with the true max.
    """
    W = np.asarray(W, dtype=np.float64)
    qmin, qmax = _qrange(bits)
    absW = np.abs(W)
    clip_percentile = INT4_CLIP_PERCENTILE if bits <= 4 else 100

    if clip_percentile >= 100:
        amax = absW.max(axis=1) if per_channel else np.full(W.shape[0], absW.max())
    elif per_channel:
        amax = np.percentile(absW, clip_percentile, axis=1)
    else:
        amax = np.full(W.shape[0], np.percentile(absW, clip_percentile))

    # Avoid divide-by-zero for an all-zero row; scale doesn't matter then
    # since the row's integer weights will all be 0 anyway.
    amax = np.where(amax == 0, 1.0, amax)
    # Round the scale to float16 *before* using it to quantize, not after.
    # Otherwise the integers would be rounded against a full-precision scale
    # while dequantization uses the float16-truncated one - a slightly
    # different number - which can push a few elements outside the error
    # bound implied by the scale actually stored.
    scale_f16 = (amax / qmax).astype(np.float16)
    if not np.all(np.isfinite(scale_f16)) or np.any(scale_f16 == 0):
        # float16 range is roughly 6e-5 to 65504; a row whose calibrated
        # max-abs falls outside scale_bits*qmax of that would silently
        # overflow to inf or underflow to 0, quantizing the whole row to
        # zero with no error raised. Fail loudly instead - this dataset's
        # actual weight magnitudes are ~30-400x inside the safe range (see
        # PROGRESS.md), so this should never trigger here, but it's cheap
        # insurance against silently corrupting a row of weights.
        raise ValueError(
            "per-row scale over/underflowed float16 range - weight magnitudes "
            "are outside what this quantization scheme supports"
        )
    scale = scale_f16.astype(np.float64)

    q = np.round(W / scale[:, None])
    q = np.clip(q, qmin, qmax).astype(_int_container_dtype(bits))

    return QModel(q_weight=q, scale=scale.astype(np.float16), bits=bits)

#this is just the forward pass, but now we have quantized weights instead of the original
def forward_quant(X, qmodel, b):
    """Run the classifier using the quantized weights.

    Dequantize on the fly and return predicted class indices for X (shape (N,)).
    """
    X = np.asarray(X, dtype=np.float64)
    W_deq = qmodel.q_weight.astype(np.float64) * qmodel.scale.astype(np.float64)[:, None]
    logits = X @ W_deq.T + np.asarray(b, dtype=np.float64)
    return np.argmax(logits, axis=1)

#this is the count of the
def stored_bits(qmodel):
    """Return the true number of bits used to store the quantized weights plus
    scales/zero-points. Used to compute the compression ratio against FP32."""
    weight_bits = qmodel.q_weight.size * qmodel.bits
    scale_bits = qmodel.scale.size * SCALE_BITS
    return weight_bits + scale_bits
