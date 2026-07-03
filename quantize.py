import numpy as np


def quantize(W, bits=8, per_channel=True, calib=None):
    """Quantize the weight matrix W (shape (C, D)) to `bits`-bit integers.

    Return a quantized-model object that stores INTEGER weights plus scales
    (and zero-points if you go asymmetric), not the original floats. Choosing
    scales per output channel (per row of W) is what keeps accuracy when the
    rows span different magnitudes.
    """
    raise NotImplementedError("Implement quantize")


def forward_quant(X, qmodel, b):
    """Run the classifier using the quantized weights.

    Dequantize on the fly and return predicted class indices for X (shape (N,)).
    """
    raise NotImplementedError("Implement forward_quant")


def stored_bits(qmodel):
    """Return the true number of bits used to store the quantized weights plus
    scales/zero-points. Used to compute the compression ratio against FP32."""
    raise NotImplementedError("Implement stored_bits")
