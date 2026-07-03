import numpy as np

from model import W_FP32, B_FP32, EVAL_X, EVAL_Y, BASELINE_ACC, accuracy
from quantize import quantize, forward_quant, stored_bits


def test_int8_accuracy_retention():
    q = quantize(W_FP32, bits=8, per_channel=True)
    acc = accuracy(forward_quant(EVAL_X, q, B_FP32), EVAL_Y)
    assert acc >= BASELINE_ACC - 0.01, f"int8 acc {acc} vs baseline {BASELINE_ACC}"


def test_int8_compression():
    q = quantize(W_FP32, bits=8, per_channel=True)
    fp32_bits = W_FP32.size * 32
    assert stored_bits(q) <= fp32_bits / 3.5


def test_returns_predictions_shape():
    q = quantize(W_FP32, bits=8, per_channel=True)
    pred = forward_quant(EVAL_X, q, B_FP32)
    assert np.asarray(pred).shape == (EVAL_X.shape[0],)
