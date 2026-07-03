Squeeze
Post-training quantization of a trained model, INT8 first.

Overview
The fastest way to make a model cheaper to serve is to stop storing it in 32-bit floats. Post-training quantization maps trained weights down to 8-bit (or 4-bit) integers plus a small set of scales, so the model is 4x to 8x smaller and the matmuls run on integer hardware, all without retraining. The catch is that a careless mapping quietly destroys accuracy: one global scale across a weight matrix whose rows span very different magnitudes spends all its precision on the big rows and rounds the small ones to noise. The fix is choosing scales well, usually per output channel.

You're going to quantize a provided trained classifier. INT8 is the target and it should come out nearly lossless. INT4 is extra credit on top: the mechanics are identical, only the precision budget changes.

Problem Statement
Given a trained FP32 classifier (weights provided) and a calibration set, implement post-training quantization of the weight matrix to INT8: choose scales (per channel is what makes this work), store integer weights plus scales, and run inference by dequantizing on the fly. Quantized accuracy on a held-out eval set must stay within a small tolerance of the FP32 baseline, and the stored model must actually be smaller. INT4 with the same pipeline is extra credit.

Getting Started
Prerequisites
Python 3.11+
Setup
Dependencies are installed automatically when you initialize the assessment with the Litmus CLI. You're ready to start coding.

Files in the workspace:

model.py builds the trained FP32 model deterministically: weight matrix W_FP32 of shape (C, D), bias B_FP32, a predict(X, W, b), a gen_data, an accuracy, the provided CALIB_X calibration inputs, the held-out EVAL_X / EVAL_Y, and the BASELINE_ACC. The rows of W_FP32 span a wide magnitude range on purpose.
quantize.py is your entrypoint. Implement quantize, forward_quant, and stored_bits.
tests/test_quantize.py is the visible test set: dequant round-trip and an INT8 accuracy-retention check. The grader runs a larger suite including per-channel behavior, the compression ratio, and the INT4 extra-credit path.
Requirements
Implement quantize(W, bits=8, per_channel=True, calib=None) returning a quantized model object that stores integer weights plus scales (and zero-points if you go asymmetric), not the original floats.
Implement forward_quant(X, qmodel, b) that runs the classifier using the quantized weights (dequantize on the fly) and returns predicted class indices.
INT8 is the primary target and should come out near-lossless: quantized accuracy on EVAL_X / EVAL_Y must be within 1.0% of BASELINE_ACC. Do it per channel (per row of W); the wide row magnitudes and the weight outliers are why per-channel scaling and calibrating the range from CALIB_X matter, and they matter most at INT4.
Implement stored_bits(qmodel) returning the true number of bits used to store the quantized weights plus scales. INT8 must achieve a real compression ratio of at least 3.5x versus the FP32 weights.
Numerical correctness: dequantized weights must reconstruct W_FP32 within the quantization error implied by your chosen scales. No overflow past the integer range; rounding and clamping must be correct.
INT4 is extra credit, and it is where calibration pays off: a naive max-abs range wastes the budget on the outliers, while clipping the range (for example to a high percentile of the calibration values) recovers most of the accuracy. Same quantize(W, bits=4, ...) path. A strong INT8 result matters more than a mediocre result on both.
Examples
Example 1: INT8 round-trip

import numpy as np
from model import W_FP32
from quantize import quantize

q = quantize(W_FP32, bits=8, per_channel=True)
# dequantizing should reconstruct W_FP32 closely
Example 2: Accuracy retention

from model import EVAL_X, EVAL_Y, B_FP32, BASELINE_ACC, accuracy
from quantize import quantize, forward_quant

q = quantize(W_FP32, bits=8, per_channel=True)
acc = accuracy(forward_quant(EVAL_X, q, B_FP32), EVAL_Y)
assert acc >= BASELINE_ACC - 0.01   # within 1%
Example 3: Compression

from quantize import stored_bits
fp32_bits = W_FP32.size * 32
assert stored_bits(q) <= fp32_bits / 3.5   # at least 3.5x smaller
Submission Guidelines
What to Submit
quantize.py with your implementation, plus any helper modules.
How to Submit
litmus submit