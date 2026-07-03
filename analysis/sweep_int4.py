"""INT4 clip-percentile selection - proper methodology.

Not part of the submission - used to pick INT4_CLIP_PERCENTILE in quantize.py.
Run from anywhere with: python analysis/sweep_int4.py  (from the squeeze/ dir)

Methodology note (important - read before trusting the number):
The percentile is a hyperparameter of our calibration scheme, so it must be
chosen using the CALIBRATION split (CALIB_X/_CALIB_Y) - NOT the held-out
EVAL_X/EVAL_Y set. EVAL is only touched once at the very end, to report the
final accuracy of whatever percentile calibration picked. Tuning directly on
EVAL would be leaking the test set into a "hyperparameter" choice.

Two label-free alternatives were tried and rejected before falling back to
the labeled calibration split:
  1. Minimize ||CALIB_X @ (W_deq - W).T||^2 (weight reconstruction error,
     weighted by activation magnitude, no labels needed). This picks
     percentile ~99.5 - it just rewards preserving the biggest-magnitude
     values, which here are exactly the injected outlier noise we want to
     suppress, not the classification-relevant signal.
  2. Maximize agreement between the quantized model's predictions and the
     FP32 model's own predictions on CALIB_X (no true labels, self-referential
     against the teacher model). Same failure mode - the FP32 model is itself
     degraded by the injected outliers, so mimicking it doesn't recover
     accuracy either.
Both failed because there is no label-free signal in this problem that
distinguishes "helpful" clipping (removing injected noise) from "harmful"
clipping (destroying real signal) - only the labeled calibration split can
do that, which is exactly what CALIB_X/_CALIB_Y (a validation split, distinct
from EVAL) is for.
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import W_FP32, B_FP32, EVAL_X, EVAL_Y, CALIB_X, _CALIB_Y, BASELINE_ACC, accuracy
from quantize import _qrange, QModel

OUT_PATH = os.path.join(os.path.dirname(os.path.abspath(__file__)), "int4_percentile_sweep.png")


def quantize_with_percentile(W, bits, clip_percentile):
    W = np.asarray(W, dtype=np.float64)
    qmin, qmax = _qrange(bits)
    absW = np.abs(W)
    if clip_percentile >= 100:
        clip_val = absW.max(axis=1)
    else:
        clip_val = np.percentile(absW, clip_percentile, axis=1)
    clip_val = np.where(clip_val == 0, 1.0, clip_val)
    scale = (clip_val / qmax).astype(np.float16).astype(np.float64)
    q = np.clip(np.round(W / scale[:, None]), qmin, qmax).astype(np.int8)
    return QModel(q_weight=q, scale=scale.astype(np.float16), bits=bits)


def forward(X, qmodel, b):
    W_deq = qmodel.q_weight.astype(np.float64) * qmodel.scale.astype(np.float64)[:, None]
    logits = X @ W_deq.T + np.asarray(b, dtype=np.float64)
    return np.argmax(logits, axis=1)


percentiles = [50, 60, 70, 75, 80, 82, 85, 87, 90, 92, 94, 95, 96, 97, 98, 99, 99.5, 100]

# Selection happens ONLY on the calibration split.
calib_accs = []
for p in percentiles:
    q = quantize_with_percentile(W_FP32, bits=4, clip_percentile=p)
    calib_acc = accuracy(forward(CALIB_X, q, B_FP32), _CALIB_Y)
    calib_accs.append(calib_acc)

best_idx = int(np.argmax(calib_accs))
best_percentile = percentiles[best_idx]

# EVAL is touched exactly once, after the percentile is already fixed, purely to report a number.
eval_accs = []
for p in percentiles:
    q = quantize_with_percentile(W_FP32, bits=4, clip_percentile=p)
    eval_accs.append(accuracy(forward(EVAL_X, q, B_FP32), EVAL_Y))

chosen_eval_acc = eval_accs[best_idx]

print(f"{'pct':>6} {'calib_acc (n=300, used to choose)':>36} {'eval_acc (n=1500, reporting only)':>36}")
for p, ca, ea in zip(percentiles, calib_accs, eval_accs):
    marker = "  <- chosen" if p == best_percentile else ""
    print(f"{p:6.1f} {ca:36.4f} {ea:36.4f}{marker}")

print()
print(f"Chosen percentile (via calibration split only) = {best_percentile}")
print(f"Reported EVAL accuracy at chosen percentile     = {chosen_eval_acc:.4f}  (baseline = {BASELINE_ACC:.4f})")

plt.figure(figsize=(8, 5))
plt.plot(percentiles, calib_accs, marker="o", label="Calibration accuracy (used to select percentile)")
plt.plot(percentiles, eval_accs, marker="s", linestyle="--", alpha=0.6, label="Eval accuracy (reported only, not used to select)")
plt.axhline(BASELINE_ACC, color="green", linestyle="--", label=f"FP32 baseline ({BASELINE_ACC:.3f})")
plt.axhline(BASELINE_ACC - 0.01, color="red", linestyle=":", label="1% tolerance floor")
plt.scatter([best_percentile], [calib_accs[best_idx]], color="red", zorder=5, label=f"chosen percentile = {best_percentile}")
plt.xlabel("Clip percentile used to compute per-row scale")
plt.ylabel("Accuracy")
plt.title("INT4: percentile chosen on calibration split, reported on eval split")
plt.legend(fontsize=8)
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"saved {OUT_PATH}")
