"""One-off exploration script: sweep the INT4 clip-percentile and plot accuracy.
Not part of the submission - used to pick INT4_CLIP_PERCENTILE in quantize.py.

Run from anywhere with: python analysis/sweep_int4.py  (from the squeeze/ dir)
"""
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from model import W_FP32, B_FP32, EVAL_X, EVAL_Y, BASELINE_ACC, accuracy
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
    scale = clip_val / qmax
    q = np.clip(np.round(W / scale[:, None]), qmin, qmax).astype(np.int8)
    return QModel(q_weight=q, scale=scale.astype(np.float16), bits=bits)


def forward(X, qmodel, b):
    W_deq = qmodel.q_weight.astype(np.float64) * qmodel.scale.astype(np.float64)[:, None]
    logits = X @ W_deq.T + np.asarray(b, dtype=np.float64)
    return np.argmax(logits, axis=1)


percentiles = [50, 60, 70, 75, 80, 82, 85, 87, 90, 92, 94, 95, 96, 97, 98, 99, 99.5, 100]
accs = []
for p in percentiles:
    q = quantize_with_percentile(W_FP32, bits=4, clip_percentile=p)
    acc = accuracy(forward(EVAL_X, q, B_FP32), EVAL_Y)
    accs.append(acc)
    print(f"percentile={p:6.1f}  int4_acc={acc:.4f}  delta_vs_baseline={BASELINE_ACC - acc:+.4f}")

best_idx = int(np.argmax(accs))
print()
print(f"BEST percentile = {percentiles[best_idx]}  acc = {accs[best_idx]:.4f}  baseline = {BASELINE_ACC:.4f}")

plt.figure(figsize=(8, 5))
plt.plot(percentiles, accs, marker="o", label="INT4 accuracy")
plt.axhline(BASELINE_ACC, color="green", linestyle="--", label=f"FP32 baseline ({BASELINE_ACC:.3f})")
plt.axhline(BASELINE_ACC - 0.01, color="red", linestyle=":", label="1% tolerance floor")
plt.scatter([percentiles[best_idx]], [accs[best_idx]], color="red", zorder=5, label=f"best = {percentiles[best_idx]}")
plt.xlabel("Clip percentile used to compute per-row scale")
plt.ylabel("EVAL accuracy")
plt.title("INT4 accuracy vs calibration clip percentile")
plt.legend()
plt.grid(alpha=0.3)
plt.tight_layout()
plt.savefig(OUT_PATH, dpi=150)
print(f"saved {OUT_PATH}")
