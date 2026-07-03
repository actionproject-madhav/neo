# Progress

## Phase 1 — DONE
Implemented in `quantize.py`:
- `QModel` dataclass: `q_weight` (int8 array), `scale` (per-row array, stored as float16), `bits`.
- `quantize()`: symmetric, per-row. scale = row's max abs weight / 127. Weight ints = round(W / scale), clamped to [-128, 127].
- `forward_quant()`: dequantize (`q_weight * scale`), then normal `X @ W.T + b`, argmax.

Checked results:
- Baseline accuracy: 0.8980
- INT8 accuracy: 0.8987 (actually above baseline, well inside the 1% allowed drop)
- Mean dequant error: 0.0216, max: 0.096 (small vs. row magnitudes of 0.3-20)
- `q_weight` uses the full int8 range (-127..127), so scale isn't wasting precision

Tests passing: `test_int8_accuracy_retention`, `test_returns_predictions_shape`
Tests not run yet (need Phase 2): `test_int8_compression` (needs `stored_bits`)

## Phase 2 — NOT STARTED
`stored_bits()` still raises `NotImplementedError`. Need to add up weight bits + scale bits (16 bits/scale) and confirm ratio >= 3.5x.

## Phase 3 — NOT STARTED
INT4 with percentile-clipped calibration, not started.
