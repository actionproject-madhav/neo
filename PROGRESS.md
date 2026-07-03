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

## Phase 2 — DONE
`stored_bits()` = (bits per weight × num weights) + (16 bits × num scales).
- fp32_bits = 9216, stored_bits = 2496, ratio = 3.69x (need >= 3.5x — passes)

All 3 visible tests now pass: `test_int8_accuracy_retention`, `test_int8_compression`, `test_returns_predictions_shape`.

## Phase 3 — NOT STARTED
INT4 with percentile-clipped calibration, not started.
