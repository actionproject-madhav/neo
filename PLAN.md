# Plan: Post-training Quantization (`quantize.py`)

## Design decisions locked in
- **Symmetric quantization**: no zero-point. Per-row scale `s_c = max_abs_c / qmax` where `qmax = 2^(bits-1) - 1` (127 for int8, 7 for int4). Integer weights: `clip(round(W[c] / s_c), -qmax-1, qmax)`, dtype `int8` (int4 values still stored in an `int8` container since numpy has no native int4).
- **Scale storage**: scales stored/counted as `float16` (16 bits each), not `float32` — required to clear the 3.5x compression bar (`float32` scales only get ~3.43x for this model's shape, C=12/D=24).
- **Calibration**: self-calibrate from `W`'s own per-row `|W|` distribution (max-abs for INT8; percentile clip for INT4). `calib` param accepted but optional/unused by default, since visible tests never pass it and `CALIB_X`'s shape doesn't map onto per-row weight calibration.
- **qmodel container**: a `dataclass` `QModel(q_weight: int8 array, scale: per-row array, bits: int)`.

## Phase 1 — Core INT8 path: `quantize()` + `forward_quant()`
- Per row `c`: `amax_c = max(|W[c]|)`, `scale_c = amax_c / (2**(bits-1) - 1)` (guard divide-by-zero for an all-zero row).
- Quantize: `q = clip(round(W[c] / scale_c), -(2**(bits-1)), 2**(bits-1)-1).astype(np.int8)`.
- `forward_quant`: dequantize `W_deq = q_weight * scale[:, None]`, `logits = X @ W_deq.T + b`, `argmax`.
- Validate: `test_int8_accuracy_retention`, `test_returns_predictions_shape` pass.

## Phase 2 — `stored_bits()` + compression validation
- `weight_bits = q_weight.size * bits`, `scale_bits = scale.size * 16`, return the sum.
- Validate: `test_int8_compression` passes (ratio >= 3.5x).
- Run full visible pytest suite.

## Phase 3 — INT4 extra credit (percentile calibration)
- Extend `quantize()` for `bits=4`: clip per-row range to a high percentile of `|W[c]|` (e.g. 99th–99.5th) instead of plain max-abs, so sparse per-row outliers don't blow out the 4-bit budget.
- Reuse `forward_quant()` / `stored_bits()` unchanged.
- Validate INT4 accuracy vs `BASELINE_ACC` manually (no visible test), tune percentile if needed. Don't regress INT8.

## Validation
- `pytest squeeze/tests/test_quantize.py -v` after each phase.
- Confirm no other required helper modules are missing per README submission checklist.
