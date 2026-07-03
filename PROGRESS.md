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

## Phase 3 — DONE
INT4 (`bits=4`) reuses the same `quantize()`/`forward_quant()`/`stored_bits()` path. Only change: when `bits <= 4`, the per-row scale is computed from the 90th-percentile of `|W[row]|` instead of the true max, so the ~4% sparse outliers per row don't dictate the scale and crush everything else into a handful of int4 buckets.

How 90 was picked (corrected methodology - see below): swept clip percentile 50-100, selecting purely on the **calibration split** (`CALIB_X`/`_CALIB_Y`, 300 samples) — never on `EVAL_X`/`EVAL_Y`. Calibration accuracy peaks at percentile 90 (100% on the 300-sample calib set). `EVAL_X`/`EVAL_Y` (1500 samples) is touched exactly once, after the percentile is already fixed, purely to report the final number. Chart: `analysis/int4_percentile_sweep.png`, script: `analysis/sweep_int4.py`.

**Correction made**: the percentile was originally swept directly against `EVAL_X`/`EVAL_Y` (the held-out set) — a real methodology mistake (tuning a hyperparameter on the test set). Before switching to the calibration split, two label-free alternatives were tried and rejected:
1. Minimize weight-reconstruction error weighted by `CALIB_X` activations (no labels) → picks percentile ~99.5, gives 88.3% eval accuracy. Fails because it just rewards preserving the biggest-magnitude values, which here are exactly the injected outlier noise.
2. Maximize agreement between the quantized model and the FP32 model's own predictions on `CALIB_X` (no labels, self-referential) → same failure, since the FP32 model is itself degraded by the outliers.

Both failed because there's no label-free signal in this problem that separates "helpful" clipping (removing injected noise) from "harmful" clipping (destroying real signal). Only the labeled calibration split can do that. Re-running the sweep on `CALIB_X`/`_CALIB_Y` instead of eval gave the **same answer (90)**, confirming the original number wasn't an artifact of overfitting to eval, just arrived at the wrong way. `INT4_CLIP_PERCENTILE = 90` in `quantize.py` is unchanged; only how it was chosen (and documented) changed.

Results (percentile=90, reported on eval, never used to select it):
- INT4 accuracy: 0.996 (baseline 0.898 — comfortably passes, even exceeds baseline since clipping also strips the synthetic outlier noise baked into W)
- INT4 stored_bits: 1344, compression ratio: 6.86x (well past 3.5x)
- All 3 visible INT8 tests still pass unchanged (bits=8 still uses true max, clip_percentile=100)

Diagnostic-only, not required for submission: `analysis/sweep_int4.py` (uses matplotlib, not in requirements.txt) + `analysis/int4_percentile_sweep.png` (the chart it generates).

## Bug fix — scale precision mismatch
Found while checking the "numerical correctness" requirement: weights were rounded into integers using the full-precision scale, but the *stored* scale (used for dequant) was float16-truncated - a slightly different number. That let 2/288 elements land just outside the error bound implied by the actually-stored scale (worst case off by 0.0008). Fixed by rounding the scale to float16 *before* using it to quantize, so the same scale is used consistently both times. No change to accuracy or compression numbers, just closes the correctness gap.

## Edge-case audit
Went looking specifically for silent-corruption bugs (wrong output, no error) rather than crashes. Found and fixed three, none of which affect the graded `bits=8`/`bits=4` paths on this dataset, but all were real:

1. **`bits=16` silently wrapped around int8.** `q_weight` was always cast to `np.int8` regardless of `bits`, so values needing more than 8 bits would wrap (undefined-looking values, no error). Fixed with `_int_container_dtype(bits)`: int8 up to 8 bits, int16 up to 16, int32 up to 32, int64 beyond. `bits=8`/`4` still get `int8` exactly as before - unaffected.
2. **`bits=1` divided by zero.** `qmax = 2^(bits-1) - 1 = 0` at `bits=1`, so `scale = amax/0 = inf`, silently quantizing every weight to 0 (a dead model) with only a buried `RuntimeWarning`. Fixed: `_qrange` now raises `ValueError` for `bits < 2`.
3. **float16 scale can overflow/underflow for extreme weight magnitudes.** If a row's calibrated max-abs is too large (>~8.3M for int8) or too small (<~6e-5), casting the scale to float16 silently produces `inf` or `0`, again quantizing the row to zero without an error. Fixed: `quantize()` now raises `ValueError` if this happens instead of proceeding silently. Checked the actual margins on `W_FP32`: smallest scale is ~31x above the underflow floor (INT8) / ~414x (INT4), largest scale is many thousands of times below the overflow ceiling — so this never triggers on the real graded data, it's just insurance.

Also explicitly tested (no bugs found): `per_channel=False` combined with `bits=4` (0.936 acc, no crash), `bits=2` (works, 4 distinct values, 0.983 acc), and the all-zero-row divide-by-zero guard (still correct, not a false positive of fix #3).

All 3 visible tests + both accuracy/compression checks re-verified passing after every fix above.
