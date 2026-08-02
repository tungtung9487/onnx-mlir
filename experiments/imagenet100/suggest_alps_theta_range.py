#!/usr/bin/env python3
"""Suggest an ALPS theta search range (ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN/MAX/
STEPS) for a given f32 ONNX model, purely from its Conv/Gemm/MatMul weight values.

WHY a *range* and not a single theta
------------------------------------
The build-time weight-ALPS pass companders each weight as

    y = asinh(theta * w) / gamma            (encode, then round to posit)
    w_hat = sinh(gamma * y_q) / theta        (decode)

and does a per-channel grid search over theta in [THETA_MIN, THETA_MAX], picking
the theta that minimises the noise-to-signal ratio NSR = sum((w-w_hat)^2)/sum(w^2)
(= 1/SQNR). gamma is derived automatically (the 99th-percentile of |asinh(theta*w)|
is mapped to ~1, i.e. the bulk of companded values land in posit's high-precision
region around +/-1). So the user only has to supply a theta *range* that BRACKETS
the optimum of every channel; the search finds the rest.

HOW the range follows from the weights
--------------------------------------
asinh(theta*w) ~= theta*w   for |theta*w| << 1   (linear / no companding)
             ~= sign(w)*ln(2*theta*|w|)  for |theta*w| >> 1  (log companding)
The "knee" of the curve sits at |w| = 1/theta. Companding only does something
useful when that knee lies inside the weight distribution:
  * theta so small that 1/theta > max|w|  -> every weight is in the linear region
    -> asinh is ~linear -> equivalent to plain posit (no benefit). This is the
    lower extreme we want to include.
  * theta so large that 1/theta < smallest |w| -> even small weights are deep in
    the log region -> over-compression -> NSR rises. Upper extreme.
The per-channel optimum lands near theta ~ 1/scale_c, where scale_c is that
channel's typical weight magnitude (we use RMS = sqrt(mean(w^2))). Across all
channels the optima therefore span roughly [1/max_c scale_c, 1/min_c scale_c];
we widen that by a margin factor K on both sides so the grid safely brackets
every channel's optimum, and use a geometric (log2) grid because theta acts
multiplicatively (knee at 1/theta).

Usage:
  gpt2/bin/python suggest_alps_theta_range.py model/imagenet100_mobilenetv2.onnx
  gpt2/bin/python suggest_alps_theta_range.py a.onnx b.onnx --margin 8 --step-ratio 1.05
"""
import argparse

import numpy as np
import onnx
from onnx import numpy_helper

WEIGHTED_OPS = {"Conv", "Gemm", "MatMul", "ConvTranspose"}


def collect_weight_channel_scales(model_path):
    """Return (per-channel RMS list, per-tensor summary rows, global |w| max).

    A "channel" is one output slice (axis 0) of a Conv/Gemm/MatMul weight, i.e.
    exactly the granularity the per-channel weight-ALPS pass operates on."""
    model = onnx.load(model_path)
    inits = {i.name: i for i in model.graph.initializer}
    # weight = input index 1 of a weighted op (Conv/Gemm/MatMul) that is a constant
    weight_names = []
    for node in model.graph.node:
        if node.op_type in WEIGHTED_OPS and len(node.input) >= 2:
            w = node.input[1]
            if w in inits and w not in weight_names:
                weight_names.append(w)

    scales = []          # per output-channel RMS across the whole model
    rows = []            # per-tensor summary
    gmax = 0.0
    for name in weight_names:
        arr = numpy_helper.to_array(inits[name]).astype(np.float64)
        if arr.size == 0:
            continue
        a = arr.reshape(arr.shape[0], -1)        # [out_channels, rest]
        absa = np.abs(a)
        gmax = max(gmax, float(absa.max()))
        rms_c = np.sqrt(np.mean(a * a, axis=1))  # per-channel RMS
        rms_c = rms_c[rms_c > 0]                 # drop dead channels
        scales.append(rms_c)
        rows.append((name, tuple(arr.shape), a.shape[0],
                     float(absa.min()), float(np.median(absa)), float(absa.max())))
    if not scales:
        raise SystemExit(f"No Conv/Gemm/MatMul constant weights found in {model_path}")
    return np.concatenate(scales), rows, gmax


def suggest(scales, gmax, margin, step_ratio):
    # Drop "dead" channels (weights ~0, RMS >~2 orders below the model median):
    # any theta maps them to ~0 anyway, so they must NOT inflate theta_max.
    med = float(np.median(scales))
    active = scales[scales > med * 1e-2]
    if active.size == 0:
        active = scales
    s_lo = float(np.percentile(active, 2))      # smallest *active* channel scale
    # theta_min: knee above the single biggest weight -> that weight (hence every
    # weight) is in the near-linear region -> lower "almost no companding" bound.
    theta_min = 0.1 / gmax if gmax > 0 else 1.0 / (margin * np.percentile(scales, 98))
    # theta_max: per-channel optimum ~ 1/scale; reach the smallest active channel,
    # widened by the margin factor K so the grid brackets it.
    theta_max = margin / s_lo
    # round to 1-2 significant figures for a clean, reportable range
    theta_min = float(f"{theta_min:.1g}")
    theta_max = float(f"{theta_max:.2g}")
    steps = int(np.ceil(np.log2(theta_max / theta_min) / np.log2(step_ratio))) + 1
    steps = int(np.clip(steps, 16, 400))
    return theta_min, theta_max, steps, s_lo, float(np.percentile(scales, 98)), active.size


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("models", nargs="+", help="f32 ONNX model(s)")
    ap.add_argument("--margin", type=float, default=8.0,
                    help="bracket factor K on each side of the optimum (default 8)")
    ap.add_argument("--step-ratio", type=float, default=1.05,
                    help="max ratio between adjacent grid thetas (default 1.05)")
    args = ap.parse_args()

    all_scales = []
    gmax = 0.0
    for m in args.models:
        scales, rows, mmax = collect_weight_channel_scales(m)
        gmax = max(gmax, mmax)
        all_scales.append(scales)
        print(f"\n=== {m} ===")
        print(f"  weighted tensors: {len(rows)}   output-channels total: {len(scales)}")
        print(f"  {'tensor':40s} {'shape':22s} {'ch':>5s} {'min|w|':>10s} "
              f"{'med|w|':>10s} {'max|w|':>10s}")
        for name, shape, ch, wmin, wmed, wmax in rows[:8]:
            print(f"  {name[:40]:40s} {str(shape):22s} {ch:5d} "
                  f"{wmin:10.3g} {wmed:10.3g} {wmax:10.3g}")
        if len(rows) > 8:
            print(f"  ... (+{len(rows)-8} more)")

    scales = np.concatenate(all_scales)
    theta_min, theta_max, steps, s_lo, s_hi, n_active = suggest(
        scales, gmax, args.margin, args.step_ratio)

    print("\n================ ALPS theta range recommendation ================")
    print(f"  global max|w|                    : {gmax:.4g}")
    print(f"  channel RMS  p2 / p50 / p98      : {np.percentile(scales,2):.4g} / "
          f"{np.median(scales):.4g} / {np.percentile(scales,98):.4g}")
    print(f"  active channels (RMS>med/100)    : {n_active}/{len(scales)}  "
          f"(dead/near-zero channels excluded from theta_max)")
    print(f"  theta_min = 0.1/max|w|           : {0.1/gmax:.3g}   "
          f"(biggest weight still near-linear -> ~direct posit)")
    print(f"  theta_max = K/min_active_scale   : {args.margin}/{s_lo:.3g} "
          f"= {args.margin/s_lo:.3g}   (K={args.margin})")
    print("\n  Recommended env (weight-ALPS grid search):")
    print(f"    ONNX_MLIR_POSIT_CONST_ALPS_THETA_MIN={theta_min:g}")
    print(f"    ONNX_MLIR_POSIT_CONST_ALPS_THETA_MAX={theta_max:g}")
    print(f"    ONNX_MLIR_POSIT_CONST_ALPS_THETA_STEPS={steps}")
    print("    ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_TARGET=1.0      # gamma auto-derived")
    print("    ONNX_MLIR_POSIT_CONST_ALPS_GAMMA_PERCENTILE=0.99 # map p99 companded -> ~1")
    print(f"  (geometric grid: {steps} points, adjacent ratio "
          f"{ (theta_max/theta_min) ** (1/(steps-1)):.4f})")


if __name__ == "__main__":
    main()
