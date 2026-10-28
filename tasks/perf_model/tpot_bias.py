"""Is the step model's decode-step error random or biased? That decides whether TPOT prediction inherits it.

A request's TPOT is the MEAN step time over the steps it decodes through. If the model's per-step error is
zero-mean and roughly independent across steps, the error of that mean shrinks as 1/sqrt(n) and a 20% per-step
MAPE becomes a few per cent on a request emitting hundreds of tokens. If the error is BIASED, no amount of
averaging removes it and the bias passes straight into predicted TPOT. So the per-step MAPE that looks alarming
(18-38% on decode steps) is the wrong statistic; the signed mean error is the right one.
"""
import sys
sys.path.insert(0, "tasks/perf_model")
import numpy as np
from chunk_predictor import StepModel, load

rows = load(sys.argv[2:])
model = StepModel(sys.argv[1])
dec = [r for r in rows if r["ptok"] == 0 and r["nd"] > 0]
pred = np.array([model.predict_ms(0.0, r["pf"], r["nd"], r["dkv"]) for r in dec])
real = np.array([r["gpu"] for r in dec])
err = (pred - real) / np.maximum(real, 1e-6)
print(f"{len(dec)} decode-only steps")
print(f"  per-step MAPE          {100*np.abs(err).mean():.1f}%  (p90 {100*np.percentile(np.abs(err),90):.1f}%)")
print(f"  per-step SIGNED bias   {100*err.mean():+.1f}%  (median {100*np.median(err):+.1f}%)")
print(f"  per-step error sd      {100*err.std():.1f}%")
# What the bias and spread imply for a request's mean over n steps, if errors were independent.
for n in (50, 200, 500):
    print(f"  implied error on a {n:>3}-token request's mean TPOT: "
          f"bias {100*err.mean():+.1f}% + noise {100*err.std()/np.sqrt(n):.1f}%")

# Calibration. A systematic bias with a stable median is a multiplicative constant, not a modelling failure, so
# fit one factor on a TRAINING half and report the residual on the held-out half. Fitting on all the data and
# reporting the resulting bias as zero would be circular -- the factor is defined to make it zero.
# The split is chronological (the trace is time-sorted), which is the honest one for a model that will be used
# online on later steps than it was fitted on.
cut = len(dec) // 2
k = float(np.median(real[:cut] / np.maximum(pred[:cut], 1e-6)))
print(f"\ncalibration: one multiplicative factor fitted on the first {cut} steps -> x{k:.4f}")
for name, lo, hi in (("train", 0, cut), ("HELD OUT", cut, len(dec))):
    e2 = (pred[lo:hi] * k - real[lo:hi]) / np.maximum(real[lo:hi], 1e-6)
    print(f"  {name:>8}: bias {100*e2.mean():+.1f}%  median {100*np.median(e2):+.1f}%  "
          f"MAPE {100*np.abs(e2).mean():.1f}%  sd {100*e2.std():.1f}%")
    if name == "HELD OUT":
        print(f"            implied error on a 500-token request's mean TPOT: "
              f"bias {100*e2.mean():+.1f}% + noise {100*e2.std()/np.sqrt(500):.1f}%")
