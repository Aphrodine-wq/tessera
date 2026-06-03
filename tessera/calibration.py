"""Calibration / metacognition substrate (decision 2026-05-28-research, A1).

Primary reference: Guo, Pleiss, Sun, Weinberger (2017). On Calibration of
Modern Neural Networks. ICML 2017. https://arxiv.org/abs/1706.04599

A `tsr:metacognition` block declares the agent's calibration policy:

    metacognition {
      temperature: 1.0
      n_bins: 15
      track_ece: true
    }

This module ships the math: Expected Calibration Error (ECE), temperature
scaling, and a bisection-based fit_temperature routine. Pure Python — no
numpy / scipy dependency — so the core stays light. A future optional
dependency could swap in a vectorized implementation when calibrating
large prediction sets.

Honest scope: ECE is one of several calibration metrics. Adaptive ECE
(Nixon et al. 2019) and Maximum Calibration Error (MCE) are useful
follow-ups. We ship ECE first because it's the standard.
"""
from __future__ import annotations

import math
from dataclasses import dataclass


def softmax(logits: list[float]) -> list[float]:
    """Numerically-stable softmax."""
    m = max(logits)
    exps = [math.exp(z - m) for z in logits]
    s = sum(exps)
    return [e / s for e in exps]


def temperature_scale(logits: list[float], T: float) -> list[float]:
    """Apply temperature T > 0 to logits before softmax.

    T > 1 softens the distribution (less confident); T < 1 sharpens it.
    Guo et al. (2017) showed that fitting a single scalar T on a held-out
    validation set is competitive with more complex calibration methods.
    """
    if T <= 0:
        raise ValueError(f"temperature must be > 0, got {T}")
    return softmax([z / T for z in logits])


def expected_calibration_error(
    probs: list[float],
    correct: list[bool],
    n_bins: int = 15,
) -> float:
    """Compute ECE per Guo et al. 2017.

    For each of n_bins equally-sized confidence bins, compute the gap
    between the average confidence in the bin and the empirical accuracy
    in the bin. ECE = sum over bins of (bin_size / total) * |gap|.

    Args:
      probs: predicted probability (max-softmax for multi-class) per sample.
             Each in [0, 1].
      correct: bool per sample — did the prediction match the label?
      n_bins: number of equal-width bins (Guo et al. use 15).

    Returns ECE in [0, 1]. Lower is better; 0 means perfect calibration.
    """
    if len(probs) != len(correct):
        raise ValueError(
            f"probs and correct must align; got {len(probs)} vs {len(correct)}"
        )
    if n_bins < 1:
        raise ValueError(f"n_bins must be >= 1, got {n_bins}")
    if not probs:
        return 0.0
    n = len(probs)
    bin_edges = [i / n_bins for i in range(n_bins + 1)]
    ece = 0.0
    for b in range(n_bins):
        lo, hi = bin_edges[b], bin_edges[b + 1]
        # Bins are [lo, hi), except the last which includes 1.0
        if b == n_bins - 1:
            in_bin = [i for i, p in enumerate(probs) if lo <= p <= hi]
        else:
            in_bin = [i for i, p in enumerate(probs) if lo <= p < hi]
        if not in_bin:
            continue
        bin_conf = sum(probs[i] for i in in_bin) / len(in_bin)
        bin_acc = sum(1.0 for i in in_bin if correct[i]) / len(in_bin)
        ece += (len(in_bin) / n) * abs(bin_conf - bin_acc)
    return ece


def _nll(logits_list: list[list[float]], labels: list[int], T: float) -> float:
    """Mean negative log likelihood under temperature T."""
    total = 0.0
    for logits, y in zip(logits_list, labels):
        probs = temperature_scale(logits, T)
        p = max(probs[y], 1e-12)
        total -= math.log(p)
    return total / len(labels)


def fit_temperature(
    logits_list: list[list[float]],
    labels: list[int],
    *,
    lo: float = 0.05,
    hi: float = 10.0,
    tol: float = 1e-3,
    max_iter: int = 50,
) -> float:
    """Fit a single temperature T that minimizes NLL on the given data.

    Uses ternary search over [lo, hi]. Convex w.r.t. T (Guo et al. 2017),
    so ternary search converges.

    Returns the fitted T. T = 1.0 means no scaling needed (already calibrated).
    """
    if not logits_list:
        return 1.0
    if len(logits_list) != len(labels):
        raise ValueError("logits_list and labels must align")
    for _ in range(max_iter):
        if hi - lo < tol:
            break
        m1 = lo + (hi - lo) / 3
        m2 = hi - (hi - lo) / 3
        if _nll(logits_list, labels, m1) < _nll(logits_list, labels, m2):
            hi = m2
        else:
            lo = m1
    return (lo + hi) / 2


@dataclass
class CalibrationReport:
    """Returned by `calibrate(agent_predictions)`."""
    n_samples: int
    ece_before: float
    ece_after: float
    temperature: float
    n_bins: int

    def to_dict(self) -> dict:
        return {
            "n_samples": self.n_samples,
            "ece_before": self.ece_before,
            "ece_after": self.ece_after,
            "temperature": self.temperature,
            "n_bins": self.n_bins,
        }


def calibrate(
    logits_list: list[list[float]],
    labels: list[int],
    *,
    n_bins: int = 15,
) -> CalibrationReport:
    """End-to-end: compute pre-scaling ECE, fit T, compute post-scaling ECE."""
    # ECE before — use the max-softmax probability + correctness against label.
    probs_before = [softmax(L) for L in logits_list]
    pred_before = [max(range(len(p)), key=lambda i: p[i]) for p in probs_before]
    correct_before = [p == y for p, y in zip(pred_before, labels)]
    max_p_before = [max(p) for p in probs_before]
    ece_before = expected_calibration_error(max_p_before, correct_before, n_bins=n_bins)

    T = fit_temperature(logits_list, labels)

    probs_after = [temperature_scale(L, T) for L in logits_list]
    pred_after = [max(range(len(p)), key=lambda i: p[i]) for p in probs_after]
    correct_after = [p == y for p, y in zip(pred_after, labels)]
    max_p_after = [max(p) for p in probs_after]
    ece_after = expected_calibration_error(max_p_after, correct_after, n_bins=n_bins)

    return CalibrationReport(
        n_samples=len(labels),
        ece_before=ece_before,
        ece_after=ece_after,
        temperature=T,
        n_bins=n_bins,
    )
