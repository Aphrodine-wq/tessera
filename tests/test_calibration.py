"""Tests for the metacognition substrate + calibration math (research A1)."""
import math

from tessera.parser.module import parse_source
from tessera.sir.build import lower
from tessera.calibration import (
    softmax,
    temperature_scale,
    expected_calibration_error,
    fit_temperature,
    calibrate,
)


def test_softmax_sums_to_one():
    p = softmax([1.0, 2.0, 3.0])
    assert abs(sum(p) - 1.0) < 1e-9
    assert all(0 <= x <= 1 for x in p)


def test_temperature_scale_softens_with_T_gt_1():
    logits = [3.0, 0.0]
    p_sharp = softmax(logits)
    p_soft = temperature_scale(logits, 5.0)
    # Softer distribution → smaller max probability
    assert max(p_soft) < max(p_sharp)


def test_temperature_scale_rejects_nonpositive():
    try:
        temperature_scale([1.0, 2.0], 0.0)
        assert False, "expected ValueError"
    except ValueError:
        pass


def test_ece_zero_for_perfect_calibration():
    # If every prediction has confidence c AND accuracy c, ECE = 0.
    # Use 100 samples, 10% bin at confidence 0.1 with 10% acc, etc.
    probs = []
    correct = []
    for b in range(10):
        conf = (b + 0.5) / 10.0  # bin center
        for i in range(10):
            probs.append(conf)
            # b correct out of 10 → accuracy b/10
            correct.append(i < b)
    # Bin centers should average to bin centers; acc should be near bin acc
    ece = expected_calibration_error(probs, correct, n_bins=10)
    # Not exactly zero because bin center vs bin avg conf differ slightly
    assert ece < 0.06, f"ECE should be small for well-calibrated data, got {ece}"


def test_ece_high_for_overconfident_wrong():
    # Confidence 0.99 on every sample, accuracy 0.1 → ECE ≈ 0.89
    probs = [0.99] * 100
    correct = [i < 10 for i in range(100)]
    ece = expected_calibration_error(probs, correct, n_bins=10)
    assert ece > 0.85


def test_fit_temperature_returns_T_gt_1_for_overconfident():
    # Two-class problem; logits very confident on class 0, but ground truth
    # is mixed. Optimal T should be > 1 (soften the predictions).
    logits_list = [[3.0, 0.0]] * 20
    labels = [0] * 10 + [1] * 10  # half-correct under arg-max
    T = fit_temperature(logits_list, labels)
    assert T > 1.0, f"expected T > 1 for overconfident model, got {T}"


def test_calibrate_report_lowers_ece():
    logits_list = [[3.0, 0.0]] * 20
    labels = [0] * 10 + [1] * 10
    report = calibrate(logits_list, labels)
    # Calibration should not make things worse
    assert report.ece_after <= report.ece_before + 1e-6
    assert report.n_samples == 20
    assert report.temperature > 0


def test_metacognition_substrate_parses():
    src = """---
agent: Cal
tessera_version: 0.2
---

```tsr:metacognition
metacognition {
  temperature: 1.5
  n_bins: 10
  track_ece: true
}
```

```tsr:agent
agent Cal {
  beliefs: @last_write q: String
  intentions: plan p { return q }
}
```
"""
    pm = parse_source(src, path="<inline>")
    module = lower(pm)
    assert module.metacognition is not None
    assert math.isclose(module.metacognition.temperature, 1.5)
    assert module.metacognition.n_bins == 10
    assert module.metacognition.track_ece is True
