"""
drift_detection.py  –  ADWIN sanity-check demo
==============================================

Standalone proof that the River ADWIN drift detector is wired correctly.

This is intentionally separate from the main online-learning experiment
(online_learning.py). The main experiment uses a real learner whose
incremental SGD updates can track moderate concept drift without
triggering ADWIN — which is correct streaming-ML behavior, but means
ADWIN may not fire in that pipeline.

To verify the detector itself is configured correctly, we feed it a
synthetic error stream with a deliberate, abrupt distribution change:

    Phase A:  100 samples drawn from "low error" regime  (~10% error)
    Phase B:  100 samples drawn from "high error" regime (~60% error)

ADWIN with delta=0.05 should detect this change within ~20 samples of
the Phase B boundary. If it does, the detector integration is sound.

Reference
---------
Bifet, A., & Gavalda, R. (2007). Learning from time-changing data with
    adaptive windowing. SIAM ICDM.
"""

import random
from river import drift

# Reproducibility
random.seed(42)

# ── Configuration ────────────────────────────────────────────────────────────
_PHASE_A_SIZE = 100
_PHASE_B_SIZE = 100
_PHASE_A_ERROR_RATE = 0.10   # well-trained learner regime
_PHASE_B_ERROR_RATE = 0.60   # post-drift, untrained regime
_ADWIN_DELTA = 0.05          # same setting used in online_learning.py


def build_error_stream() -> list[int]:
    """Bernoulli error stream with an abrupt rate change at index 100."""
    stream = []
    for _ in range(_PHASE_A_SIZE):
        stream.append(1 if random.random() < _PHASE_A_ERROR_RATE else 0)
    for _ in range(_PHASE_B_SIZE):
        stream.append(1 if random.random() < _PHASE_B_ERROR_RATE else 0)
    return stream


def detect_drift(values: list[int]) -> list[int]:
    """Feed values to ADWIN, return indices where drift was detected."""
    adwin = drift.ADWIN(delta=_ADWIN_DELTA)
    drift_points: list[int] = []

    for i, value in enumerate(values):
        adwin.update(value)
        if adwin.drift_detected:
            drift_points.append(i)
            print(f"[ADWIN] Drift detected at index {i}")

    if not drift_points:
        print("[ADWIN] No drift detected.")
    return drift_points


if __name__ == "__main__":
    print("=" * 62)
    print("  ADWIN SANITY-CHECK DEMO")
    print("=" * 62)
    print(f"  Phase A: {_PHASE_A_SIZE} samples @ {_PHASE_A_ERROR_RATE:.0%} error rate")
    print(f"  Phase B: {_PHASE_B_SIZE} samples @ {_PHASE_B_ERROR_RATE:.0%} error rate")
    print(f"  ADWIN delta: {_ADWIN_DELTA}")
    print(f"  Drift boundary: index {_PHASE_A_SIZE}")
    print()

    stream = build_error_stream()
    drift_points = detect_drift(stream)

    print()
    if drift_points:
        first_detection = drift_points[0]
        delay = first_detection - _PHASE_A_SIZE
        if delay >= 0:
            print(f"  ✓ ADWIN integration verified.")
            print(f"    First detection at index {first_detection} "
                  f"({delay} samples after the true boundary)")
        else:
            print(f"  ⚠ ADWIN fired BEFORE the true boundary (index "
                  f"{first_detection} < {_PHASE_A_SIZE}); investigate.")
    else:
        print("  ✗ ADWIN did NOT fire on a 10% → 60% error rate change.")
        print("    Either the delta is mis-set or the integration is broken.")
    print("=" * 62)