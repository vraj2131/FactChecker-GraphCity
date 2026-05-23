"""
Piecewise linear confidence calibration.

Maps raw confidence scores [0, 1] to calibrated probabilities [0, 1]
using configurable breakpoints.  The default breakpoints enforce:

- Floor at 0.05  — never claim absolute zero confidence
- Ceiling at 0.95 — never claim absolute certainty
- S-curve-like compression at the extremes

If we later want to train on FEVER dev-set predictions vs ground truth,
we can replace the breakpoints with learned values — the interface stays
the same.
"""

from typing import List, Tuple

from backend.app.utils.constants import CALIBRATION_BREAKPOINTS


def calibrate(
    raw_score: float,
    breakpoints: List[Tuple[float, float]] = CALIBRATION_BREAKPOINTS,
) -> float:
    """
    Map a raw confidence score to a calibrated probability using
    piecewise linear interpolation.

    Args:
        raw_score:   Raw confidence value (will be clamped to [0, 1]).
        breakpoints: Sorted list of (raw, calibrated) tuples.

    Returns:
        Calibrated confidence in [breakpoints[0][1], breakpoints[-1][1]].
    """
    # Clamp input
    x = max(0.0, min(1.0, raw_score))

    # Edge cases: at or beyond the boundary breakpoints
    if x <= breakpoints[0][0]:
        return breakpoints[0][1]
    if x >= breakpoints[-1][0]:
        return breakpoints[-1][1]

    # Walk segments and interpolate
    for i in range(len(breakpoints) - 1):
        x0, y0 = breakpoints[i]
        x1, y1 = breakpoints[i + 1]
        if x0 <= x <= x1:
            return _interpolate(x, x0, y0, x1, y1)

    # Fallback (should never reach here with valid breakpoints)
    return breakpoints[-1][1]


def _interpolate(
    x: float, x0: float, y0: float, x1: float, y1: float
) -> float:
    """Linear interpolation between two points."""
    if x1 == x0:
        return y0
    t = (x - x0) / (x1 - x0)
    return y0 + t * (y1 - y0)
