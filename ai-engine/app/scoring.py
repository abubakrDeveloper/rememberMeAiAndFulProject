from __future__ import annotations


def confidence_band(score: float) -> str:
    """Map a [0,1] confidence score to a coarse band label.

    Shared by attendance and reporting so the thresholds stay in one place.
    """
    if score >= 0.75:
        return "high"
    if score >= 0.4:
        return "medium"
    return "low"
