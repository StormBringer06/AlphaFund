"""VIX floor: hold no positions while the market is unusually calm.

At each close, if VIX is below the floor every pair's target weight is set to
zero, so the book is flat from the next day until VIX closes at or above the
floor again. The decision uses only that close's VIX, and the backtester
applies targets to the next day's returns, so there is no look-ahead.

The floor is applied after the exit classifier (whose decisions are made on
the un-gated positions) and before leverage.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

VIX_FLOOR = 12.0


def apply_vix_floor(daily: pd.DataFrame, floor: float = VIX_FLOOR) -> pd.DataFrame:
    """Zero the target weights on dates whose close-of-day VIX is below ``floor``."""
    if not math.isfinite(floor) or floor < 0:
        raise ValueError("floor must be finite and non-negative.")
    missing = {"vix", "target_weight_a", "target_weight_b"}.difference(daily.columns)
    if missing:
        raise ValueError(f"Daily results are missing columns: {sorted(missing)}.")
    vix = pd.to_numeric(daily["vix"], errors="coerce")
    if not np.isfinite(vix.to_numpy(dtype=float)).all():
        raise ValueError("VIX values must be finite.")

    out = daily.copy()
    closed = vix < floor
    out["vix_gate_open"] = ~closed
    for column in ("target_weight_a", "target_weight_b", "target_signal", "signal",
                   "target_gross_exposure"):
        if column in out:
            out.loc[closed, column] = 0.0
    return out
