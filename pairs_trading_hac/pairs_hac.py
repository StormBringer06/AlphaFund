"""One-sided paired Newey-West test for the naive versus VIX-adaptive pitch.

Import paired_hac_test(naive_returns, vix_returns). Run this file directly for
a synthetic demonstration. README.md explains the mathematics and assumptions.
"""

from __future__ import annotations

import json
from numbers import Real
from typing import Iterable

import numpy as np
from scipy.stats import norm
import statsmodels.api as sm

# These are thresholds for ONE p-value, not independent confirmations.
DEFAULT_LEVELS = (0.05, 0.025, 0.01, 0.005, 0.001)


def _as_returns(values, name: str) -> np.ndarray:
    """Require finite 1D numeric returns; copy to protect the caller's data."""
    # np.asarray strips MaskedArray masks and exposes their underlying data.
    # Reject missing entries BEFORE conversion rather than silently using them.
    if np.ma.isMaskedArray(values) and np.any(np.ma.getmaskarray(values)):
        raise ValueError(f"{name} contains masked observations; repair alignment upstream.")
    raw = np.asarray(values)
    if raw.ndim != 1 or raw.dtype.kind not in "iuf":
        raise ValueError(f"{name} must be a one-dimensional real numeric array.")
    arr = raw.astype(float, copy=True)
    if not np.isfinite(arr).all():
        raise ValueError(f"{name} contains NaN or infinity; repair alignment upstream.")
    return arr


def paired_hac_test(
    naive_returns: np.ndarray,
    vix_returns: np.ndarray,
    *,
    maxlags: int | None = None,
    significance_levels: Iterable[float] = DEFAULT_LEVELS,
) -> dict:
    """Test H0: E[VIX - naive] <= 0 against H1: E[VIX - naive] > 0.

    Parameters
    ----------
    naive_returns, vix_returns : 1D NumPy arrays
        Synchronized chronological daily NET portfolio returns in decimals.
        Entry i must be the same day in both arrays. Include idle days. Both
        strategies should use the same five annually reselected pairs and a
        consistent capital convention. At least 3 nonconstant differences are
        needed computationally; this does not establish statistical adequacy.
    maxlags : int or None
        Largest daily lag in the Bartlett covariance, between 0 and T-1.
        Zero omits positive-lag dependence adjustments. None uses the library
        heuristic floor(4*(T/100)**(2/9)), not the Newey-West (1994) plug-in rule.
        Prespecify a justified primary lag before examining results.
    significance_levels : iterable of floats
        One-sided levels in (0, 0.05]. Defaults: 5%, 2.5%, 1%, 0.5%, 0.1%.
        Duplicate levels are removed; results are sorted in descending order.

    Returns
    -------
    dict
        nobs, maxlags, mean_difference (daily VIX minus naive),
        hac_standard_error, z_statistic, p_value_one_sided,
        log_p_value_one_sided, and decisions. Each decision gives alpha,
        critical_z, reject_h0, and a one-sided (1-alpha) lower_confidence_bound.
        Return estimates and bounds use the same decimal units as the inputs.

    Notes
    -----
    Fits intercept-only OLS with statsmodels HAC, Bartlett weights,
    use_correction=False and use_t=False. Thus Var(mean) = estimated long-run
    variance / T, without a degrees-of-freedom multiplier. Inference is
    asymptotic normal, not an exact Student t-test. Normal daily returns are
    unnecessary; appropriate moments, weak dependence, stability and bandwidth
    conditions remain necessary. The function cannot verify backtest integrity.
    """
    naive = _as_returns(naive_returns, "naive_returns")
    vix = _as_returns(vix_returns, "vix_returns")
    if naive.size != vix.size or naive.size < 3:
        raise ValueError("Arrays must have equal lengths and at least 3 observations.")
    n = naive.size

    # Validate settings instead of silently truncating lags or changing levels.
    if maxlags is None:
        lag = min(n - 1, int(np.floor(4 * (n / 100) ** (2 / 9))))
    else:
        if isinstance(maxlags, (bool, np.bool_)) or not isinstance(maxlags, (int, np.integer)):
            raise ValueError("maxlags must be an integer in [0, T-1].")
        lag = int(maxlags)
        if not 0 <= lag < n:
            raise ValueError("maxlags must be an integer in [0, T-1].")
    try:
        raw_levels = list(significance_levels)
    except TypeError:
        raise ValueError("significance_levels must be a nonempty iterable in (0, 0.05].") from None
    if not raw_levels or any(
        isinstance(a, (bool, np.bool_)) or not isinstance(a, Real)
        or not np.isfinite(a) or not 0 < a <= 0.05 for a in raw_levels
    ):
        raise ValueError("significance_levels must be finite numbers in (0, 0.05].")
    levels = sorted({float(a) for a in raw_levels}, reverse=True)

    # Pair observations first: our estimand is the VIX strategy's advantage.
    with np.errstate(over="ignore", invalid="ignore"):
        differences = vix - naive
    if not np.isfinite(differences).all():
        raise ValueError("Return subtraction overflowed; inspect the return units.")
    scale = float(np.max(np.abs(differences)))
    if scale == 0 or np.ptp(differences / scale) <= 32 * np.finfo(float).eps:
        raise ValueError("Differences are constant or numerically degenerate; HAC uncertainty is undefined.")

    # Positive rescaling leaves z and p unchanged but prevents avoidable
    # overflow/underflow when the covariance calculation squares residuals.
    # Convert the fitted mean and SE back to the caller's original units.
    scaled_differences = differences / scale
    # A column of ones makes the OLS coefficient the sample mean.
    # HAC changes its estimated uncertainty, not the mean estimate.
    fit = sm.OLS(scaled_differences, np.ones((n, 1)), missing="raise").fit(
        cov_type="HAC",
        cov_kwds={"maxlags": lag, "kernel": "bartlett", "use_correction": False},
        use_t=False,
    )
    mean = float(fit.params[0]) * scale
    se = float(fit.bse[0]) * scale
    if not np.isfinite(mean) or not np.isfinite(se) or se <= 0:
        raise ValueError("Nonpositive or nonfinite HAC standard error; inspect returns and lag choice.")
    z = mean / se

    # Upper tail: a NEGATIVE advantage must produce a LARGE p-value.
    # sf avoids cancellation in 1-cdf; do not blindly halve fit.pvalues[0].
    p = float(norm.sf(z))
    log_p = float(norm.logsf(z))
    if not np.isfinite([z, p, log_p]).all():
        raise ValueError("Numerical overflow during inference; inspect the return scale.")
    decisions = []
    for alpha in levels:
        critical = float(norm.isf(alpha))
        lower = mean - critical * se
        if not np.isfinite(lower):
            raise ValueError("Numerical overflow in confidence bound.")
        decisions.append({"alpha": alpha, "critical_z": critical,
                          "reject_h0": bool(z > critical),
                          "lower_confidence_bound": lower})
    return {"nobs": int(n), "maxlags": lag,
            "mean_difference": mean, "hac_standard_error": se,
            "z_statistic": z, "p_value_one_sided": p,
            "log_p_value_one_sided": log_p, "decisions": decisions}


if __name__ == "__main__":
    # Demonstration only. Replace these arrays with actual backtest returns.
    rng = np.random.default_rng(20261121)
    naive = rng.normal(0.0001, 0.004, 2520)
    vix = naive + rng.normal(0.00008, 0.0015, 2520)
    print("SYNTHETIC DEMONSTRATION ONLY: these are not your strategy results.")
    print(json.dumps(paired_hac_test(naive, vix), indent=2, allow_nan=False))
