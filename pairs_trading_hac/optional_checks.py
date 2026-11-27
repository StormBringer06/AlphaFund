"""Supporting checks kept separate from the primary mean hypothesis test.

There is no overall pass/fail decision and no selection of the smallest p-value.
Annual mean tables are descriptive. The separately invoked annual_grouped_test
is a formal robustness test with additional independence/normality assumptions.
"""

from __future__ import annotations

from datetime import date, datetime
from typing import Iterable

import numpy as np
from scipy.stats import t as student_t

from pairs_hac import DEFAULT_LEVELS, _as_returns, paired_hac_test


def lag_sensitivity(naive_returns, vix_returns, lags: Iterable[int] = (0, 1, 5, 10, 21, 63, 126, 252)) -> list[dict]:
    """Repeat the SAME mean test with prespecified daily HAC lag lengths.

    Returns each valid test result, plus a warning when lag/T >= 0.1. Such rows
    retain normal critical values; no fixed-b calibration is implemented.
    Lags >= T are listed with a skipped reason. Invalid negative, boolean or
    noninteger lags raise ValueError. This does not alter the primary result.
    """
    a, b = _as_returns(naive_returns, "naive_returns"), _as_returns(vix_returns, "vix_returns")
    if a.size != b.size or a.size < 3:
        raise ValueError("Arrays must have equal lengths and at least 3 observations.")
    try:
        requested = list(lags)
    except TypeError:
        raise ValueError("lags must be a nonempty iterable of nonnegative integers.") from None
    if not requested or any(isinstance(k, (bool, np.bool_)) or not isinstance(k, (int, np.integer)) or k < 0 for k in requested):
        raise ValueError("lags must be a nonempty iterable of nonnegative integers.")
    rows = []
    for lag in sorted(set(int(k) for k in requested)):
        if lag >= a.size:
            rows.append({"maxlags": lag, "skipped": "lag must be smaller than sample size"})
            continue
        row = paired_hac_test(a, b, maxlags=lag)
        row["warning"] = (
            "Large lag/sample ratio: normal-reference inference is a sensitivity check; no fixed-b calibration."
            if lag / a.size >= 0.1 else None
        )
        rows.append(row)
    return rows


def annual_grouped_test(naive_returns, vix_returns, dates) -> dict:
    """Optional one-sided t-test across 2 to 14 observed annual daily means.

    Each year receives equal weight. The estimate is the mean of annual DAILY
    return differences, not a mean of compounded annual returns. It can differ
    from the full-sample day-weighted mean when observed year lengths differ.

    Requires approximately independent, Gaussian, unbiased annual estimates of
    a common population mean. Their variances may differ: Ibragimov and Muller
    (2010), Section 2.1, justify conservative one-sided t critical values at 5%
    and lower for q <= 14 under independent Gaussian common-mean observations.
    This does NOT establish those assumptions for these data. Persistent regimes,
    skewed annual shocks, incomplete years and retrospective rule selection matter.
    It is not ordinary daily HAC with its reference distribution switched to t.

    Returns annual_means (coverage and means), n_years, degrees_of_freedom,
    mean_of_annual_daily_means, standard_error, t_statistic, upper-tail p and
    log-p values, decisions at DEFAULT_LEVELS and an assumption note. All effect
    estimates, SEs and bounds remain in decimal DAILY return units. No data is
    silently excluded. Dates have the same contract as annual_mean_checks.
    """
    annual_rows = annual_mean_checks(naive_returns, vix_returns, dates)
    q = len(annual_rows)
    if not 2 <= q <= 14:
        raise ValueError("annual_grouped_test supports 2 to 14 observed years for its one-sided 5% heterogeneous-variance justification.")
    means = np.array([row["mean_daily_difference"] for row in annual_rows])
    scale = float(np.max(np.abs(means)))
    if scale == 0 or np.ptp(means / scale) <= 32 * np.finfo(float).eps:
        raise ValueError("Annual means are constant or numerically degenerate; year-level uncertainty is undefined.")
    normalized = means / scale
    mean = float(np.mean(normalized)) * scale
    se = float(np.std(normalized, ddof=1) / np.sqrt(q)) * scale
    if not np.isfinite(mean) or not np.isfinite(se) or se <= 0:
        raise ValueError("Year-level standard error is nonpositive or nonfinite.")
    statistic = mean / se
    p, log_p = float(student_t.sf(statistic, q - 1)), float(student_t.logsf(statistic, q - 1))
    if not np.isfinite([statistic, p, log_p]).all():
        raise ValueError("Numerical overflow in year-level inference.")
    decisions = []
    for alpha in DEFAULT_LEVELS:
        critical = float(student_t.isf(alpha, q - 1))
        lower = mean - critical * se
        if not np.isfinite(lower):
            raise ValueError("Numerical overflow in year-level confidence bound.")
        decisions.append({"alpha": alpha, "critical_t": critical,
                          "reject_h0": bool(statistic > critical),
                          "lower_confidence_bound": lower})
    fields = ("year", "nobs", "first_observed_date", "last_observed_date", "mean_daily_difference")
    return {"n_years": q, "degrees_of_freedom": q - 1,
            "mean_of_annual_daily_means": mean, "standard_error": se,
            "t_statistic": statistic, "p_value_one_sided": p,
            "log_p_value_one_sided": log_p, "decisions": decisions,
            "annual_means": [{key: row[key] for key in fields} for row in annual_rows],
            "assumption_note": "Optional equal-year-weighted test. Requires independent, approximately Gaussian annual estimates of a common mean; unequal variances allowed for q<=14 and one-sided levels<=5%. No automatic correction for across-year dependence, skewed annual shocks or model selection. Inspect partial-year coverage."}


def annual_mean_checks(naive_returns, vix_returns, dates) -> list[dict]:
    """Report annual and leave-one-year-out DAILY mean differences only.

    dates must be matched increasing daily ISO YYYY-MM-DD strings, date objects,
    midnight datetime objects, or daily numpy.datetime64 values. Rejects times,
    timezones, numeric labels, missing dates and duplicate days. Observed dates
    need not form complete years; this function cannot check a market calendar.
    No artificial 252-day years are created. No HAC is fitted after deleting a
    year, because concatenating the remainder creates an artificial time gap.
    """
    a, b = _as_returns(naive_returns, "naive_returns"), _as_returns(vix_returns, "vix_returns")
    if np.ma.isMaskedArray(dates) and np.any(np.ma.getmaskarray(dates)):
        raise ValueError("dates contains masked observations; repair alignment upstream.")
    raw = np.asarray(dates)
    if a.size != b.size or a.size < 3 or raw.ndim != 1 or raw.size != a.size:
        raise ValueError("Return arrays and dates must have equal lengths of at least 3.")
    parsed = []
    for value in raw:
        if isinstance(value, np.datetime64):
            # Coarser-than-daily units (year, month, week) cast to their first
            # day and would otherwise compare equal, passing as daily dates.
            if np.datetime_data(value.dtype)[0] in ("Y", "M", "W", "generic"):
                raise ValueError("dates must be valid daily dates without times.")
            if np.isnat(value) or value != value.astype("datetime64[D]"):
                raise ValueError("dates must be valid daily dates without times.")
            text = str(value.astype("datetime64[D]"))
        elif isinstance(value, datetime):
            if value.tzinfo is not None or value.time() != datetime.min.time():
                raise ValueError("dates must not contain times or timezones.")
            text = value.date().isoformat()
        elif isinstance(value, date):
            text = value.isoformat()
        elif isinstance(value, (str, np.str_)):
            text = str(value)
        else:
            raise ValueError("dates must be ISO date strings or daily date objects.")
        try:
            parsed_date = date.fromisoformat(text)
            if text != parsed_date.isoformat():
                raise ValueError
        except ValueError:
            raise ValueError("dates must use daily YYYY-MM-DD format.") from None
        parsed.append(parsed_date)
    if any(parsed[i] >= parsed[i + 1] for i in range(len(parsed) - 1)):
        raise ValueError("dates must be strictly increasing and unique.")
    with np.errstate(over="ignore", invalid="ignore"):
        d = b - a
    if not np.isfinite(d).all():
        raise ValueError("Return subtraction overflowed.")
    # Mean of scaled values avoids overflowing a sum of finite inputs.
    scale = float(np.max(np.abs(d))) or 1.0
    scaled_d = d / scale
    years = np.array([day.year for day in parsed])
    rows = []
    for year in np.unique(years):
        mask = years == year
        kept = scaled_d[~mask]
        rows.append({"year": int(year), "nobs": int(mask.sum()),
                     "first_observed_date": parsed[np.flatnonzero(mask)[0]].isoformat(),
                     "last_observed_date": parsed[np.flatnonzero(mask)[-1]].isoformat(),
                     "mean_daily_difference": float(np.mean(scaled_d[mask])) * scale,
                     "mean_daily_difference_omitting_this_year": float(np.mean(kept)) * scale if kept.size else None})
    return rows
