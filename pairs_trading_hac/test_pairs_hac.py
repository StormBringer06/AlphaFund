"""Independent numerical checks, input validation and optional-check tests.

Run with: python -m pytest test_pairs_hac.py -q
Tests do not need your actual backtest data and never access the network.
"""

from datetime import datetime, timezone
import json

import numpy as np
import pytest
from scipy.stats import norm, ttest_1samp

from pairs_hac import paired_hac_test
from optional_checks import annual_grouped_test, annual_mean_checks, lag_sensitivity


def manual_hac_se(d, lag):
    """Independent scalar Bartlett formula, not a second statsmodels fit."""
    n = len(d)
    u = d - np.mean(d)
    omega = sum(x * x for x in u) / n
    for k in range(1, lag + 1):
        gamma = sum(u[t] * u[t - k] for t in range(k, n)) / n
        omega += 2 * (1 - k / (lag + 1)) * gamma
    return np.sqrt(omega / n)


@pytest.mark.parametrize("lag", [0, 1, 5, 21, 63])
def test_matches_independent_formula(lag):
    rng = np.random.default_rng(18)
    # Non-Gaussian and changing-scale fixture checks covariance arithmetic.
    d = rng.standard_t(6, 250) * np.linspace(0.001, 0.004, 250) + 0.0002
    result = paired_hac_test(np.zeros(250), d, maxlags=lag)
    se = manual_hac_se(d, lag)
    assert result["mean_difference"] == pytest.approx(d.mean(), abs=1e-17)
    assert result["hac_standard_error"] == pytest.approx(se, rel=1e-12)
    assert result["z_statistic"] == pytest.approx(d.mean() / se)
    assert result["p_value_one_sided"] == pytest.approx(norm.sf(d.mean() / se))


def test_lag_zero_uses_explicit_uncorrected_variance():
    d = np.array([-0.01, 0.005, 0.003, -0.004, 0.012])
    result = paired_hac_test(np.zeros(5), d, maxlags=0)
    assert result["hac_standard_error"] == pytest.approx(d.std(ddof=0) / np.sqrt(5))


def test_signed_one_sided_tail():
    rng = np.random.default_rng(4)
    a = rng.normal(0, 0.003, 500)
    b = a + rng.normal(-0.0003, 0.001, 500)
    ab, ba = paired_hac_test(a, b, maxlags=10), paired_hac_test(b, a, maxlags=10)
    assert ab["p_value_one_sided"] > 0.99
    assert ab["p_value_one_sided"] + ba["p_value_one_sided"] == pytest.approx(1)
    assert ab["z_statistic"] == pytest.approx(-ba["z_statistic"])
    assert not any(row["reject_h0"] for row in ab["decisions"])


def test_known_z_all_levels_and_bounds():
    rng = np.random.default_rng(5)
    d = rng.normal(0, 0.001, 800)
    d -= d.mean()
    d += 2.2 * d.std(ddof=0) / np.sqrt(len(d))
    result = paired_hac_test(np.zeros(len(d)), d, maxlags=0)
    assert result["z_statistic"] == pytest.approx(2.2)
    assert result["p_value_one_sided"] == pytest.approx(0.013903447513498595)
    assert [row["reject_h0"] for row in result["decisions"]] == [True, True, False, False, False]
    for row in result["decisions"]:
        assert row["reject_h0"] == (row["lower_confidence_bound"] > 0)
        assert row["critical_z"] == pytest.approx(norm.isf(row["alpha"]))


def test_pairing_invariance_scaling_and_immutable_inputs():
    rng = np.random.default_rng(9)
    a = rng.normal(0, 0.003, 1000)
    b = a + rng.normal(0.0001, 0.001, 1000)
    saved_a, saved_b = a.copy(), b.copy()
    baseline = paired_hac_test(a, b)
    # A common daily shock cancels; unit scaling changes means/SE equally.
    common = rng.normal(0, 0.005, 1000)
    shocked = paired_hac_test(a + common, b + common)
    scaled = paired_hac_test(a * 100, b * 100)
    assert shocked["z_statistic"] == pytest.approx(baseline["z_statistic"])
    assert scaled["mean_difference"] == pytest.approx(baseline["mean_difference"] * 100)
    assert scaled["p_value_one_sided"] == pytest.approx(baseline["p_value_one_sided"])
    assert np.array_equal(a, saved_a) and np.array_equal(b, saved_b)
    assert baseline["maxlags"] == int(np.floor(4 * (1000 / 100) ** (2 / 9)))
    json.dumps(baseline, allow_nan=False)


@pytest.mark.parametrize("a,b", [
    ([], []), ([0, 1], [0, 1]), ([0, 0, 0], [0, 0]),
    ([[0], [0], [0]], [0, 0.01, 0.02]), ([0, np.nan, 0], [0, 0.01, 0.02]),
    ([0, np.inf, 0], [0, 0.01, 0.02]), ([0j, 0j, 0j], [0, 0.01, 0.02]),
    ([True, False, True], [0, 0.01, 0.02]), (["0", "0", "0"], [0, 0.01, 0.02]),
])
def test_bad_arrays(a, b):
    with pytest.raises(ValueError):
        paired_hac_test(np.asarray(a), np.asarray(b))


@pytest.mark.parametrize("lag", [-1, 3, 3.0, True])
def test_bad_primary_lag(lag):
    with pytest.raises(ValueError, match="maxlags"):
        paired_hac_test(np.zeros(3), np.array([0.01, 0.02, 0.03]), maxlags=lag)


@pytest.mark.parametrize("levels", [[], [0], [0.1], [-0.01], [np.nan], [True], ["0.05"], None])
def test_bad_levels(levels):
    with pytest.raises(ValueError, match="significance_levels"):
        paired_hac_test(np.zeros(3), np.array([0.01, 0.02, 0.03]), significance_levels=levels)


@pytest.mark.parametrize("constant", [0, 0.001, -0.001])
def test_degenerate_differences(constant):
    with pytest.raises(ValueError, match="degenerate"):
        paired_hac_test(np.zeros(50), np.full(50, constant))


def test_lag_sensitivity_is_separate_and_reports_skips():
    # Persistent fixture makes the impact of serial dependence unambiguous.
    d = np.repeat(np.array([-1, 1] * 20), 25) * 0.001 + 0.0001
    primary = paired_hac_test(np.zeros(len(d)), d, maxlags=0)
    rows = lag_sensitivity(np.zeros(len(d)), d, lags=[0, 20, 126, 2000])
    assert rows[1]["hac_standard_error"] > 3 * primary["hac_standard_error"]
    assert rows[2]["warning"] is not None
    assert "skipped" in rows[3]
    assert "warning" not in primary
    assert primary == paired_hac_test(np.zeros(len(d)), d, maxlags=0)
    with pytest.raises(ValueError):
        lag_sensitivity(np.zeros(len(d)), d, lags=[1.5])


@pytest.mark.parametrize("date_type", ["string", "datetime64"])
def test_annual_means_are_descriptive_and_use_actual_dates(date_type):
    a = np.array([-0.1, 0.03, 0.01, -0.02, 0.04, 0.01])
    b = np.array([-0.05, 0.02, 0.02, 0.01, -0.02, 0.03])
    dates = np.array(["2020-12-30", "2020-12-31", "2021-01-04", "2021-01-05", "2022-01-03", "2022-01-04"])
    if date_type == "datetime64":
        dates = dates.astype("datetime64[D]")
    rows = annual_mean_checks(a, b, dates)
    assert [row["nobs"] for row in rows] == [2, 2, 2]
    for j, row in enumerate(rows):
        mask = np.arange(6) // 2 == j
        assert row["mean_daily_difference"] == pytest.approx((b - a)[mask].mean())
        assert row["mean_daily_difference_omitting_this_year"] == pytest.approx((b - a)[~mask].mean())
        assert "p_value_one_sided" not in row


@pytest.mark.parametrize("dates", [
    ["2020-01-02", "2020-01-01", "2020-01-03"],
    ["2020-01-01", "2020-01-01", "2020-01-03"],
    ["2020-01-01", None, "2020-01-03"], ["2020-01-01", "2020-01-02"],
    [1, 2, 3], ["bad", "worse", "invalid"],
    ["2020-01-01T12:00", "2020-01-02T12:00", "2020-01-03T12:00"],
    np.array(["2020-01-01", "NaT", "2020-01-03"], dtype="datetime64[D]"),
    # Coarser-than-daily units must not pass as daily dates.
    np.array(["2020-01", "2020-02", "2020-03"], dtype="datetime64[M]"),
    np.array(["2020-01-02", "2020-01-09", "2020-01-16"], dtype="datetime64[D]").astype("datetime64[W]"),
    np.array(["2020", "2021", "2022"], dtype="datetime64[Y]"),
    [datetime(2020, 1, j, tzinfo=timezone.utc) for j in [1, 2, 3]],
])
def test_bad_dates(dates):
    with pytest.raises(ValueError):
        annual_mean_checks(np.zeros(3), np.array([0.01, 0.02, 0.03]), dates)


def test_single_year_check_cannot_omit_the_only_year():
    rows = annual_mean_checks(np.zeros(3), np.array([0.01, 0.02, 0.03]),
                              ["2020-01-01", "2020-01-02", "2020-01-03"])
    assert rows[0]["mean_daily_difference_omitting_this_year"] is None


def test_long_numpy_arrays():
    rng = np.random.default_rng(71)
    naive = rng.normal(0, 0.004, 50_000)
    vix = naive + rng.normal(0.00002, 0.001, len(naive))
    result = paired_hac_test(naive, vix)
    assert result["nobs"] == 50_000
    assert 0 <= result["p_value_one_sided"] <= 1
    assert result["hac_standard_error"] > 0


def test_extreme_upper_tail_keeps_log_p_value():
    d = 0.01 + np.tile([-0.00001, 0.00001], 100)
    result = paired_hac_test(np.zeros(len(d)), d, maxlags=0)
    assert result["p_value_one_sided"] == 0.0  # floating-point underflow, not literal certainty
    assert np.isfinite(result["log_p_value_one_sided"])
    assert result["log_p_value_one_sided"] < -1000


@pytest.mark.parametrize("lag", [0, 1, 4, 16])
@pytest.mark.parametrize("pattern", ["alternating", "persistent", "skewed"])
def test_bartlett_matches_independent_padded_moving_sum_identity(lag, pattern):
    # Bartlett's PSD identity is a second independent covariance construction:
    # zero-padded rolling sums squared, divided by (L+1)*T^2. It neither loops
    # over autocovariances nor calls another statsmodels covariance estimator.
    if pattern == "alternating":
        d = np.tile([-0.003, 0.003], 9)[:17]
    elif pattern == "persistent":
        d = np.linspace(-0.004, 0.005, 17)
    else:
        d = np.r_[np.full(15, -0.001), 0.003, 0.02]
    residual = d - d.mean()
    padded_sums = np.convolve(residual, np.ones(lag + 1), mode="full")
    expected_variance = np.sum(padded_sums ** 2) / ((lag + 1) * len(d) ** 2)
    result = paired_hac_test(np.zeros(len(d)), d, maxlags=lag)
    assert result["hac_standard_error"] ** 2 == pytest.approx(expected_variance, rel=1e-11)


@pytest.mark.parametrize("which", ["naive", "vix"])
def test_masked_returns_are_rejected_before_mask_is_lost(which):
    a = np.array([0.001, 0.002, -0.003, 0.004])
    b = np.array([0.002, -0.001, 0.004, 0.003])
    masked = np.ma.array(a if which == "naive" else b, mask=[False, True, False, False])
    if which == "naive":
        a = masked
    else:
        b = masked
    with pytest.raises(ValueError, match="masked"):
        paired_hac_test(a, b)


def test_masked_dates_are_rejected():
    dates = np.ma.array(["2020-01-01", "2020-01-02", "2020-01-03"], mask=[False, True, False])
    with pytest.raises(ValueError, match="masked"):
        annual_mean_checks(np.zeros(3), np.array([0.001, -0.002, 0.003]), dates)


def test_no_missing_values_in_masked_array_is_acceptable():
    a = np.ma.array([0.001, -0.002, 0.003], mask=False)
    b = np.array([-0.001, 0.002, 0.004])
    assert paired_hac_test(a, b) == paired_hac_test(np.asarray(a), b)


def test_zero_mean_has_half_upper_tail_and_no_rejections():
    d = np.array([-0.002, -0.001, 0.001, 0.002])
    result = paired_hac_test(np.zeros(len(d)), d, maxlags=0)
    assert result["mean_difference"] == pytest.approx(0, abs=1e-18)
    assert result["p_value_one_sided"] == pytest.approx(0.5)
    assert not any(row["reject_h0"] for row in result["decisions"])


@pytest.mark.parametrize("exponent", [-200, -150, 150, 200])
def test_covariance_is_stable_under_extreme_positive_unit_rescaling(exponent):
    d = np.array([-0.003, 0.002, -0.001, 0.005])
    baseline = paired_hac_test(np.zeros(len(d)), d, maxlags=1)
    factor = 10.0 ** exponent
    result = paired_hac_test(np.zeros(len(d)), d * factor, maxlags=1)
    assert result["z_statistic"] == pytest.approx(baseline["z_statistic"], rel=1e-12)
    assert result["p_value_one_sided"] == pytest.approx(baseline["p_value_one_sided"])
    assert result["mean_difference"] / factor == pytest.approx(baseline["mean_difference"])
    assert result["hac_standard_error"] / factor == pytest.approx(baseline["hac_standard_error"])


def test_annual_finite_inputs_do_not_overflow_mean_accumulation():
    rows = annual_mean_checks(np.zeros(3), np.array([1e308, 1e308, -1e308]),
                              ["2020-01-01", "2020-01-02", "2021-01-01"])
    assert rows[0]["mean_daily_difference"] == 1e308
    assert rows[1]["mean_daily_difference_omitting_this_year"] == 1e308
    json.dumps(rows, allow_nan=False)


@pytest.mark.parametrize("lags", [None, 5, [], [True], [-1]])
def test_bad_optional_lag_settings_fail_consistently(lags):
    with pytest.raises(ValueError, match="lags"):
        lag_sensitivity(np.zeros(3), np.array([0.01, -0.02, 0.03]), lags=lags)


def test_annual_grouped_test_matches_independent_scipy_t_test():
    yearly_means = np.array([-0.0002, 0.0001, 0.0005, -0.0001, 0.0003,
                             0.0002, -0.0003, 0.0004, 0.0001, 0.0002])
    dates = [f"{year}-01-{day:02d}" for year in range(2010, 2020) for day in range(1, 4)]
    differences = np.repeat(yearly_means, 3) + np.tile([-0.00001, 0, 0.00001], 10)
    result = annual_grouped_test(np.zeros(30), differences, dates)
    independent = ttest_1samp(yearly_means, 0, alternative="greater")
    assert result["n_years"] == 10 and result["degrees_of_freedom"] == 9
    assert result["t_statistic"] == pytest.approx(independent.statistic)
    assert result["p_value_one_sided"] == pytest.approx(independent.pvalue)
    for decision in result["decisions"]:
        assert decision["reject_h0"] == (result["p_value_one_sided"] < decision["alpha"])
        assert decision["reject_h0"] == (decision["lower_confidence_bound"] > 0)
    reversed_result = annual_grouped_test(differences, np.zeros(30), dates)
    assert reversed_result["p_value_one_sided"] == pytest.approx(1 - independent.pvalue)


def test_annual_grouped_test_explicitly_weights_years_equally():
    d = np.r_[np.full(3, -0.001), np.full(5, 0.003)]
    dates = [f"2020-01-{j:02d}" for j in range(1, 4)] + [f"2021-01-{j:02d}" for j in range(1, 6)]
    result = annual_grouped_test(np.zeros(8), d, dates)
    assert result["mean_of_annual_daily_means"] == pytest.approx(0.001)
    assert result["mean_of_annual_daily_means"] != pytest.approx(d.mean())


@pytest.mark.parametrize("n_years", [1, 15])
def test_annual_grouped_test_checks_its_supported_year_count(n_years):
    dates = [f"{year}-01-{day:02d}" for year in range(2000, 2000 + n_years) for day in range(1, 4)]
    d = np.arange(len(dates)) * 0.00001
    with pytest.raises(ValueError, match="2 to 14"):
        annual_grouped_test(np.zeros(len(d)), d, dates)


def test_annual_grouped_test_rejects_degenerate_annual_means():
    dates = [f"{year}-01-{day:02d}" for year in range(2010, 2013) for day in range(1, 4)]
    d = np.tile([-0.001, 0, 0.001], 3)
    with pytest.raises(ValueError, match="degenerate"):
        annual_grouped_test(np.zeros(9), d, dates)
