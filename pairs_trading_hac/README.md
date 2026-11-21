# Quant pitch Newey West hypothesis test

The main script answers your agreed question: **does the VIX-adaptive pairs-trading strategy have a higher expected daily net portfolio return than the naive strategy?** It accepts two long NumPy arrays. It performs the paired mean test and reports one-sided decisions at 5%, 2.5%, 1%, 0.5% and 0.1%.

The context is your ten-year quant pitch: a naive spread Z-score policy versus VIX-dependent entry and exit conditions, applied to the **same five pairs within each year**, with annual reselection using past cointegration screening. Your original screening window was ten years; a shorter window must be specified as part of the strategy before evaluation. The hypothesis-testing script does not choose that window or backtest either strategy.

The primary test is in `pairs_hac.py`. Lag sensitivity, annual mean comparisons, leave-one-year-out means and the separately invoked annual grouped test are supporting checks in `optional_checks.py`. They do not change the primary test or generate an overall pass/fail verdict. Sharpe, Sortino, drawdown and other performance machinery are excluded from this simplified implementation, following your later instruction to focus on the hypothesis test.

The subsequent implementation audit, literature review and synthetic calibration findings are in `AUDIT_AND_LITERATURE_REVIEW.md`.

## Agreed interpretation for the quant pitch

The question discussed was: “The current plan is to use the test to assess whether one strategy is better than another, and then use performance statistics to evaluate profitability. Is this good?”

**Yes, this is a sensible structure, provided “better” is defined as higher expected daily net return for the hypothesis test.** It does not mean universally better across all risk and performance measures. Keep relative statistical performance and economic profitability as complementary assessments.

Our analysis has two distinct questions. First, the paired test assesses whether the VIX-adaptive strategy has a higher expected daily net portfolio return than the naive benchmark. Second, performance statistics assess realised profitability and risk after realistic costs, including return, Sharpe, Sortino, drawdown, leverage and turnover. The mean test does not establish superiority on every performance measure.

Both strategies could lose money while VIX significantly outperforms. VIX could also improve Sharpe or drawdown without a statistically significant increase in mean return. These outcomes answer different questions and do not contradict one another. Performance statistics describe the backtest; they do not independently establish future profitability.

**The test targets the right hypothesis, but the default result alone is not sufficient evidence for the pitch.** Its credibility depends on strategies specified before evaluation, correctly paired net returns with credible costs, a sufficiently stable return-difference process, and a standard error that captures the relevant dependence. With annual portfolio reselection, the default eight daily lags for approximately 2,520 observations may miss persistent annual effects.

Keep the daily HAC test, examine sensitivity to justified lag choices, and use the annual grouped check to assess sensitivity to year-level uncertainty when independence and approximately Gaussian annual estimates are credible. These checks strengthen the case; they do not certify the assumptions. Agreement uses the same underlying history and is not independent replication. If conclusions conflict, disclose sensitivity to assumptions about dependence instead of selecting the favourable p-value. Prespecify how these checks will inform the claim before viewing the actual results.

No actual return arrays have been supplied. Therefore the implementation can be assessed, but its calibration for the actual strategy and the strength of any outperformance claim remain unverified. A correctly implemented calculation can produce an unreliable p-value when its statistical assumptions fail.

Suggested pitch wording: “We test whether the VIX-adaptive strategy increases mean daily net returns relative to the naive benchmark, then assess its realised profitability and risk using performance statistics. We qualify statistical conclusions using lag sensitivity and year-level robustness checks, while acknowledging the limited number of annual portfolio selections.”

## Installation and use

Use Python 3.10 or later. From this folder:

```text
python -m pip install -r requirements.txt
python pairs_hac.py
python example_usage.py
python -m pytest test_pairs_hac.py -q
```

Running `pairs_hac.py` directly prints a clearly labeled synthetic demonstration. `example_usage.py` loads the included synthetic CSV and demonstrates the primary call, followed by optional checks. Neither file contains actual evidence for your strategy.

For your real arrays, the complete primary analysis is:

```python
from pairs_hac import paired_hac_test

# Both variables are your already-computed NumPy return arrays.
result = paired_hac_test(naive_returns, vix_returns)
print(result)
```

An explicit lag choice is optional:

```python
result = paired_hac_test(
    naive_returns,
    vix_returns,
    maxlags=21,  # illustrative monthly window, not a claim that 21 is optimal
    significance_levels=(0.05, 0.025, 0.01, 0.005, 0.001),
)
print(result["p_value_one_sided"])
```

Choose the primary lag on substantive grounds before looking at the p-value. Dates are unnecessary for the primary test because it uses array order; you remain responsible for correct daily matching.

## Files and functions

| Item | What it does |
| --- | --- |
| `pairs_hac.py` | Contains `paired_hac_test`, input validation, and a synthetic demonstration under the usual `if __name__ == "__main__"` guard. Importing it does not run the demonstration. |
| `paired_hac_test(...)` | Computes the daily paired differences, fits intercept-only OLS with Bartlett HAC covariance, calculates the upper-tail normal p-value, and reports decisions and lower confidence bounds. |
| `_as_returns(...)` | Internal helper: validates and copies a one-dimensional finite numeric array. Users need not call it. |
| `optional_checks.py` | Keeps robustness checks separate from the main inference. Import it only when needed. |
| `lag_sensitivity(...)` | Repeats the same mean test at a prespecified set of lag lengths. Returns all rows, including explicit skipped-lag records. |
| `annual_mean_checks(...)` | Uses actual dates to report each observed calendar year's daily mean advantage and the mean remaining when that year is omitted. Computes no annual p-values. |
| `annual_grouped_test(...)` | Separately invoked one-sided t-test across 2 to 14 annual daily mean estimates. Weights years equally and requires additional assumptions; never runs inside the primary HAC test. |
| `example_usage.py` | Demonstrates how to load the CSV, call the main function, interpret decimal units, and call the optional functions. |
| `example_returns_SYNTHETIC.csv` | Reproducible illustrative daily data, with columns `date`, `naive_return`, `vix_return`. Contains every weekday, including exchange holidays; this is an illustrative calendar, not actual market data. The last year is partial. |
| `test_pairs_hac.py` | Automated numerical and validation tests. Uses synthetic fixtures and an independent scalar HAC calculation. |
| `requirements.txt` | Compatible dependency ranges. NumPy handles arrays; statsmodels fits OLS/HAC; SciPy supplies normal tails and quantiles; pytest runs tests. Statsmodels installs its own transitive dependencies. |
| `requirements-tested.txt` | Exact installed package versions in the environment used for the reported verification. |
| `example_results_SYNTHETIC.json` | Saved main-test output for the included CSV, labeled as synthetic and generated by the delivered implementation. |
| `AUDIT_AND_LITERATURE_REVIEW.md` | Audit findings, fixes, literature synthesis and a qualified recommendation for the annual-dependence issue. |
| `audit_calibration_SYNTHETIC.json` | Rejection rates from stated toy data-generating processes, including Monte Carlo uncertainty. These are not actual strategy results. |

## Input requirements

| Parameter | Meaning and default |
| --- | --- |
| `naive_returns` | One-dimensional numeric array of the naive strategy's daily net portfolio returns. Required. |
| `vix_returns` | Corresponding VIX-adaptive returns in the same order. Required. |
| `maxlags` | Largest included positive daily lag, an integer from 0 through T−1. Default `None` uses `floor(4*(T/100)**(2/9))`, capped below T. |
| `significance_levels` | One-sided thresholds. Default `(0.05, 0.025, 0.01, 0.005, 0.001)`. Values must be finite numbers greater than zero and no larger than 0.05. Duplicate levels are removed and results sorted in descending order. |

Returns should be **decimals**: `0.01` means 1%, and `0.0001` is one basis point. Each observation must represent the same trading day and comparable capital convention in both portfolios. Include idle days with the actual portfolio cash/idle return. Using only VIX trade days changes the question and can create selection bias. Do not flatten five pair-level return arrays into a longer series: the observation is the aggregate portfolio's daily return.

“Net” should include commissions, spreads, slippage, borrow fees, financing and annual reselection turnover under your backtest convention. The function cannot check these costs or verify identical annual pair selection. Different leverage and market exposure can explain different mean returns; this test does not estimate risk-adjusted alpha or isolate a causal VIX effect.

The code rejects mismatched lengths, nonnumeric or multidimensional input, NaN/infinity, masked missing observations, invalid settings, fewer than three observations, and constant or numerically degenerate differences. Three is a computational minimum, not a sufficient sample for credible asymptotics. It never silently sorts, drops or imputes observations. It also checks for numerical overflow and an invalid standard error. It does not impose an arbitrary return-range rule: a consistent P&L/capital convention remains your responsibility.

## Hypothesis and mathematical implementation

For each matched trading day:

\[
d_t=r_{\mathrm{VIX},t}-r_{\mathrm{naive},t},\qquad
H_0:\mu=E[d_t]\leq0,\qquad H_1:\mu>0.
\]

We fit the intercept-only model \(d_t=\mu+u_t\). With a column of ones as the only regressor, the OLS coefficient is \(\hat\mu=\bar d\). Pairing retains the same-day covariance between strategies, which an independent two-sample comparison would discard.

For \(\hat u_t=d_t-\bar d\), the scalar Bartlett Newey–West calculation is

\[
\hat\gamma_k=\frac1T\sum_{t=k+1}^{T}\hat u_t\hat u_{t-k},
\]

\[
\hat\Omega_L=\hat\gamma_0+
2\sum_{k=1}^{L}\left(1-\frac{k}{L+1}\right)\hat\gamma_k,
\qquad SE_{\mathrm{HAC}}(\bar d)=\sqrt{\hat\Omega_L/T}.
\]

The kernel downweights longer lags and gives the positive semidefinite Newey–West construction. Its consistency relies on regularity conditions; it is not protection against arbitrary dependence. [Newey and West (1987), *Econometrica* 55, 703–708](https://www.nber.org/papers/t0055).

The actual fit uses `statsmodels.OLS(...).fit(cov_type="HAC", cov_kwds={"maxlags": L, "kernel": "bartlett", "use_correction": False}, use_t=False)`. For numerical stability, differences are divided by their maximum absolute value before the fit; the fitted mean and SE are then restored to the original units. This positive scaling does not alter the statistic or p-value. There is no degrees-of-freedom multiplier. This matches the formula above; a T/(T−1) multiplier would be another finite-sample convention and would not solve the small-number-of-years problem. [Statsmodels covariance source](https://www.statsmodels.org/stable/_modules/statsmodels/stats/sandwich_covariance.html).

At the zero-mean boundary of the null:

\[
z=\frac{\bar d}{SE_{\mathrm{HAC}}(\bar d)},\qquad
p=1-\Phi(z).
\]

The reference is **asymptotic standard normal**, despite the common phrase “Newey–West t-statistic.” This is not an exact Student t-test with T−1 degrees of freedom. Under the maintained asymptotic location approximation, negative null means move the statistic further into the non-rejection region.

The script calculates `norm.sf(z)`, the upper normal tail, and `norm.logsf(z)` for very small tails. Statsmodels' standard coefficient p-value is two-sided. Blindly dividing it by two would be wrong when the observed mean advantage is negative. The explicit `use_t=False` makes the intended reference unambiguous. [Statsmodels robust inference documentation](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html).

## Every output field

| Field | Interpretation |
| --- | --- |
| `nobs` | T, the number of matched daily observations. This is not the number of independent annual portfolio selections. |
| `maxlags` | L, the actual daily lag used. It records the heuristic result when the argument was `None`. |
| `mean_difference` | Sample daily VIX-minus-naive advantage. Multiply by 10,000 for basis points per day. |
| `hac_standard_error` | Estimated uncertainty of the sample mean, in daily return units. It is not the daily return standard deviation. |
| `z_statistic` | Mean advantage divided by its HAC standard error. Larger positive values favour VIX outperformance. |
| `p_value_one_sided` | Upper-tail asymptotic p-value at the zero-mean boundary. It is not the probability that H0 is true. |
| `log_p_value_one_sided` | Natural logarithm of that p-value, useful if ordinary floating-point p underflows to zero. |
| `decisions` | A list of results for the requested significance levels; all use the same statistic and p-value. |
| `decisions[i]["alpha"]` | The nominal one-sided significance level. |
| `decisions[i]["critical_z"]` | `norm.isf(alpha)`, the upper normal critical value. |
| `decisions[i]["reject_h0"]` | Boolean decision using `z > critical_z`, equivalent to `p < alpha` apart from rounding. Equality is treated as non-rejection. |
| `decisions[i]["lower_confidence_bound"]` | `mean_difference - critical_z * hac_standard_error`. The one-sided (1−alpha) confidence set runs from this value to infinity. A positive lower bound agrees with rejection. |

| One-sided significance | Critical z |
| --- | ---: |
| 5% | 1.644854 |
| 2.5% | 1.959964 |
| 1% | 2.326348 |
| 0.5% | 2.575829 |
| 0.1% | 3.090232 |

These are nested thresholds for one result, not five replications. Displaying them does not itself create a five-test multiplicity problem. Fix the primary 5% criterion and direction before viewing outcomes. Lower alpha reduces nominal false positives under valid assumptions and also reduces power; it does not repair bias or a poor asymptotic approximation. Non-rejection is inconclusive, not proof of no advantage.

## Optional supporting checks

```python
from optional_checks import lag_sensitivity, annual_mean_checks

lag_rows = lag_sensitivity(naive_returns, vix_returns)
year_rows = annual_mean_checks(naive_returns, vix_returns, dates)
```

`lag_sensitivity` takes the same two arrays and an optional `lags` iterable, defaulting to `(0, 1, 5, 10, 21, 63, 126, 252)`. Each valid row contains all the main-test fields plus `warning`. Lag zero adjusts for heteroskedasticity without positive-lag autocovariances. Longer lags can increase or decrease the SE; negative autocovariances matter too. Lags at least T return `maxlags` and a `skipped` explanation. Invalid negative/noninteger/boolean lags raise an error. Report the full table; never pick the smallest p-value.

Rows with L/T at least 0.1 receive a warning that ordinary normal critical values may be unreliable. Ten percent is a reporting heuristic, not a validity theorem. Fixed-b theory treats bandwidth as a nonvanishing sample fraction and yields nonstandard reference distributions. This implementation does not fit fixed-b critical values. [Kiefer and Vogelsang (2005), *Econometric Theory* 21, 1130–1164](https://users.ssc.wisc.edu/~behansen/718/KieferVogelsang.pdf).

`annual_mean_checks` also requires aligned, unique increasing `dates`, as ISO `YYYY-MM-DD` strings, daily NumPy datetime64 values, date objects, or midnight datetime objects without timezone. It rejects missing dates, numeric labels, duplicate days, times and timezones. It does not check exchange holidays or completeness. Each row returns:

- `year` and `nobs`: the calendar year and number of observed days.
- `first_observed_date` and `last_observed_date`: the coverage, so partial years remain visible.
- `mean_daily_difference`: that year's daily mean advantage.
- `mean_daily_difference_omitting_this_year`: the mean over all remaining observations, or `None` if no observations remain.

These are daily arithmetic means, not compounded annual returns. `annual_mean_checks` computes no annual t-test or p-values. A sign reversal after omitting one year reveals concentration. HAC is not refitted to a series formed by removing an interior year, because concatenation would create false daily adjacency. Calendar years also need not match your selection cohorts if reselection occurs on another schedule. These descriptive checks neither prove assumptions nor correct the primary p-value.

## Optional annual grouped test

```python
from optional_checks import annual_grouped_test
year_test = annual_grouped_test(naive_returns, vix_returns, dates)
```

This separate check uses each observed year's daily mean difference as one estimate. With q years, it computes the equal-year mean, its SE `sample_std(annual_means)/sqrt(q)`, and a t-statistic with q−1 reference degrees of freedom. The upper Student t tail gives the one-sided p-value. It uses the same five default significance levels. It does not merely change the daily HAC statistic's reference distribution.

The input arrays and dates follow the contracts above. The check supports 2 to 14 observed years: under independent Gaussian estimates of a common mean, the usual one-sided t-test at 5% and below is conservative even with unequal group variances in this range. Approximate Gaussianity and independence make the empirical justification approximate. This is specifically the one-sided result in Section 2.1, rather than an unqualified extrapolation of a two-sided theorem. [Ibragimov and Müller (2010), *Journal of Business & Economic Statistics* 28, 453–468](https://www.princeton.edu/~umueller/tstat.pdf).

Output fields are `n_years`, `degrees_of_freedom`, `mean_of_annual_daily_means`, `standard_error`, `t_statistic`, `p_value_one_sided`, `log_p_value_one_sided`, `annual_means`, `decisions` and `assumption_note`. `annual_means` records each year's coverage, observation count and daily mean difference. Decisions contain `alpha`, `critical_t`, `reject_h0` and the one-sided lower confidence bound. Effects, SEs and bounds remain in daily decimal return units. Constant or numerically degenerate annual means raise an error instead of producing spurious certainty.

Years receive equal weight, so the estimate can differ from the daily test's day-weighted mean when year lengths differ. It is not a test of compounded annual returns. Partial years are shown and never silently dropped; decide the evaluation coverage before testing. A long within-year sample does not make a shared skewed annual shock Gaussian. Overlapping screening histories and persistent market regimes may also violate independence between years. This check addresses a particular uncertainty structure; it is not a universal repair for the daily result.

## Strengths for our quant pitch

The test targets the incremental return question directly, uses matched days on the same annual pairs, and estimates the covariance of the actual portfolio return difference. It allows suitable heteroskedasticity and serial correlation without assuming Gaussian or symmetric daily returns. The intercept-only specification is transparent and requires no extra predictive model. Reporting an effect estimate, SE and lower bound also distinguishes economic size from a binary significance label.

The same-day pairing is especially useful because both strategies encounter the same pairs and market shocks. However, a VIX policy may alter trading frequency, leverage and exposure, so matching does not force those shocks to cancel completely. A positive mean result supports an average return comparison under the tested procedures; it does not by itself establish a volatility-timing mechanism, better Sharpe, safer tails or investability.

## Assumptions and the annual portfolio problem

**Weak dependence and bandwidth.** The mean must satisfy an appropriate time-series central limit theorem, and the HAC covariance must consistently estimate its long-run variance. A standard sufficient framework involves decaying dependence, summable relevant autocovariances, suitable moments and a positive finite long-run variance. These conditions work together; allowing some serial correlation does not allow arbitrary persistence. In conventional HAC asymptotics under general short memory, the lag window grows at an appropriate rate while remaining small relative to T. A permanently fixed lag is not universally consistent for an infinite-memory process.

Bandwidth trades omitted covariance against noisy estimation. Andrews (1991) develops kernel/bandwidth theory; Newey and West (1994) develop data-dependent automatic lag selection and report finite-sample size distortions. Our default sample-size heuristic, documented in the library source above, is **not their 1994 plug-in algorithm** and is not claimed optimal for this strategy. It gives approximately eight lags for 2,520 observations. [Andrews (1991), author institution working paper](https://cowles.yale.edu/node/139104), [Newey and West (1994), *Review of Economic Studies* 61, 631–653](https://doi.org/10.2307/2297912).

**Persistent annual effects.** This is the main concern specific to your pitch. Consider the illustrative model

\[
d_{y,t}=\mu+a_y+e_{y,t}.
\]

If the selected annual portfolio gives the VIX policy a favourable or unfavourable environment for the whole year, days months apart share an effect. A short lag window can miss it and substantially underestimate uncertainty. As an applied illustration, with ten independent equal-length years, m days per year, independent annual shocks of variance \(\sigma_a^2\) and independent daily noise of variance \(\sigma_e^2\), direct variance decomposition gives

\[
\operatorname{Var}(\bar d)=\sigma_a^2/10+\sigma_e^2/(10m).
\]

More daily observations reduce the second component, but do not manufacture more than ten draws of the annual effect. Actual annual effects can also be dependent through persistent regimes and overlapping formation windows. A 252-lag sensitivity row helps reveal fragility; Bartlett downweighting, long-bandwidth estimation error and few annual environments prevent treating it as an automatic fix.

**Moments and non-normality.** Individual daily returns need not be normal. That does not make heavy tails harmless. A typical dependent-data CLT needs suitable moments beyond finite variance under additional dependence restrictions; covariance-estimator consistency can need stronger conditions. Finite variance alone is no blanket guarantee. Skewness, rare convergence failures and crisis losses can impair finite-sample normal approximation, especially at 0.1%. Infinite-variance or long-memory processes can invalidate this framework.

**Stability.** The strategy-selection/trading procedure must have a meaningful sufficiently stable average advantage and long-run variance. Stationarity with appropriate regularity is a standard justification; general HAC results can allow some heterogeneity. Unrestricted structural breaks or changing the VIX policy throughout evaluation are not automatically covered. Annual reselection is part of the strategy; it does not automatically invalidate HAC, but its induced persistence and stability must be credible.

## Selection bias and backtest integrity

Using only past prices for cointegration screening can make pair selection causally implementable. It does not make the VIX rule out of sample if you chose its thresholds, holding rules, formation length, stops or leverage after looking at the evaluation history. Shortening the screening window changes the trading procedure and its estimation trade-off; it does not directly repair the mean test. Searching many candidate pairs may also produce unstable relationships. None of these issues can be diagnosed from two return arrays alone.

White (2000) studies inference for the best model encountered in a specification search. Sullivan, Timmermann and White (1999) apply data-snooping adjustment to trading rules. These address selection uncertainty, which is separate from estimating one sequence's long-run covariance. This script does not implement their corrections. A 1% threshold is no substitute for an untouched evaluation period or a valid correction covering the full search. [White (2000), *Econometrica* 68, 1097–1126](https://users.ssc.wisc.edu/~behansen/718/White2000.pdf), [Sullivan et al. (1999), *Journal of Finance* 54, 1647–1691](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00163).

For the pitch, freeze the complete selection/trading rule, use information genuinely available at each decision time, make execution and borrow assumptions credible, and retain an untouched period or a properly nested chronological evaluation. Prespecify the one-sided direction. Testing one direction only after observing its sign is another selection step.

## How much to trust the result

**We can verify the arithmetic more strongly than the empirical assumptions.** The tests establish that the delivered estimator and tail calculation match their stated definitions. No actual backtest arrays were provided, so there is no basis yet for claiming VIX outperformance or assigning a confidence rating to your eventual result.

For your ten-year design, I would treat a positive, economically meaningful mean advantage with a positive lower bound as useful evidence only when the rule was fixed before evaluation, costs are credible, gains are spread across years and reasonable bandwidth choices do not overturn the conclusion. These conditions make a result more persuasive; they cannot certify future profitability.

If significance disappears at longer horizons, one year drives the advantage, or rules were selected using the same evaluation sample, the pitch should describe the evidence as fragile or exploratory. A tiny nominal p-value does not overrule those weaknesses. The earlier chat alternated between daily HAC and annual tests too readily: annual t-tests have few observations and strong small-sample assumptions, while sign flips need appropriate joint sign symmetry. Agreement on the same history would be robustness evidence, not independent replication. No one of these methods makes ten market years an unlimited sample.

The defensible claim is evidence about **the mean daily net return difference between these two specified procedures over a credible evaluation design**. Multiplying the daily mean difference by 252 is an arithmetic annualized quantity, not a test of expected compounded annual returns. The result also does not test Sharpe, Sortino or drawdown. Statistical significance and an economically investable advantage are separate judgments.

## Literature review summary

| Source linked above | What it contributes to this implementation |
| --- | --- |
| Newey and West 1987 | Bartlett HAC construction and conditional consistency; supports the covariance calculation, not unrestricted robustness. |
| Andrews 1991 | Kernel/bandwidth choice matters; motivates explicit lag selection and sensitivity. |
| Newey and West 1994 | Genuine automatic bandwidth selection differs from a simple sample-size heuristic; finite-sample distortion can remain. |
| Kiefer and Vogelsang 2005 | Large bandwidth fractions require care with the reference distribution; motivates the warning on long-lag rows. |
| White 2000 | Testing a selected winner is a separate inferential problem; HAC does not correct it. |
| Sullivan, Timmermann and White 1999 | Makes the data-snooping issue concrete for trading-rule searches such as VIX thresholds. |
| Official statsmodels documentation and source | Confirms the implementation options, Bartlett normalization, optional correction and normal-versus-t choice. |

These sources establish the methodology and its limitations. They do not validate your VIX trading rule empirically. The context-specific judgments above apply those ideas to your annually reselected five-pair design.

## Verification

The delivered tests compare the fitted SE with an independent scalar Bartlett calculation across five lag lengths, check the uncorrected lag-zero normalization, negative-mean tail direction, known decisions at all five levels, confidence-bound agreement, invariance to common shocks/unit changes, protection of input arrays, invalid inputs and optional annual/lag checks. They run without your data or network access. The example CSV and direct script are also executed to verify the complete usage path.

Verification result after audit: **80 tests passed**, using Python 3.12.14, NumPy 2.5.5, SciPy 1.18.1 and statsmodels 0.15.0. The long-array test uses 50,000 observations; additional tests cover masked inputs, extreme positive unit rescaling, a second independent Bartlett variance identity, stable annual mean accumulation and the annual grouped test. A separate randomized check covered 138 covariance cases. Both runnable examples were exercised during the audit.

Rerun the tests after changing dependencies or the estimator. Numerical tests establish implementation correctness under their fixtures; they do not prove that your actual return process satisfies the assumptions.
