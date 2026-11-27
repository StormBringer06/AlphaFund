# Implementation audit and literature review

Reviewed 21 November 2026 for the quant pitch comparing a naive spread Z-score strategy with VIX-adaptive entry and exit conditions on the same five pairs reselected annually using past cointegration screening.

## Audit conclusion

**The primary routine implements the requested one-sided paired Bartlett Newey–West mean test correctly. It is complete for that specified calculation. Its empirical calibration for the actual strategy remains unverified.** No actual backtest arrays have been provided.

The audit found input-validation and numerical edge cases, which have been fixed without changing the statistical hypothesis or adding performance machinery to the primary function. The requested daily HAC test remains the main routine. The annual grouped test is a separately invoked robustness check with additional assumptions.

## Bugs and implementation issues addressed

| Issue reproduced | Consequence | Fix and evidence |
| --- | --- | --- |
| NumPy masked return arrays lost their mask when converted with `np.asarray` | A value explicitly marked missing could silently enter the test | Detect masked observations before conversion and reject them; regression tests cover each strategy. Fully unmasked masked-array containers remain acceptable. |
| The same conversion issue affected optional date arrays | A date marked missing could be used in annual grouping | Reject masked dates before conversion; dedicated regression test. |
| Squaring extremely small or large residuals could underflow/overflow | Finite nonconstant inputs could fail solely because of their units | Fit the intercept-only regression after positive normalization, then restore the mean and SE to original units. Tests cover factors from 10^-200 to 10^200. |
| Optional annual means could overflow while summing finite inputs | Annual summaries could contain infinity | Average normalized differences and restore the units; finite-input overflow fixture now stays finite and JSON serializable. |
| A noniterable optional lag setting raised an inconsistent exception | Confusing input-error handling | Return the same explanatory ValueError used for other invalid lag settings. |
| The earlier download archive would have contained superseded files after these changes | Download and visible source could disagree | Rebuild the archive from the final files and verify archive contents against the sources. |

The extreme-scale cases are numerical edge cases, not realistic return forecasts. The masked-input issue is more material because missing financial observations are common. NumPy documents that `asarray` returns a base ndarray for subclasses, which explains why explicit validation is needed. [NumPy documentation](https://numpy.org/doc/stable/reference/generated/numpy.asarray.html).

## Mathematical and implementation checks

| Required step | Audit finding |
| --- | --- |
| Matched differences | Correct sign: VIX minus naive, one aggregate portfolio observation per day. |
| Intercept-only OLS | Correct design: a column of ones; the coefficient estimates the sample mean. Internal positive scaling is algebraically equivalent and reversed for reported estimates. |
| HAC covariance | Bartlett weights `1-k/(L+1)`, residual autocovariance normalization by T, and final division by T for mean variance are correct. |
| Small-sample factor | Explicitly disabled. This is a documented convention matching the stated formula, not a missing required step. |
| Reference distribution | Explicit asymptotic standard normal. No false claim of an exact daily Student t distribution. |
| Composite null | One-sided inference at the zero-mean boundary for H0: mean difference <= 0. |
| Tail direction | Correct upper tail. Negative mean differences produce large p-values, rather than blindly halving a two-sided p-value. |
| Numerical tails | SciPy survival and logarithmic survival functions avoid avoidable upper-tail cancellation. |
| Decisions and bounds | Correct one-sided critical values and lower bounds at 5%, 2.5%, 1%, 0.5% and 0.1%. |
| Lag default | Matches the library's sample-size heuristic. It is correctly distinguished from the data-dependent Newey–West 1994 selector. |
| Optional year calculations | Use actual supplied dates, never artificial 252-observation years. Removing a year produces descriptive means without fitting HAC across the resulting time gap. |

The installed statsmodels covariance source was inspected directly, as well as the official API documentation. SciPy documents survival, logarithmic survival and inverse-survival functions used in the implementation. [Statsmodels covariance source](https://www.statsmodels.org/stable/_modules/statsmodels/stats/sandwich_covariance.html), [statsmodels inference options](https://www.statsmodels.org/stable/generated/statsmodels.regression.linear_model.RegressionResults.get_robustcov_results.html), [SciPy normal distribution](https://docs.scipy.org/doc/scipy/reference/generated/scipy.stats.norm.html).

## Verification beyond checking the library against itself

The automated suite passes **83 tests**. Numerical checks include a direct scalar autocovariance calculation and a second independent identity: Bartlett covariance can be constructed from squared zero-padded moving sums of residuals. That second construction tests the lag weights, edge treatment and normalization without repeating an autocovariance loop or calling another statsmodels covariance routine.

A separate randomized verification covered **138 combinations** of sample size, lag and non-Gaussian observations, including maximum admissible lags. The largest relative SE discrepancy against the moving-sum identity was approximately **5.3 × 10^-15**. Tests also cover signed tails, known significance decisions, confidence bounds, common-shock cancellation, input immutability, long arrays, masked data, invalid settings and optional grouped inference against an independent SciPy t-test.

The software tests establish agreement with the specified formulas. They do not establish the assumptions for the actual return-generating process. The tested environment is recorded in `requirements-tested.txt`; other versions allowed by the dependency ranges need their own test run.

## Literature review

**Newey and West (1987), Econometrica 55, 703–708.** The foundational contribution is a positive semidefinite heteroskedasticity/autocorrelation consistent covariance estimator under regularity conditions. It supports the Bartlett construction used here. It does not turn arbitrary financial dependence into valid finite-sample inference. [Paper record and abstract](https://www.nber.org/papers/t0055).

**Andrews (1991), Econometrica 59, 817–858.** Kernel and bandwidth choices affect covariance estimation, including the bias–variance trade-off. For this implementation, the implication is to disclose lag selection and assess sensitivity. It does not imply that every kernel or lag must be implemented to have a correct Bartlett estimator. [Author institution working-paper record](https://cowles.yale.edu/node/139104).

**Newey and West (1994), Review of Economic Studies 61, 631–653.** They propose data-dependent automatic lag selection and discuss remaining size distortions. The current code implements the simpler explicitly labeled library heuristic, not that plug-in algorithm. Omitting the plug-in selector does not make a prespecified-lag Newey–West test mathematically incomplete. [Author working-paper abstract](https://www.nber.org/papers/t0144), [published article](https://doi.org/10.2307/2297912).

**Andrews and Monahan (1992), Econometrica 60, 953–966.** Prewhitening can improve covariance bias and test size in their simulations, while also introducing estimation trade-offs. It is a possible methodological extension, not a mandatory missing step or a guaranteed remedy for annual latent effects. The primary implementation deliberately remains the agreed unprewhitened Bartlett estimator. [Author institution working-paper abstract](https://cowles.yale.edu/node/145654).

**Kiefer and Vogelsang (2005), Econometric Theory 21, 1130–1164.** Their alternative asymptotics retain the bandwidth/sample fraction and produce kernel- and bandwidth-dependent reference distributions. Their discussion also identifies finite-sample overrejection under persistence. This supports treating large-lag normal-reference rows as sensitivity evidence, not calibrated fixed-b inference. [Full author manuscript](https://users.ssc.wisc.edu/~behansen/718/KieferVogelsang.pdf).

**White (2000), Econometrica 68, 1097–1126; Sullivan, Timmermann and White (1999), Journal of Finance 54, 1647–1691.** These address data snooping and model/rule searches. For your pitch, past-only pair screening does not automatically protect against retrospectively chosen VIX thresholds or formation windows. HAC covariance adjustment does not account for selecting a favourable strategy using the evaluation history. [White full paper](https://users.ssc.wisc.edu/~behansen/718/White2000.pdf), [Sullivan et al. publisher abstract](https://onlinelibrary.wiley.com/doi/10.1111/0022-1082.00163).

**Ibragimov and Müller (2010), Journal of Business & Economic Statistics 28, 453–468.** Section 2.1 explicitly supports one-sided t-tests at 5% and below for 2 to 14 independent Gaussian common-mean observations despite unequal variances. It motivates the optional check across annual mean estimates. Applying that result to annual estimates requires credible approximate Gaussianity and independence; the theorem does not establish those properties from the number of daily observations. [Full author paper](https://www.princeton.edu/~umueller/tstat.pdf).

**Cameron and Miller (2015), Journal of Human Resources 50, 317–372.** Their review explains why few-cluster inference can overreject even after familiar variance and t-reference adjustments. This is a reason not to present ordinary year-clustered standard errors, a bootstrap or a changed reference distribution as an automatic repair with ten years. [Full author paper](https://cameron.econ.ucdavis.edu/research/Cameron_Miller_JHR_2015.pdf).

The covariance equations and library behavior were checked directly. Some historical sources were accessible through author/institution abstracts and records rather than searchable full text; the conclusions attributed to those records are limited to their stated methodological contributions. The full Kiefer–Vogelsang, White, Ibragimov–Müller and Cameron–Miller texts support the more detailed inferential qualifications. This is a focused review of this test, not an exhaustive review of pairs-trading profitability.

## Synthetic calibration stress check

Each scenario contains 1,000 simulated histories of 2,520 observations with **zero population mean advantage**. Histories are not demeaned before testing. The fixed seed is 4562026. Innovations are 0.001 times Student t with seven degrees of freedom. The AR scenario uses coefficient 0.3 and a 500-observation burn-in. The annual-effect scenario adds ten independent Gaussian effects with standard deviation 0.0005, each shared by 252 observations. These are stated toy processes, not fitted models of your returns.

| Scenario | HAC L=8 | HAC L=63 | HAC L=252 | Annual grouped check |
| --- | ---: | ---: | ---: | ---: |
| Independent t7 differences | 4.5% | 4.9% | 5.7% | 3.8% |
| Short-memory AR(1) t7 differences | 5.4% | 5.2% | 5.7% | 4.0% |
| Same AR process plus persistent annual effects | 34.3% | 23.1% | 12.8% | 6.4% |

Entries are rejection frequencies at a nominal one-sided 5% level. All daily HAC rows retain normal critical values. The grouped test uses ten annual daily mean estimates and a t9 reference. Full results at all five requested significance levels and Wilson intervals for Monte Carlo uncertainty are in `audit_calibration_SYNTHETIC.json`.

For the stressed scenario, the Monte Carlo 95% interval is approximately 31.4%–37.3% at L=8, versus 5.04%–8.09% for the grouped check. The latter substantially reduces the distortion in this toy setting but is not exact 5% empirical calibration in these 1,000 histories. The AR process also links adjacent years weakly. Neither the simulation design nor these intervals establishes the actual strategy's false-positive rate. At 0.1%, 1,000 repetitions imply only one expected rejection under ideal calibration, so tail assessment is particularly imprecise.

## Addressing the annual effect problem

The annual effect is a source of uncertainty shared by many daily observations. It must be reflected in the inference rather than treated as extra independent information. Keep the daily HAC test as the specified baseline and separately invoke `annual_grouped_test` when its assumptions are credible. That check uses variation across annual mean estimates, with an equal weight per year and nine reference degrees of freedom for ten years. It can differ from the day-weighted daily estimand in unbalanced samples; both estimates must be labeled.

The grouped check is appropriate for the simulated Gaussian annual-effect mechanism much more directly than an eight-day covariance window. It is not guaranteed for skewed annual shocks, dependence between years or unstable policy means. If regimes persist across years, neither calendar grouping nor an annual bootstrap automatically solves the problem. Broader prespecified blocks may reduce dependence but leave even fewer independent units; more genuinely out-of-sample years provide stronger information.

Increasing the daily lag, applying T/(T−1), switching only the daily statistic to t9, or adding annual fixed effects does not by itself solve the missing-annual-uncertainty problem. Nor does shortening the cointegration formation window. Those changes either address another issue, change the estimand, or require further assumptions.

For the pitch, freeze the complete rules and inference plan before evaluation, report the primary mean/SE/p-value and economic effect size, disclose the lag table and annual results, and qualify conflicting conclusions. Treat robustness checks as checks of one substantive claim, not an opportunity to select the most favourable test. Realised profitability and risk statistics answer complementary economic questions; they do not repair an unreliable significance calculation.
