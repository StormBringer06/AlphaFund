"""Primary test and separate optional checks using a SYNTHETIC example CSV."""

from pathlib import Path
import numpy as np
from pairs_hac import paired_hac_test

# Replace these arrays with your actual matched daily net portfolio returns.
csv_path = Path(__file__).with_name("example_returns_SYNTHETIC.csv")
data = np.genfromtxt(csv_path, delimiter=",", names=True, dtype=None, encoding="utf-8")
naive_returns, vix_returns = data["naive_return"], data["vix_return"]

# This is the complete primary test. No dates or extra checks are required.
result = paired_hac_test(naive_returns, vix_returns)
print("SYNTHETIC EXAMPLE ONLY")
print("Mean advantage (bp/day):", result["mean_difference"] * 10_000)
print("HAC SE (bp/day):", result["hac_standard_error"] * 10_000)
print("One-sided p-value:", result["p_value_one_sided"])
for decision in result["decisions"]:
    print(decision)

# OPTIONAL: these assess robustness; they do not change the primary test.
from optional_checks import annual_grouped_test, annual_mean_checks, lag_sensitivity

print("Optional lag sensitivity:")
for row in lag_sensitivity(naive_returns, vix_returns):
    if "skipped" in row:
        print(row)
    else:
        print({key: row[key] for key in ("maxlags", "hac_standard_error", "p_value_one_sided", "warning")})
print("Optional annual means:")
for row in annual_mean_checks(naive_returns, vix_returns, data["date"]):
    print(row)

# A formal additional check, requiring independent approximately Gaussian
# annual estimates of a common mean. Inspect the partial-year coverage first.
print("Optional annual grouped test (synthetic example has a partial final year):")
year_result = annual_grouped_test(naive_returns, vix_returns, data["date"])
print({key: year_result[key] for key in ("n_years", "degrees_of_freedom", "mean_of_annual_daily_means", "t_statistic", "p_value_one_sided", "assumption_note")})
