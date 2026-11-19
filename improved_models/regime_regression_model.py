"""Regression pairs model with EWMA z-score and linear VIX regime inputs.

Builds on ``RegressionPairsTradingModel`` (OLS forecast of the next-day spread
change, traded proportionally) with two changes:

1. **EWMA z-score.** The static z-score standardizes the spread with the
   mean / std of the 2-year estimation window. Out of sample the spread drifts
   away from that mean and its scale changes, so test-period z-scores had a std
   of ~1.9 and a persistent bias. Here the z-score is

       z_t = (spread_t - m_{t-1}) / s_{t-1}

   where ``m`` and ``s`` are exponentially weighted mean / std of the spread
   (half-life ``zscore_halflife`` trading days), lagged one day so today's
   spread does not standardize itself. The spread is recomputed with the
   current hedge ratio over the whole history at each refit, so the EWMA is
   always on a consistent series.

2. **Regimes only as linear inputs.** No position gating. Two discrete VIX
   states enter the regression, each as a level term and an interaction with z
   (the interaction is what lets a regime change the *strength of mean
   reversion*; the level term alone can only add a directional drift):

   * ``regime``      +1 / 0 / -1 for VIX below / between / above the
     estimation window's terciles (as in the parent model),
   * ``vix_extreme`` +1 / 0 / -1 for VIX below the 10th / between / above the
     90th percentile of the estimation window. This captures crisis spikes
     (e.g. March 2020) in its own coefficient instead of stretching the
     continuous ``z * vix`` term.

   The continuous ``z * vix`` / ``z * vol`` interactions are kept.

All thresholds and coefficients are re-estimated every
``refit_interval_months`` from data strictly before the refit date.

Caveat: the extreme-regime coefficient is only informative if the estimation
window contained extreme days; with a 2-year lookback it learns from the
window's own top / bottom decile, not from past crises outside the window.
"""

from __future__ import annotations

import argparse
import dataclasses
from dataclasses import dataclass
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest, summarize_backtest
from base_model import NaivePairsTradingModel
from improved_models.regression_model import (
    _VOL_FEATURES,
    RegressionPairsTradingModel,
    _Parameters,
    portfolio_metrics,
)


@dataclass(frozen=True)
class _RegimeParameters(_Parameters):
    extreme_low: float
    extreme_high: float


class RegimeRegressionPairsTradingModel(RegressionPairsTradingModel):
    """OLS spread-change forecast with EWMA z-score and VIX regime inputs."""

    def __init__(
        self,
        zscore_method: str = "ewma",
        zscore_halflife: float = 120.0,
        use_extreme_regime: bool = True,
        extreme_low_quantile: float = 0.10,
        extreme_high_quantile: float = 0.90,
        full_exposure_multiple: float = 2.0,
        vol_window: int = 20,
        lookback_years: int = 2,
        refit_interval_months: int = 6,
        regime_low_quantile: float = 1 / 3,
        regime_high_quantile: float = 2 / 3,
        min_train_observations: int = 120,
    ) -> None:
        super().__init__(
            regime_mode="input",
            full_exposure_multiple=full_exposure_multiple,
            vol_window=vol_window,
            lookback_years=lookback_years,
            refit_interval_months=refit_interval_months,
            regime_low_quantile=regime_low_quantile,
            regime_high_quantile=regime_high_quantile,
            min_train_observations=min_train_observations,
        )
        if zscore_method not in ("ewma", "static"):
            raise ValueError("zscore_method must be 'ewma' or 'static'.")
        if not math.isfinite(zscore_halflife) or zscore_halflife <= 0:
            raise ValueError("zscore_halflife must be finite and positive.")
        if not 0 < extreme_low_quantile < extreme_high_quantile < 1:
            raise ValueError(
                "Need 0 < extreme_low_quantile < extreme_high_quantile < 1."
            )
        self.zscore_method = zscore_method
        self.zscore_halflife = float(zscore_halflife)
        self.use_extreme_regime = use_extreme_regime
        self.extreme_low_quantile = extreme_low_quantile
        self.extreme_high_quantile = extreme_high_quantile

    # ------------------------------------------------------------ features

    def _features(self, frame: pd.DataFrame, params: _Parameters) -> pd.DataFrame:
        out = super()._features(frame, params)
        if self.zscore_method == "ewma":
            spread = out["spread"]
            ewm = spread.ewm(
                halflife=self.zscore_halflife,
                min_periods=int(self.zscore_halflife),
            )
            mean = ewm.mean().shift(1)
            std = ewm.std().shift(1)
            out["spread_zscore"] = (spread - mean) / std.where(std > 0)
        if isinstance(params, _RegimeParameters):
            out["vix_extreme"] = self.compute_regime(
                out["vix"], params.extreme_low, params.extreme_high
            )
        return out

    def _design(self, features: pd.DataFrame, params: _Parameters) -> np.ndarray:
        design = super()._design(features, params)
        if not self.use_extreme_regime:
            return design
        z = features["spread_zscore"].to_numpy()
        extreme = features["vix_extreme"].to_numpy(dtype=float)
        return np.column_stack([design, extreme, extreme * z])

    # ---------------------------------------------------------- estimation

    def _estimate(self, known: pd.DataFrame, cutoff: pd.Timestamp) -> _Parameters:
        """Fit hedge ratio, regime thresholds, scaling and OLS on [cutoff, end]."""
        window = known.loc[known["date"] >= cutoff]
        if len(window) < self.min_train_observations:
            raise ValueError(
                f"Estimation window has {len(window)} rows; "
                f"need {self.min_train_observations}."
            )
        log_a = np.log(window["adjusted_close_a"])
        log_b = np.log(window["adjusted_close_b"])
        hedge_ratio = float(log_b.cov(log_a) / log_b.var())
        if not math.isfinite(hedge_ratio) or abs(hedge_ratio) < 1e-12:
            raise ValueError("Estimation window produced an invalid hedge ratio.")
        intercept = float(log_a.mean() - hedge_ratio * log_b.mean())
        spread = log_a - (intercept + hedge_ratio * log_b)
        vix = window["vix"]

        params = _RegimeParameters(
            intercept=intercept,
            hedge_ratio=hedge_ratio,
            spread_mean=float(spread.mean()),
            spread_std=float(spread.std()),
            regime_low=float(vix.quantile(self.regime_low_quantile)),
            regime_high=float(vix.quantile(self.regime_high_quantile)),
            feature_means={},
            feature_stds={},
            coefficients=np.empty(0),
            forecast_std=float("nan"),
            extreme_low=float(vix.quantile(self.extreme_low_quantile)),
            extreme_high=float(vix.quantile(self.extreme_high_quantile)),
        )
        # Rolling / EWMA features use earlier rows too, so compute on all known data.
        features = self._features(known, params)
        features = features.loc[features["date"] >= cutoff].dropna(
            subset=[*_VOL_FEATURES, "spread_zscore", "next_spread_change"]
        )
        if len(features) < self.min_train_observations:
            raise ValueError("Too few complete feature rows to fit the regression.")

        means = {name: float(features[name].mean()) for name in _VOL_FEATURES}
        stds = {name: float(features[name].std()) or 1.0 for name in _VOL_FEATURES}
        params = dataclasses.replace(params, feature_means=means, feature_stds=stds)
        design = self._design(features, params)
        target = features["next_spread_change"].to_numpy()
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        forecast_std = float(np.std(design @ coefficients, ddof=1))
        if not math.isfinite(forecast_std) or forecast_std <= 0:
            raise ValueError("Regression produced a degenerate forecast.")
        return dataclasses.replace(
            params, coefficients=coefficients, forecast_std=forecast_std
        )

    def _trim(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        # Keep ~5 EWMA half-lives before the estimation window as warm-up
        # (residual weight of the truncated tail is ~3%).
        trimmed = super()._trim(data, as_of)
        if self.zscore_method != "ewma":
            return trimmed
        warmup = pd.Timedelta(days=math.ceil(5 * self.zscore_halflife * 7 / 5))
        cutoff = as_of - pd.DateOffset(years=self.lookback_years) - warmup
        cutoff = min(cutoff, trimmed["date"].iloc[0])
        return (
            data.loc[data["date"] >= cutoff, list(trimmed.columns)]
            .sort_values("date")
            .reset_index(drop=True)
        )

    def _positions(self, segment: pd.DataFrame, params: _Parameters) -> pd.DataFrame:
        out = super()._positions(segment, params)
        # Rows without a z-score (EWMA warm-up) must stay flat.
        missing = out["spread_zscore"].isna()
        if missing.any():
            out.loc[missing, ["signal", "target_weight_a", "target_weight_b",
                              "target_gross_exposure"]] = 0.0
        return out


# ---------------------------------------------------------------- comparison


def model_variants() -> dict[str, object]:
    """Naive baseline, the original regression, the new model and ablations."""
    return {
        "naive_zscore": NaivePairsTradingModel,
        "lr_regime_input": lambda: RegressionPairsTradingModel(regime_mode="input"),
        "regime_lr_static_z": lambda: RegimeRegressionPairsTradingModel(
            zscore_method="static"
        ),
        "regime_lr_ewma_z_no_extreme": lambda: RegimeRegressionPairsTradingModel(
            use_extreme_regime=False
        ),
        "regime_lr_ewma_z": RegimeRegressionPairsTradingModel,
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", type=Path, default=Path("data") / "screened_pair_prices.csv"
    )
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtest_results") / "regime_regression",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=["date"])
    daily = pd.concat(
        [
            run_walk_forward_backtest(
                data,
                model_factory=factory,
                start_year=args.start_year,
                end_year=args.end_year,
                train_years=args.train_years,
                transaction_cost_bps=args.transaction_cost_bps,
                model_name=name,
            )
            for name, factory in model_variants().items()
        ],
        ignore_index=True,
    )
    summary = summarize_backtest(daily)
    comparison = portfolio_metrics(daily)

    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.transaction_cost_bps:g}bps"
    daily.to_csv(args.output_dir / f"daily_results{suffix}.csv", index=False)
    summary.to_csv(args.output_dir / f"summary{suffix}.csv", index=False)
    comparison.to_csv(args.output_dir / f"comparison{suffix}.csv", index=False)
    with pd.option_context("display.width", 200, "display.precision", 4):
        print(comparison.to_string(index=False))
        print()
        print(
            summary.pivot_table(
                index="pair_id", columns="model_name", values="sharpe_ratio"
            ).round(2)
        )


if __name__ == "__main__":
    main()
