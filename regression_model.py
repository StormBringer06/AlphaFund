"""Linear-regression pairs-trading model with volatility features and a regime.

For each pair the model:

1. Estimates the hedge ratio by OLS of log price A on log price B over a
   trailing window (as in ``NaivePairsTradingModel``) and forms the spread
   ``log(A) - (intercept + hedge_ratio * log(B))`` and its z-score.
2. Builds daily features known at the close:
   * ``vix`` - VIX close,
   * ``vol_a`` / ``vol_b`` - rolling annualized volatility of each stock's log
     returns,
   * ``spread_vol`` - rolling annualized volatility of daily spread changes,
   * ``regime`` - a -1 / 0 / +1 heuristic volatility state (see below).
3. Fits OLS of the *next-day spread change* on those features over the same
   trailing window, then trades the prediction: a positive forecast goes long
   the spread (long A, short ``hedge_ratio`` of B), sized linearly and capped at
   full gross exposure when the forecast reaches ``full_exposure_multiple``
   in-sample standard deviations of the fitted forecasts.

The volatility features enter as interactions with the z-score
(``z * vix``, ``z * vol_a``, ...). On their own they say how much the spread may
move but not in which direction; multiplied by the z-score they let the
regression learn how strongly the spread mean-reverts in each volatility
environment.

Regime (VIX-based heuristic): +1 when VIX is below the ``regime_low_quantile``
of its trailing estimation window (calm, so mean reversion should be reliable),
-1 when it is above ``regime_high_quantile`` (stress, when pairs tend to break
down), 0 otherwise. Thresholds come only from the estimation window. Override
``RegressionPairsTradingModel.compute_regime`` to try other definitions.

``regime_mode`` controls how the regime is used:

* ``"none"``  - ignored (reference model),
* ``"input"`` - added to the regression as ``regime`` and ``regime * z``,
* ``"gate"``  - not in the regression; position scaled by 0 / 0.5 / 1 for
  regime -1 / 0 / +1,
* ``"both"``  - regression input, and flat whenever the regime is -1.

Everything (hedge ratio, standardization, regime thresholds, coefficients) is
re-estimated every ``refit_interval_months`` from observations strictly before
the refit date, so predictions use no future data.
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


_PRICE_COLUMNS = ("adjusted_close_a", "adjusted_close_b")
_INPUT_COLUMNS = ("date", *_PRICE_COLUMNS, "vix")
_VOL_FEATURES = ("vix", "vol_a", "vol_b", "spread_vol")
REGIME_MODES = ("none", "input", "gate", "both")
_GATE_SCALE = {-1: 0.0, 0: 0.5, 1: 1.0}


@dataclass(frozen=True)
class _Parameters:
    intercept: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    regime_low: float
    regime_high: float
    feature_means: dict[str, float]
    feature_stds: dict[str, float]
    coefficients: np.ndarray
    forecast_std: float


class RegressionPairsTradingModel:
    """Per-pair OLS forecast of next-day spread change; see module docstring."""

    def __init__(
        self,
        regime_mode: str = "input",
        full_exposure_multiple: float = 2.0,
        vol_window: int = 20,
        lookback_years: int = 2,
        refit_interval_months: int = 6,
        regime_low_quantile: float = 1 / 3,
        regime_high_quantile: float = 2 / 3,
        min_train_observations: int = 120,
    ) -> None:
        if regime_mode not in REGIME_MODES:
            raise ValueError(f"regime_mode must be one of {REGIME_MODES}.")
        if not math.isfinite(full_exposure_multiple) or full_exposure_multiple <= 0:
            raise ValueError("full_exposure_multiple must be finite and positive.")
        if vol_window < 2:
            raise ValueError("vol_window must be at least 2.")
        if lookback_years < 1 or refit_interval_months < 1:
            raise ValueError("lookback_years and refit_interval_months must be >= 1.")
        if not 0 < regime_low_quantile < regime_high_quantile < 1:
            raise ValueError("Need 0 < regime_low_quantile < regime_high_quantile < 1.")

        self.regime_mode = regime_mode
        self.full_exposure_multiple = float(full_exposure_multiple)
        self.vol_window = vol_window
        self.lookback_years = lookback_years
        self.refit_interval_months = refit_interval_months
        self.regime_low_quantile = regime_low_quantile
        self.regime_high_quantile = regime_high_quantile
        self.min_train_observations = min_train_observations

        self._params: _Parameters | None = None
        self._history: pd.DataFrame | None = None
        self._next_refit_date: pd.Timestamp | None = None
        self._last_refit_date: pd.Timestamp | None = None

    # ------------------------------------------------------------------ API

    def fit(self, train_data: pd.DataFrame) -> RegressionPairsTradingModel:
        train = self._prepare(train_data, "training")
        last_date = train["date"].iloc[-1]
        cutoff = last_date - pd.DateOffset(years=self.lookback_years)
        self._params = self._estimate(train, cutoff)
        self._history = self._trim(train, last_date)
        self._last_refit_date = last_date
        self._next_refit_date = last_date + pd.DateOffset(
            months=self.refit_interval_months
        )
        return self

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if self._params is None or self._history is None:
            raise RuntimeError("Call fit with training data before predict.")
        test = self._prepare(test_data, "test")
        if test["date"].iloc[0] <= self._history["date"].iloc[-1]:
            raise ValueError("Test dates must be strictly after previously seen dates.")

        combined = pd.concat(
            [self._history, test[list(_INPUT_COLUMNS)]], ignore_index=True
        )
        test_dates = test["date"]

        # Split the test period into segments that share one set of parameters.
        segments = self._segments(test_dates)

        params = self._params
        last_refit_date = self._last_refit_date
        frames: list[pd.DataFrame] = []
        for start, end, refit in segments:
            if refit:
                known = combined.loc[combined["date"] < start]
                params = self._estimate(
                    known, start - pd.DateOffset(years=self.lookback_years)
                )
                last_refit_date = start
            upto = combined.loc[combined["date"] <= end]
            features = self._features(upto, params)
            segment = features.loc[features["date"] >= start].copy()
            segment["beta_refit"] = segment["date"] == start if refit else False
            segment["beta_refit_date"] = last_refit_date
            frames.append(self._positions(segment, params))
        signals = pd.concat(frames, ignore_index=True)

        self._params = params
        self._last_refit_date = last_refit_date
        self._history = self._trim(combined, test_dates.iloc[-1])

        output = test.merge(
            signals.drop(columns=list(_INPUT_COLUMNS[1:])),
            on="date",
            how="left",
            validate="one_to_one",
        )
        return output

    # ------------------------------------------------------------ internals

    def compute_regime(self, vix: pd.Series, low: float, high: float) -> pd.Series:
        """Heuristic volatility state: +1 calm, 0 normal, -1 stressed.

        ``low``/``high`` are VIX quantiles from the estimation window. Override
        in a subclass to try a different -1 / 0 / +1 definition.
        """
        regime = pd.Series(0, index=vix.index, dtype=int)
        regime[vix < low] = 1
        regime[vix > high] = -1
        return regime

    def _segments(
        self, test_dates: pd.Series
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, bool]]:
        """Group test dates by refit schedule; advances ``_next_refit_date``."""
        segments: list[list[object]] = []
        for date in test_dates:
            if date >= self._next_refit_date:
                segments.append([date, date, True])
                while self._next_refit_date <= date:
                    self._next_refit_date += pd.DateOffset(
                        months=self.refit_interval_months
                    )
            elif not segments:
                segments.append([date, date, False])
            else:
                segments[-1][1] = date
        return [(start, end, refit) for start, end, refit in segments]

    def _features(self, frame: pd.DataFrame, params: _Parameters) -> pd.DataFrame:
        """Backward-looking features (plus next-day target) for every row."""
        out = frame.reset_index(drop=True).copy()
        log_a = np.log(out["adjusted_close_a"])
        log_b = np.log(out["adjusted_close_b"])
        spread = log_a - (params.intercept + params.hedge_ratio * log_b)
        annualize = math.sqrt(252.0)
        window = self.vol_window

        out["hedge_ratio"] = params.hedge_ratio
        out["spread"] = spread
        out["spread_zscore"] = (spread - params.spread_mean) / params.spread_std
        out["vol_a"] = log_a.diff().rolling(window).std() * annualize
        out["vol_b"] = log_b.diff().rolling(window).std() * annualize
        out["spread_vol"] = spread.diff().rolling(window).std() * annualize
        out["regime"] = self.compute_regime(
            out["vix"], params.regime_low, params.regime_high
        )
        out["next_spread_change"] = spread.shift(-1) - spread
        return out

    def _design(self, features: pd.DataFrame, params: _Parameters) -> np.ndarray:
        z = features["spread_zscore"].to_numpy()
        columns = [np.ones(len(features)), z]
        for name in _VOL_FEATURES:
            standardized = (
                features[name].to_numpy() - params.feature_means[name]
            ) / params.feature_stds[name]
            columns.append(z * standardized)
        if self.regime_mode in ("input", "both"):
            regime = features["regime"].to_numpy(dtype=float)
            columns.extend([regime, regime * z])
        return np.column_stack(columns)

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

        provisional = _Parameters(
            intercept=intercept,
            hedge_ratio=hedge_ratio,
            spread_mean=float(spread.mean()),
            spread_std=float(spread.std()),
            regime_low=float(window["vix"].quantile(self.regime_low_quantile)),
            regime_high=float(window["vix"].quantile(self.regime_high_quantile)),
            feature_means={},
            feature_stds={},
            coefficients=np.empty(0),
            forecast_std=float("nan"),
        )
        # Rolling features use earlier rows too, so compute on all known data.
        features = self._features(known, provisional)
        features = features.loc[features["date"] >= cutoff].dropna(
            subset=[*_VOL_FEATURES, "next_spread_change"]
        )
        if len(features) < self.min_train_observations:
            raise ValueError("Too few complete feature rows to fit the regression.")

        means = {name: float(features[name].mean()) for name in _VOL_FEATURES}
        stds = {name: float(features[name].std()) or 1.0 for name in _VOL_FEATURES}
        provisional = _replace(provisional, feature_means=means, feature_stds=stds)
        design = self._design(features, provisional)
        target = features["next_spread_change"].to_numpy()
        coefficients, *_ = np.linalg.lstsq(design, target, rcond=None)
        forecast_std = float(np.std(design @ coefficients, ddof=1))
        if not math.isfinite(forecast_std) or forecast_std <= 0:
            raise ValueError("Regression produced a degenerate forecast.")
        return _replace(
            provisional, coefficients=coefficients, forecast_std=forecast_std
        )

    def _positions(self, segment: pd.DataFrame, params: _Parameters) -> pd.DataFrame:
        segment = segment.copy()
        complete = segment[list(_VOL_FEATURES)].notna().all(axis=1)
        forecast = np.full(len(segment), np.nan)
        if complete.any():
            forecast[complete.to_numpy()] = (
                self._design(segment.loc[complete], params) @ params.coefficients
            )
        signal = np.clip(
            forecast / (self.full_exposure_multiple * params.forecast_std), -1.0, 1.0
        )
        signal = np.nan_to_num(signal, nan=0.0)
        regime = segment["regime"].to_numpy()
        if self.regime_mode == "gate":
            signal = signal * np.vectorize(_GATE_SCALE.get)(regime)
        elif self.regime_mode == "both":
            signal = np.where(regime == -1, 0.0, signal)

        scale = 1.0 + abs(params.hedge_ratio)
        segment["predicted_spread_change"] = forecast
        segment["signal"] = signal
        segment["target_weight_a"] = signal / scale
        segment["target_weight_b"] = -params.hedge_ratio * signal / scale
        segment["target_gross_exposure"] = (
            segment["target_weight_a"].abs() + segment["target_weight_b"].abs()
        )
        return segment.drop(columns=["next_spread_change"])

    def _trim(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        # Keep the lookback plus a buffer so rolling features have warm-up data.
        cutoff = as_of - pd.DateOffset(years=self.lookback_years, months=3)
        return (
            data.loc[data["date"] >= cutoff, list(_INPUT_COLUMNS)]
            .sort_values("date")
            .reset_index(drop=True)
        )

    @staticmethod
    def _prepare(data: pd.DataFrame, label: str) -> pd.DataFrame:
        missing = set(_INPUT_COLUMNS).difference(data.columns)
        if missing:
            raise ValueError(f"{label} data is missing columns: {sorted(missing)}.")
        if data.empty:
            raise ValueError(f"{label} data must not be empty.")
        prepared = data.copy()
        prepared["date"] = pd.to_datetime(prepared["date"], errors="raise")
        if prepared["date"].duplicated().any():
            raise ValueError(f"{label} data contains duplicate dates.")
        prepared = prepared.sort_values("date").reset_index(drop=True)
        for column in _INPUT_COLUMNS[1:]:
            values = pd.to_numeric(prepared[column], errors="coerce")
            if values.isna().any() or (values <= 0).any() or np.isinf(values).any():
                raise ValueError(f"{label} column {column} must be finite and positive.")
            prepared[column] = values
        return prepared


def _replace(params: _Parameters, **changes: object) -> _Parameters:
    return dataclasses.replace(params, **changes)


# ---------------------------------------------------------------- comparison


def model_variants() -> dict[str, object]:
    """Models compared by ``main``: naive baseline plus each regime mode."""
    variants: dict[str, object] = {"naive_zscore": NaivePairsTradingModel}
    for mode in REGIME_MODES:
        variants[f"lr_regime_{mode}"] = (
            lambda mode=mode: RegressionPairsTradingModel(regime_mode=mode)
        )
    return variants


def portfolio_metrics(daily: pd.DataFrame) -> pd.DataFrame:
    """Equal-weight daily portfolio across pairs, summarized per model."""
    rows = []
    for model_name, group in daily.groupby("model_name", sort=False):
        returns = group.groupby("date")["strategy_return"].mean()
        vol = float(returns.std())
        wealth = (1.0 + returns).cumprod()
        years = len(returns) / 252.0
        rows.append(
            {
                "model_name": model_name,
                "annualized_return": float(wealth.iloc[-1] ** (1 / years) - 1),
                "annualized_volatility": vol * math.sqrt(252.0),
                "sharpe_ratio": float(returns.mean() / vol * math.sqrt(252.0)),
                "max_drawdown": float((wealth / wealth.cummax() - 1).min()),
                "average_gross_exposure": float(group["position_gross_exposure"].mean()),
                "average_daily_turnover": float(group["turnover"].mean()),
            }
        )
    return pd.DataFrame(rows)


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
        "--output-dir", type=Path, default=Path("backtest_results") / "regression"
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=["date"])
    daily_frames = []
    for name, factory in model_variants().items():
        daily_frames.append(
            run_walk_forward_backtest(
                data,
                model_factory=factory,
                start_year=args.start_year,
                end_year=args.end_year,
                train_years=args.train_years,
                transaction_cost_bps=args.transaction_cost_bps,
                model_name=name,
            )
        )
    daily = pd.concat(daily_frames, ignore_index=True)
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
