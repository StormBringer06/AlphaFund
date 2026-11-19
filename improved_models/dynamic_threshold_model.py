"""Volatility-dependent entry/exit thresholds optimized for training Sharpe."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.optimize import differential_evolution

from backtest import run_walk_forward_backtest, summarize_backtest
from improved_models.regression_model import RegressionPairsTradingModel, _Parameters


_THRESHOLD_FEATURES = ("vix", "vol_a", "vol_b", "spread_vol")
_MIN_ENTRY_Z = 0.5
_MAX_ENTRY_Z = 2.75
_MIN_EXIT_Z = 0.0
_MIN_HYSTERESIS_GAP = 0.1
_PARAMETER_BOUNDS = (
    (1.0, 2.75),
    (-0.5, 0.5),
    (-0.5, 0.5),
    (0.1, 1.5),
    (-0.5, 0.5),
    (-0.5, 0.5),
)


class SharpeOptimizedThresholdModel(RegressionPairsTradingModel):
    """Regress daily z-score thresholds, fitting coefficients to net Sharpe.

    The entry and exit thresholds are linear functions of standardized VIX and
    pair volatility. Their coefficients are fit by maximizing net daily Sharpe
    over the latest lookback window. Thresholds are predicted each test date.
    Once entered, the spread position scales with divergence and is capped at
    ``max_gross_exposure``. It exits when the spread returns inside the
    predicted exit threshold or crosses through the mean.
    """

    def __init__(
        self,
        max_gross_exposure: float = 1.0,
        full_size_zscore: float = 3.0,
        transaction_cost_bps: float = 0.0,
        optimizer_maxiter: int = 8,
        optimizer_popsize: int = 5,
        vol_window: int = 20,
        lookback_years: int = 2,
        refit_interval_months: int = 6,
        min_train_observations: int = 120,
    ) -> None:
        super().__init__(
            regime_mode="none",
            vol_window=vol_window,
            lookback_years=lookback_years,
            refit_interval_months=refit_interval_months,
            min_train_observations=min_train_observations,
        )
        if not math.isfinite(max_gross_exposure) or max_gross_exposure <= 0:
            raise ValueError("max_gross_exposure must be finite and positive.")
        if not math.isfinite(full_size_zscore) or full_size_zscore <= _MAX_ENTRY_Z:
            raise ValueError(
                f"full_size_zscore must be greater than {_MAX_ENTRY_Z}."
            )
        if not math.isfinite(transaction_cost_bps) or transaction_cost_bps < 0:
            raise ValueError("transaction_cost_bps must be finite and non-negative.")
        if optimizer_maxiter < 1 or optimizer_popsize < 4:
            raise ValueError("optimizer_maxiter must be positive and optimizer_popsize >= 4.")

        self.max_gross_exposure = float(max_gross_exposure)
        self.full_size_zscore = float(full_size_zscore)
        self.transaction_cost_bps = float(transaction_cost_bps)
        self.optimizer_maxiter = optimizer_maxiter
        self.optimizer_popsize = optimizer_popsize
        self._threshold_means: np.ndarray | None = None
        self._threshold_stds: np.ndarray | None = None
        self._threshold_coefficients: np.ndarray | None = None
        self._threshold_fit_sharpe: float | None = None
        self._active_side = 0

    def fit(self, train_data: pd.DataFrame) -> SharpeOptimizedThresholdModel:
        """Fit spread features, then optimize threshold-regression coefficients."""
        self._active_side = 0
        super().fit(train_data)
        if self._params is None:
            raise RuntimeError("The regression model failed to initialize.")

        train = self._prepare(train_data, "training")
        features = self._features(train, self._params)
        cutoff = train["date"].iloc[-1] - pd.DateOffset(
            years=self.lookback_years
        )
        features = features.loc[features["date"] >= cutoff].copy()
        complete = features[list(_THRESHOLD_FEATURES)].notna().all(axis=1)
        features = features.loc[complete].reset_index(drop=True)
        if len(features) < self.min_train_observations:
            raise ValueError(
                "Too few complete observations to optimize entry and exit thresholds."
            )

        self._threshold_means = features[list(_THRESHOLD_FEATURES)].mean().to_numpy()
        self._threshold_stds = (
            features[list(_THRESHOLD_FEATURES)].std(ddof=1).replace(0.0, 1.0).to_numpy()
        )
        threshold_design = self._threshold_design(features)
        training_returns = self._training_returns(train, features["date"])
        hedge_ratio = self._params.hedge_ratio

        initial = np.array([2.0, 0.0, 0.0, 0.5, 0.0, 0.0])

        def objective(coefficients: np.ndarray) -> float:
            entry, exit_threshold = self._predict_thresholds(
                threshold_design, coefficients
            )
            target_a, target_b, _, _ = self._make_targets(
                features["spread_zscore"].to_numpy(),
                entry,
                exit_threshold,
                hedge_ratio,
                initial_side=0,
            )
            strategy_returns = self._net_training_returns(
                target_a,
                target_b,
                training_returns[:, 0],
                training_returns[:, 1],
            )
            volatility = float(strategy_returns.std(ddof=1))
            if not math.isfinite(volatility) or volatility <= 1e-12:
                return 1e6
            sharpe = float(strategy_returns.mean() / volatility * math.sqrt(252.0))
            return -sharpe if math.isfinite(sharpe) else 1e6

        result = differential_evolution(
            objective,
            bounds=_PARAMETER_BOUNDS,
            x0=initial,
            seed=0,
            maxiter=self.optimizer_maxiter,
            popsize=self.optimizer_popsize,
            polish=False,
            updating="immediate",
            workers=1,
        )
        if not np.isfinite(result.x).all() or not math.isfinite(float(result.fun)):
            raise RuntimeError("Threshold optimization did not produce finite parameters.")

        self._threshold_coefficients = np.asarray(result.x, dtype=float)
        self._threshold_fit_sharpe = -float(result.fun)
        return self

    def _threshold_design(self, features: pd.DataFrame) -> np.ndarray:
        if self._threshold_means is None or self._threshold_stds is None:
            raise RuntimeError("Fit the model before predicting thresholds.")
        standardized = (
            features[list(_THRESHOLD_FEATURES)].to_numpy(dtype=float)
            - self._threshold_means
        ) / self._threshold_stds
        standardized = np.nan_to_num(standardized, nan=0.0, posinf=3.0, neginf=-3.0)
        standardized = np.clip(standardized, -3.0, 3.0)
        market_volatility = standardized[:, 0]
        pair_volatility = standardized[:, 1:].mean(axis=1)
        return np.column_stack(
            [np.ones(len(features)), market_volatility, pair_volatility]
        )

    @staticmethod
    def _predict_thresholds(
        design: np.ndarray, coefficients: np.ndarray
    ) -> tuple[np.ndarray, np.ndarray]:
        entry = np.clip(design @ coefficients[:3], _MIN_ENTRY_Z, _MAX_ENTRY_Z)
        exit_threshold = np.clip(
            design @ coefficients[3:],
            _MIN_EXIT_Z,
            _MAX_ENTRY_Z,
        )
        exit_threshold = np.minimum(
            exit_threshold, entry - _MIN_HYSTERESIS_GAP
        )
        return entry, np.maximum(exit_threshold, _MIN_EXIT_Z)

    def _positions(
        self, segment: pd.DataFrame, params: _Parameters
    ) -> pd.DataFrame:
        if self._threshold_coefficients is None:
            raise RuntimeError("Call fit with training data before predict.")
        segment = segment.copy()
        design = self._threshold_design(segment)
        entry, exit_threshold = self._predict_thresholds(
            design, self._threshold_coefficients
        )
        valid = segment[list(_THRESHOLD_FEATURES)].notna().all(axis=1).to_numpy()
        zscore = segment["spread_zscore"].to_numpy(dtype=float)
        target_a, target_b, entries, exits = self._make_targets(
            zscore,
            entry,
            exit_threshold,
            params.hedge_ratio,
            initial_side=self._active_side,
            valid=valid,
        )
        if len(zscore):
            self._active_side = self._ending_side(
                zscore,
                entry,
                exit_threshold,
                self._active_side,
                valid,
            )

        segment["entry_z_threshold"] = entry
        segment["exit_z_threshold"] = exit_threshold
        segment["volatility_score"] = design[:, 1:].mean(axis=1)
        segment["trade_entry"] = entries
        segment["trade_exit"] = exits
        segment["signal"] = target_a * (1.0 + abs(params.hedge_ratio))
        segment["target_weight_a"] = target_a
        segment["target_weight_b"] = target_b
        segment["target_gross_exposure"] = np.abs(target_a) + np.abs(target_b)
        return segment.drop(columns=["next_spread_change"])

    def _make_targets(
        self,
        zscore: np.ndarray,
        entry: np.ndarray,
        exit_threshold: np.ndarray,
        hedge_ratio: float,
        initial_side: int,
        valid: np.ndarray | None = None,
    ) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        targets_a = np.zeros(len(zscore), dtype=float)
        targets_b = np.zeros(len(zscore), dtype=float)
        entries = np.zeros(len(zscore), dtype=bool)
        exits = np.zeros(len(zscore), dtype=bool)
        side = initial_side
        scale = 1.0 + abs(hedge_ratio)

        for index, z in enumerate(zscore):
            if valid is not None and not valid[index]:
                continue
            if side and side * z >= -exit_threshold[index]:
                side = 0
                exits[index] = True
            if side == 0 and abs(z) >= entry[index]:
                side = -1 if z > 0 else 1
                entries[index] = True
            if side:
                exposure = self.max_gross_exposure * np.clip(
                    (abs(z) - exit_threshold[index])
                    / (self.full_size_zscore - exit_threshold[index]),
                    0.0,
                    1.0,
                )
                targets_a[index] = side * exposure / scale
                targets_b[index] = -hedge_ratio * targets_a[index]
        return targets_a, targets_b, entries, exits

    @staticmethod
    def _ending_side(
        zscore: np.ndarray,
        entry: np.ndarray,
        exit_threshold: np.ndarray,
        initial_side: int,
        valid: np.ndarray,
    ) -> int:
        side = initial_side
        for index, z in enumerate(zscore):
            if not valid[index]:
                continue
            if side and side * z >= -exit_threshold[index]:
                side = 0
            if side == 0 and abs(z) >= entry[index]:
                side = -1 if z > 0 else 1
        return side

    @staticmethod
    def _training_returns(
        train: pd.DataFrame, dates: pd.Series
    ) -> np.ndarray:
        returns = train.set_index("date").loc[
            dates, ["adjusted_close_a", "adjusted_close_b"]
        ].pct_change(fill_method=None)
        return returns.fillna(0.0).to_numpy(dtype=float)

    def _net_training_returns(
        self,
        target_a: np.ndarray,
        target_b: np.ndarray,
        returns_a: np.ndarray,
        returns_b: np.ndarray,
    ) -> np.ndarray:
        position_a = np.r_[0.0, target_a[:-1]]
        position_b = np.r_[0.0, target_b[:-1]]
        gross = position_a * returns_a + position_b * returns_b
        turnover = np.abs(target_a - np.r_[0.0, target_a[:-1]]) + np.abs(
            target_b - np.r_[0.0, target_b[:-1]]
        )
        if len(turnover):
            turnover[-1] = abs(position_a[-1]) + abs(position_b[-1])
        return gross - turnover * (self.transaction_cost_bps / 10_000.0)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "screened_pair_prices.csv",
    )
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--transaction-cost-bps", type=float, default=0.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("backtest_results") / "dynamic_threshold",
    )
    args = parser.parse_args()

    data = pd.read_csv(args.input, parse_dates=["date"])
    model_factory = lambda: SharpeOptimizedThresholdModel(
        transaction_cost_bps=args.transaction_cost_bps
    )
    daily = run_walk_forward_backtest(
        data,
        model_factory=model_factory,
        start_year=args.start_year,
        end_year=args.end_year,
        train_years=args.train_years,
        transaction_cost_bps=args.transaction_cost_bps,
        model_name="sharpe_optimized_threshold",
    )
    summary = summarize_backtest(daily)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    suffix = f"_{args.transaction_cost_bps:g}bps"
    daily.to_csv(args.output_dir / f"daily_results{suffix}.csv", index=False)
    summary.to_csv(args.output_dir / f"summary{suffix}.csv", index=False)
    print(
        f"Wrote {len(daily)} daily rows and {len(summary)} pair/fold summaries "
        f"to {args.output_dir}."
    )


if __name__ == "__main__":
    main()
