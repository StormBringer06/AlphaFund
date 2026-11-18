"""A simple formation-period, z-score pairs-trading model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


_PRICE_COLUMNS = ("adjusted_close_a", "adjusted_close_b")
_PAIR_METADATA_COLUMNS = ("pair_id", "ticker_a", "ticker_b")


@dataclass(frozen=True)
class _FitState:
    intercept: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float
    last_train_date: pd.Timestamp
    pair_metadata: dict[str, str]


class NaivePairsTradingModel:
    """Fit one pair on a training period and predict continuous test weights.

    The model regresses log price A on log price B, standardizes the resulting
    residual using training-period statistics, then scales exposure linearly
    with the negative z-score. Test-period observations do not update the fit.
    """

    def __init__(
        self,
        full_exposure_zscore: float = 2.0,
        min_train_observations: int = 30,
    ) -> None:
        if not math.isfinite(full_exposure_zscore) or full_exposure_zscore <= 0:
            raise ValueError("full_exposure_zscore must be a finite positive number.")
        if min_train_observations < 3:
            raise ValueError("min_train_observations must be at least 3.")

        self.full_exposure_zscore = float(full_exposure_zscore)
        self.min_train_observations = min_train_observations
        self._fit_state: _FitState | None = None

    def fit(self, train_data: pd.DataFrame) -> NaivePairsTradingModel:
        """Estimate the hedge ratio and spread distribution from one pair."""
        train = self._prepare_data(train_data, "training")
        if len(train) < self.min_train_observations:
            raise ValueError(
                f"Training data has {len(train)} observations; "
                f"at least {self.min_train_observations} are required."
            )

        log_a = train["adjusted_close_a"].map(math.log)
        log_b = train["adjusted_close_b"].map(math.log)
        variance_b = float(log_b.var(ddof=1))
        if not math.isfinite(variance_b) or variance_b <= 0:
            raise ValueError("Training prices for ticker_b must have nonzero variance.")

        hedge_ratio = float(log_b.cov(log_a) / variance_b)
        if not math.isfinite(hedge_ratio) or abs(hedge_ratio) < 1e-12:
            raise ValueError("The estimated hedge ratio must be finite and nonzero.")

        intercept = float(log_a.mean() - hedge_ratio * log_b.mean())
        spread = log_a - (intercept + hedge_ratio * log_b)
        spread_mean = float(spread.mean())
        spread_std = float(spread.std(ddof=1))
        if not math.isfinite(spread_std) or spread_std <= 0:
            raise ValueError("Training spread must have nonzero finite variance.")

        pair_metadata = {
            column: value
            for column in _PAIR_METADATA_COLUMNS
            if (value := self._metadata_value(train, column)) is not None
        }
        self._fit_state = _FitState(
            intercept=intercept,
            hedge_ratio=hedge_ratio,
            spread_mean=spread_mean,
            spread_std=spread_std,
            last_train_date=train["date"].iloc[-1],
            pair_metadata=pair_metadata,
        )
        return self

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Return out-of-sample spread features and target portfolio weights.

        Weights are close-of-day targets. A backtest should apply them to the
        following period's returns rather than the same day's returns.
        """
        if self._fit_state is None:
            raise RuntimeError("Call fit with training data before predict.")

        test = self._prepare_data(test_data, "test")
        state = self._fit_state
        if test["date"].iloc[0] <= state.last_train_date:
            raise ValueError("Test dates must be strictly after the training dates.")
        self._validate_pair_metadata(test, state.pair_metadata)

        log_a = test["adjusted_close_a"].map(math.log)
        log_b = test["adjusted_close_b"].map(math.log)
        spread = log_a - (state.intercept + state.hedge_ratio * log_b)
        spread_zscore = (spread - state.spread_mean) / state.spread_std
        spread_signal = (-spread_zscore / self.full_exposure_zscore).clip(-1.0, 1.0)

        predictions = test.copy()
        predictions["hedge_ratio"] = state.hedge_ratio
        predictions["spread"] = spread
        predictions["spread_zscore"] = spread_zscore
        predictions["target_weight_a"] = spread_signal / (
            1.0 + abs(state.hedge_ratio)
        )
        predictions["target_weight_b"] = (
            -state.hedge_ratio
            * spread_signal
            / (1.0 + abs(state.hedge_ratio))
        )
        predictions["target_gross_exposure"] = (
            predictions["target_weight_a"].abs()
            + predictions["target_weight_b"].abs()
        )
        return predictions

    @staticmethod
    def _prepare_data(data: pd.DataFrame, label: str) -> pd.DataFrame:
        required_columns = {"date", *_PRICE_COLUMNS}
        missing_columns = required_columns.difference(data.columns)
        if missing_columns:
            missing = ", ".join(sorted(missing_columns))
            raise ValueError(f"{label.capitalize()} data is missing columns: {missing}.")
        if data.empty:
            raise ValueError(f"{label.capitalize()} data must not be empty.")

        prepared = data.copy()
        prepared["date"] = pd.to_datetime(prepared["date"], errors="coerce")
        if prepared["date"].isna().any():
            raise ValueError(f"{label.capitalize()} data contains invalid dates.")
        if prepared["date"].duplicated().any():
            raise ValueError(f"{label.capitalize()} data contains duplicate dates.")
        prepared = prepared.sort_values("date").reset_index(drop=True)

        for column in _PRICE_COLUMNS:
            prices = pd.to_numeric(prepared[column], errors="coerce")
            if prices.isna().any() or not prices.map(math.isfinite).all():
                raise ValueError(
                    f"{label.capitalize()} column {column} must contain finite prices."
                )
            if (prices <= 0).any():
                raise ValueError(
                    f"{label.capitalize()} column {column} must contain positive prices."
                )
            prepared[column] = prices
        return prepared

    @staticmethod
    def _metadata_value(data: pd.DataFrame, column: str) -> str | None:
        if column not in data.columns:
            return None
        values = data[column].dropna().astype(str).unique()
        if len(values) > 1:
            raise ValueError(f"A model input must contain only one {column}.")
        return str(values[0]) if len(values) == 1 else None

    @classmethod
    def _validate_pair_metadata(
        cls, test: pd.DataFrame, pair_metadata: dict[str, str]
    ) -> None:
        for column, expected_value in pair_metadata.items():
            actual_value = cls._metadata_value(test, column)
            if actual_value != expected_value:
                raise ValueError(
                    f"Test data {column} must match the fitted pair "
                    f"({expected_value})."
                )