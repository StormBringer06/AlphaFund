"""A simple formation-period, z-score pairs-trading model."""

from __future__ import annotations

from dataclasses import dataclass
import math

import pandas as pd


_PRICE_COLUMNS = ("adjusted_close_a", "adjusted_close_b")
_PAIR_METADATA_COLUMNS = ("pair_id", "ticker_a", "ticker_b")


@dataclass(frozen=True)
class _ModelParameters:
    intercept: float
    hedge_ratio: float
    spread_mean: float
    spread_std: float


@dataclass(frozen=True)
class _FitState:
    parameters: _ModelParameters
    last_processed_date: pd.Timestamp
    next_refit_date: pd.Timestamp
    last_refit_date: pd.Timestamp
    history: pd.DataFrame
    pair_metadata: dict[str, str]


class NaivePairsTradingModel:
    """Fit one pair and periodically refresh its rolling hedge ratio.

    The model regresses log price A on log price B, standardizes the resulting
    residual using the estimation-window statistics, then scales exposure
    linearly with the negative z-score. It refits on a trailing history window
    at fixed calendar intervals, using only observations before each prediction.
    """

    def __init__(
        self,
        full_exposure_zscore: float = 2.0,
        min_train_observations: int = 30,
        refit_interval_months: int = 6,
        hedge_lookback_years: int = 2,
    ) -> None:
        if not math.isfinite(full_exposure_zscore) or full_exposure_zscore <= 0:
            raise ValueError("full_exposure_zscore must be a finite positive number.")
        if min_train_observations < 3:
            raise ValueError("min_train_observations must be at least 3.")
        if refit_interval_months < 1:
            raise ValueError("refit_interval_months must be at least 1.")
        if hedge_lookback_years < 1:
            raise ValueError("hedge_lookback_years must be at least 1.")

        self.full_exposure_zscore = float(full_exposure_zscore)
        self.min_train_observations = min_train_observations
        self.refit_interval_months = refit_interval_months
        self.hedge_lookback_years = hedge_lookback_years
        self._fit_state: _FitState | None = None

    def fit(self, train_data: pd.DataFrame) -> NaivePairsTradingModel:
        """Fit initial parameters using the latest available lookback window."""
        train = self._prepare_data(train_data, "training")
        last_date = train["date"].iloc[-1]
        history = self._trim_history(train, last_date)
        parameters = self._estimate_parameters(history, "training")

        pair_metadata = {
            column: value
            for column in _PAIR_METADATA_COLUMNS
            if (value := self._metadata_value(train, column)) is not None
        }
        self._fit_state = _FitState(
            parameters=parameters,
            last_processed_date=last_date,
            next_refit_date=last_date
            + pd.DateOffset(months=self.refit_interval_months),
            last_refit_date=last_date,
            history=history,
            pair_metadata=pair_metadata,
        )
        return self

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        """Return test-period spread features and target portfolio weights.

        Weights are close-of-day targets. A backtest should apply them to the
        following period's returns rather than the same day's returns. At each
        scheduled refit, the current observation is scored using parameters
        estimated strictly from earlier observations.
        """
        if self._fit_state is None:
            raise RuntimeError("Call fit with training data before predict.")

        test = self._prepare_data(test_data, "test")
        state = self._fit_state
        if test["date"].iloc[0] <= state.last_processed_date:
            raise ValueError("Test dates must be strictly after previously processed dates.")
        self._validate_pair_metadata(test, state.pair_metadata)

        history = state.history.copy()
        parameters = state.parameters
        next_refit_date = state.next_refit_date
        last_refit_date = state.last_refit_date
        pending_history: list[dict[str, object]] = []
        prediction_rows: list[dict[str, object]] = []

        for _, row in test.iterrows():
            date = row["date"]
            refit_today = date >= next_refit_date
            if refit_today:
                if pending_history:
                    history = pd.concat(
                        [history, pd.DataFrame(pending_history)], ignore_index=True
                    )
                    history = self._trim_history(history, date)
                    pending_history = []

                parameters = self._estimate_parameters(history, "rolling refit")
                last_refit_date = date
                while next_refit_date <= date:
                    next_refit_date += pd.DateOffset(
                        months=self.refit_interval_months
                    )

            spread = math.log(row["adjusted_close_a"]) - (
                parameters.intercept
                + parameters.hedge_ratio * math.log(row["adjusted_close_b"])
            )
            spread_zscore = (
                spread - parameters.spread_mean
            ) / parameters.spread_std
            spread_signal = max(
                -1.0,
                min(1.0, -spread_zscore / self.full_exposure_zscore),
            )
            weight_a = spread_signal / (1.0 + abs(parameters.hedge_ratio))
            weight_b = (
                -parameters.hedge_ratio
                * spread_signal
                / (1.0 + abs(parameters.hedge_ratio))
            )

            prediction = row.to_dict()
            prediction.update(
                {
                    "hedge_ratio": parameters.hedge_ratio,
                    "spread": spread,
                    "spread_zscore": spread_zscore,
                    "target_gross_exposure": abs(weight_a) + abs(weight_b),
                    "target_weight_a": weight_a,
                    "target_weight_b": weight_b,
                    "beta_refit": refit_today,
                    "beta_refit_date": last_refit_date,
                }
            )
            prediction_rows.append(prediction)
            pending_history.append(
                {
                    "date": date,
                    "adjusted_close_a": row["adjusted_close_a"],
                    "adjusted_close_b": row["adjusted_close_b"],
                }
            )

        if pending_history:
            history = pd.concat(
                [history, pd.DataFrame(pending_history)], ignore_index=True
            )
            history = self._trim_history(history, test["date"].iloc[-1])

        self._fit_state = _FitState(
            parameters=parameters,
            last_processed_date=test["date"].iloc[-1],
            next_refit_date=next_refit_date,
            last_refit_date=last_refit_date,
            history=history,
            pair_metadata=state.pair_metadata,
        )
        return pd.DataFrame(prediction_rows)

    def _trim_history(
        self, data: pd.DataFrame, as_of_date: pd.Timestamp
    ) -> pd.DataFrame:
        cutoff = as_of_date - pd.DateOffset(years=self.hedge_lookback_years)
        columns = ["date", *_PRICE_COLUMNS]
        return (
            data.loc[data["date"] >= cutoff, columns]
            .sort_values("date")
            .reset_index(drop=True)
        )

    def _estimate_parameters(
        self, data: pd.DataFrame, label: str
    ) -> _ModelParameters:
        if len(data) < self.min_train_observations:
            raise ValueError(
                f"The {label} window has {len(data)} observations; "
                f"at least {self.min_train_observations} are required."
            )

        log_a = data["adjusted_close_a"].map(math.log)
        log_b = data["adjusted_close_b"].map(math.log)
        variance_b = float(log_b.var(ddof=1))
        if not math.isfinite(variance_b) or variance_b <= 0:
            raise ValueError(f"The {label} window has no variance in ticker_b prices.")

        hedge_ratio = float(log_b.cov(log_a) / variance_b)
        if not math.isfinite(hedge_ratio) or abs(hedge_ratio) < 1e-12:
            raise ValueError(f"The {label} window produced an invalid hedge ratio.")

        intercept = float(log_a.mean() - hedge_ratio * log_b.mean())
        spread = log_a - (intercept + hedge_ratio * log_b)
        spread_mean = float(spread.mean())
        spread_std = float(spread.std(ddof=1))
        if not math.isfinite(spread_std) or spread_std <= 0:
            raise ValueError(f"The {label} window has no finite spread variance.")
        return _ModelParameters(
            intercept=intercept,
            hedge_ratio=hedge_ratio,
            spread_mean=spread_mean,
            spread_std=spread_std,
        )

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