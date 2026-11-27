"""Trading signal: vol-scaled pairs model with a low-volatility size boost.

Per pair, at each close:

1. **Spread.** OLS hedge ratio of log A on log B over the trailing two years,
   refit every six months from data strictly before the refit date.
2. **z-score.** ``z = (spread - m) / s`` with ``m`` / ``s`` the EWMA mean and std
   of the spread (half-life 120 days), lagged one day.
3. **Size.** ``clip(-z / 2, -1, 1)`` times two multipliers, capped at gross
   exposure 1:

   * **VIX overlay** ``clip(1 + 2 * vix_scale * (vix_pct - 0.5), 0.5, 1.5)``
     (``vix_pct`` = VIX percentile over the last 504 days);
   * **low-volatility boost** ``clip(reference_vol / spread_vol, 1, 1.5)``:
     ``spread_vol`` is the 60-day std of the pair's daily spread return and
     ``reference_vol`` its median over the previous 504 days (both lagged one
     day), so a pair is sized up - never down - while its spread is calmer
     than usual.
4. **No-trade band.** The position only moves when the target differs by more
   than 0.2.

``vix_scale`` is chosen each test year from {0, 0.25} by an inner walk-forward
on the four years before it (``select_parameters``, run without the boost).
Weights are close-of-day targets; the backtester applies them to the next day's
returns.
"""

from __future__ import annotations

from dataclasses import dataclass
import itertools
import math

import numpy as np
import pandas as pd

from backtest import run_walk_forward_backtest


PARAMETER_GRID = {
    "vix_scale": (0.0, 0.25),
    "band": (0.2,),
    "band_vix_scale": (0.0,),
}


_INPUT_COLUMNS = ("date", "adjusted_close_a", "adjusted_close_b", "vix")


def _prepare(data: pd.DataFrame, label: str) -> pd.DataFrame:
    """Validate and sort one pair's price/VIX frame."""
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


@dataclass(frozen=True)
class _Hedge:
    intercept: float
    hedge_ratio: float


class VolScaledNaivePairsTradingModel:
    """z-score sizing with VIX and low-volatility multipliers and a no-trade band.

    ``vol_boost_cap=1`` switches the low-volatility boost off.
    """

    def __init__(
        self,
        vix_scale: float = 0.0,
        band: float = 0.0,
        band_vix_scale: float = 0.0,
        zscore_halflife: float = 120.0,
        full_exposure_zscore: float = 2.0,
        max_gross_exposure: float = 1.0,
        vix_window: int = 504,
        vol_boost_cap: float = 1.5,
        vol_boost_window: int = 60,
        vol_boost_reference_window: int = 504,
        lookback_years: int = 2,
        refit_interval_months: int = 6,
        min_train_observations: int = 120,
    ) -> None:
        for name, value in (("vix_scale", vix_scale), ("band", band),
                            ("band_vix_scale", band_vix_scale)):
            if not math.isfinite(value) or value < 0:
                raise ValueError(f"{name} must be finite and non-negative.")
        if vix_scale > 0.5 or band_vix_scale > 0.5:
            raise ValueError("vix_scale and band_vix_scale must be at most 0.5.")
        if zscore_halflife <= 0 or full_exposure_zscore <= 0 or max_gross_exposure <= 0:
            raise ValueError(
                "zscore_halflife, full_exposure_zscore and max_gross_exposure "
                "must be positive."
            )
        if vix_window < 20:
            raise ValueError("vix_window must be at least 20.")
        if not math.isfinite(vol_boost_cap) or not 1.0 <= vol_boost_cap <= 2.0:
            raise ValueError("vol_boost_cap must be between 1 and 2.")
        if vol_boost_window < 5 or vol_boost_reference_window < vol_boost_window:
            raise ValueError(
                "vol_boost_window must be >= 5 and at most vol_boost_reference_window."
            )
        if lookback_years < 1 or refit_interval_months < 1:
            raise ValueError("lookback_years and refit_interval_months must be >= 1.")

        self.vix_scale = float(vix_scale)
        self.band = float(band)
        self.band_vix_scale = float(band_vix_scale)
        self.zscore_halflife = float(zscore_halflife)
        self.full_exposure_zscore = float(full_exposure_zscore)
        self.max_gross_exposure = float(max_gross_exposure)
        self.vix_window = vix_window
        self.vol_boost_cap = float(vol_boost_cap)
        self.vol_boost_window = vol_boost_window
        self.vol_boost_reference_window = vol_boost_reference_window
        self.lookback_years = lookback_years
        self.refit_interval_months = refit_interval_months
        self.min_train_observations = min_train_observations

        self._hedge: _Hedge | None = None
        self._history: pd.DataFrame | None = None
        self._next_refit_date: pd.Timestamp | None = None
        self._last_refit_date: pd.Timestamp | None = None
        self._position = 0.0

    # ------------------------------------------------------------------ API

    def fit(self, train_data: pd.DataFrame) -> VolScaledNaivePairsTradingModel:
        train = _prepare(train_data, "training")
        last_date = train["date"].iloc[-1]
        self._hedge = self._estimate(train, last_date + pd.Timedelta(days=1))
        self._history = train[list(_INPUT_COLUMNS)].reset_index(drop=True)
        self._last_refit_date = last_date
        self._next_refit_date = last_date + pd.DateOffset(
            months=self.refit_interval_months
        )
        self._position = 0.0
        return self

    def predict(self, test_data: pd.DataFrame) -> pd.DataFrame:
        if self._hedge is None or self._history is None:
            raise RuntimeError("Call fit with training data before predict.")
        test = _prepare(test_data, "test")
        if test["date"].iloc[0] <= self._history["date"].iloc[-1]:
            raise ValueError("Test dates must be strictly after previously seen dates.")

        combined = pd.concat(
            [self._history, test[list(_INPUT_COLUMNS)]], ignore_index=True
        )
        hedge = self._hedge
        last_refit = self._last_refit_date
        frames: list[pd.DataFrame] = []
        for start, end, refit in self._segments(test["date"]):
            if refit:
                hedge = self._estimate(combined.loc[combined["date"] < start], start)
                last_refit = start
            features = self._features(combined.loc[combined["date"] <= end], hedge)
            segment = features.loc[features["date"] >= start].copy()
            segment["beta_refit"] = (segment["date"] == start) & refit
            segment["beta_refit_date"] = last_refit
            frames.append(self._positions(segment, hedge))
        signals = pd.concat(frames, ignore_index=True)

        self._hedge = hedge
        self._last_refit_date = last_refit
        self._history = self._trim(combined, test["date"].iloc[-1])
        return test.merge(
            signals.drop(columns=list(_INPUT_COLUMNS[1:])),
            on="date",
            how="left",
            validate="one_to_one",
        )

    # ------------------------------------------------------------ internals

    def _estimate(self, known: pd.DataFrame, as_of: pd.Timestamp) -> _Hedge:
        window = known.loc[
            known["date"] >= as_of - pd.DateOffset(years=self.lookback_years)
        ]
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
        return _Hedge(float(log_a.mean() - hedge_ratio * log_b.mean()), hedge_ratio)

    def _features(self, frame: pd.DataFrame, hedge: _Hedge) -> pd.DataFrame:
        out = frame.reset_index(drop=True).copy()
        spread = np.log(out["adjusted_close_a"]) - (
            hedge.intercept + hedge.hedge_ratio * np.log(out["adjusted_close_b"])
        )
        ewm = spread.ewm(
            halflife=self.zscore_halflife, min_periods=int(self.zscore_halflife)
        )
        mean, std = ewm.mean().shift(1), ewm.std().shift(1)
        out["hedge_ratio"] = hedge.hedge_ratio
        out["spread"] = spread
        out["spread_zscore"] = (spread - mean) / std.where(std > 0)
        out["vix_pct"] = (
            out["vix"]
            .rolling(self.vix_window, min_periods=self.vix_window // 4)
            .rank(pct=True)
        )
        # Low-volatility boost inputs: lagged spread vol and its trailing median.
        spread_return = out["adjusted_close_a"].pct_change() - hedge.hedge_ratio * (
            out["adjusted_close_b"].pct_change()
        )
        spread_vol = spread_return.rolling(
            self.vol_boost_window, min_periods=max(5, self.vol_boost_window // 2)
        ).std().shift(1)
        reference_vol = spread_vol.rolling(
            self.vol_boost_reference_window,
            min_periods=max(20, self.vol_boost_reference_window // 4),
        ).median().shift(1)
        out["spread_vol"] = spread_vol
        out["spread_vol_ratio"] = reference_vol / spread_vol.where(spread_vol > 0)
        return out

    def _positions(self, segment: pd.DataFrame, hedge: _Hedge) -> pd.DataFrame:
        z = segment["spread_zscore"].to_numpy(dtype=float)
        vix_pct = segment["vix_pct"].fillna(0.5).to_numpy(dtype=float)
        base = np.clip(-z / self.full_exposure_zscore, -1.0, 1.0)
        multiplier = np.clip(1.0 + 2.0 * self.vix_scale * (vix_pct - 0.5), 0.5, 1.5)
        # Unknown ratio (warm-up) -> no boost; zero spread vol -> full boost.
        ratio = np.nan_to_num(
            segment["spread_vol_ratio"].to_numpy(dtype=float),
            nan=1.0, posinf=self.vol_boost_cap,
        )
        boost = np.clip(ratio, 1.0, self.vol_boost_cap)
        target = np.clip(
            base * multiplier * boost, -self.max_gross_exposure, self.max_gross_exposure
        )
        target = np.nan_to_num(target, nan=0.0)  # flat during EWMA warm-up
        band = self.band * np.clip(
            1.0 + 2.0 * self.band_vix_scale * (0.5 - vix_pct), 0.0, 2.0
        )

        position = np.empty(len(target))
        traded = np.zeros(len(target), dtype=bool)
        current = self._position
        for index, (wanted, width) in enumerate(zip(target, band)):
            if abs(wanted - current) > width:
                current = wanted
                traded[index] = True
            position[index] = current
        self._position = current

        scale = 1.0 + abs(hedge.hedge_ratio)
        segment["vix_multiplier"] = multiplier
        segment["vol_boost"] = boost
        segment["band_width"] = band
        segment["target_signal"] = target
        segment["rebalanced"] = traded
        segment["signal"] = position
        segment["target_weight_a"] = position / scale
        segment["target_weight_b"] = -hedge.hedge_ratio * position / scale
        segment["target_gross_exposure"] = np.abs(position)
        return segment

    def _segments(
        self, dates: pd.Series
    ) -> list[tuple[pd.Timestamp, pd.Timestamp, bool]]:
        segments: list[list[object]] = []
        for date in dates:
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
        return [(s, e, r) for s, e, r in segments]

    def _trim(self, data: pd.DataFrame, as_of: pd.Timestamp) -> pd.DataFrame:
        warmup_days = max(
            math.ceil(5 * self.zscore_halflife * 7 / 5),
            math.ceil(self.vix_window * 7 / 5),
        )
        cutoff = (
            as_of
            - pd.DateOffset(years=self.lookback_years)
            - pd.Timedelta(days=warmup_days)
        )
        return data.loc[data["date"] >= cutoff].reset_index(drop=True)


def _portfolio_sharpe(daily: pd.DataFrame) -> float:
    returns = daily.groupby("date")["strategy_return"].mean()
    vol = float(returns.std())
    return float(returns.mean() / vol * math.sqrt(252.0)) if vol > 0 else -math.inf


def select_parameters(
    data: pd.DataFrame,
    test_year: int,
    transaction_cost_bps: float,
    train_years: int = 4,
    inner_test_years: int = 2,
    grid: dict[str, tuple[float, ...]] = PARAMETER_GRID,
    vol_boost_cap: float = 1.0,
) -> tuple[dict[str, float], pd.DataFrame]:
    """Pick overlay/band parameters on the training window before ``test_year``.

    Runs an inner walk-forward over the last ``inner_test_years`` of the
    ``train_years`` before ``test_year`` (each inner year trained on the years
    before it) and maximizes the net Sharpe of the equal-weight portfolio of all
    pairs in ``data``. Only data before ``test_year`` is used.

    The search runs with the low-volatility boost off (``vol_boost_cap=1``), as
    in V5; the chosen values are then traded with the boost on.
    """
    inner_train_years = train_years - inner_test_years
    data = data.loc[
        (data["date"] >= pd.Timestamp(year=test_year - train_years, month=1, day=1))
        & (data["date"] < pd.Timestamp(year=test_year, month=1, day=1))
    ]
    rows = []
    for values in itertools.product(*grid.values()):
        params = dict(zip(grid, values))
        if params["band"] == 0 and params["band_vix_scale"] > 0:
            continue  # same as band_vix_scale = 0
        daily = run_walk_forward_backtest(
            data,
            model_factory=lambda p=params: VolScaledNaivePairsTradingModel(
                **p, vol_boost_cap=vol_boost_cap
            ),
            start_year=test_year - inner_test_years,
            end_year=test_year - 1,
            train_years=inner_train_years,
            transaction_cost_bps=transaction_cost_bps,
        )
        rows.append({**params, "inner_sharpe": _portfolio_sharpe(daily)})
    scores = pd.DataFrame(rows)
    best = scores.loc[scores["inner_sharpe"].idxmax(), list(grid)].to_dict()
    return {k: float(v) for k, v in best.items()}, scores


def run_signal(
    data: pd.DataFrame, selection: pd.DataFrame, years: range, train_years: int
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Yearly parameter search at 0 bps, then each test year's walk-forward run.

    Returns the gross daily results (one fold per test year) and the chosen
    parameters per year.
    """
    frames, chosen = [], []
    for fold_id, year in enumerate(years, start=1):
        pairs = selection.loc[selection["test_year"] == year, "pair_id"].unique()
        if not len(pairs):
            raise ValueError(f"No selected pairs for test year {year}.")
        year_data = data.loc[data["pair_id"].isin(pairs)]
        params, _ = select_parameters(year_data, year, 0.0, train_years)
        chosen.append({"test_year": year, **params, "pairs": ",".join(pairs)})
        print(f"  {year}: {params}", flush=True)
        daily = run_walk_forward_backtest(
            year_data, model_factory=lambda p=params: VolScaledNaivePairsTradingModel(**p),
            start_year=year, end_year=year, train_years=train_years,
            transaction_cost_bps=0.0, model_name="signal",
        )
        daily["fold_id"] = fold_id
        frames.append(daily)
    return pd.concat(frames, ignore_index=True), pd.DataFrame(chosen)
