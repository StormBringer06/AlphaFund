"""V6 adds a pair-specific realized-volatility overlay to V5's sizing."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import apply_positions_to_returns, run_walk_forward_backtest
from improved_models.ciaranMaxModel import EXIT_RULE, apply_exit_targets
from improved_models.exit_model import (
    checkpoint_rows,
    load_context,
    load_ridge_weights,
    ridge_decisions,
    save_ridge_weights,
    traded_checkpoints,
    train_ridge_weights,
)
from improved_models.vol_scaled_naive_model import (
    _Hedge,
    VolScaledNaivePairsTradingModel,
)
from improved_models.vol_scaled_naive_model_v4 import (
    dynamic_leverage,
    lever,
    portfolio_returns,
)
from screen_pairs import UNIVERSE


MODEL_NAME = "v6_pair_vol"
PAIR_VOL_SCALE_GRIDS = {
    "signed": (0.0, -0.25, 0.25, -0.5, 0.5),
    "vix_style": (0.0, 0.25),
    "off": (0.0,),
}
VIX_SCALE_GRID = (0.0, 0.25, 0.5)
RISK_EXPONENT_GRID = (0.0, 0.25, 0.5, 1.0)
V6_EXIT_WEIGHTS_PATH = (
    Path("models") / "exit_ridge_weights_v6_lowvol_pit_2008_2017.json"
)


def add_pair_vol_features(
    frame: pd.DataFrame,
    hedge_ratio: float,
    vol_window: int = 60,
    rank_window: int = 504,
) -> pd.DataFrame:
    """Rank spread volatility relative to the volatility of its two legs.

    All rolling volatility estimates are shifted one day. The ratio is
    ``spread_vol / sqrt(vol_a**2 + (beta * vol_b)**2)``; its trailing rank is
    a scale-free measure of how much constituent movement is appearing in
    the hedged spread.
    """
    if vol_window < 5 or rank_window < vol_window:
        raise ValueError("rank_window must be at least vol_window, and vol_window >= 5.")

    out = frame.reset_index(drop=True).copy()
    log_return_a = np.log(out["adjusted_close_a"]).diff()
    log_return_b = np.log(out["adjusted_close_b"]).diff()
    spread_return = log_return_a - hedge_ratio * log_return_b
    min_vol_periods = max(5, vol_window // 2)

    vol_a = log_return_a.rolling(
        vol_window, min_periods=min_vol_periods
    ).std().shift(1)
    vol_b = log_return_b.rolling(
        vol_window, min_periods=min_vol_periods
    ).std().shift(1)
    spread_vol = spread_return.rolling(
        vol_window, min_periods=min_vol_periods
    ).std().shift(1)
    leg_vol = np.sqrt(vol_a.pow(2) + (hedge_ratio * vol_b).pow(2))
    ratio = spread_vol / leg_vol.where(leg_vol > 0)
    out["pair_vol_ratio"] = ratio
    out["pair_vol_percentile"] = ratio.rolling(
        rank_window,
        min_periods=max(20, rank_window // 4),
    ).rank(pct=True)
    return out


def add_pair_risk_features(
    frame: pd.DataFrame,
    hedge_ratio: float,
    vol_window: int = 60,
    reference_window: int = 504,
    risk_exponent: float = 1.0,
    risk_multiplier_cap: float = 1.5,
) -> pd.DataFrame:
    """Estimate lagged risk per unit of pair gross exposure and its target."""
    if vol_window < 5 or reference_window < vol_window:
        raise ValueError(
            "reference_window must be at least vol_window, and vol_window >= 5."
        )
    if not math.isfinite(risk_exponent) or not 0.0 <= risk_exponent <= 1.0:
        raise ValueError("risk_exponent must be finite and between 0 and 1.")
    if not math.isfinite(risk_multiplier_cap) or not 1.0 <= risk_multiplier_cap <= 2.0:
        raise ValueError("risk_multiplier_cap must be finite and between 1 and 2.")
    out = frame.reset_index(drop=True).copy()
    return_a = out["adjusted_close_a"].pct_change()
    return_b = out["adjusted_close_b"].pct_change()
    spread_return = return_a - hedge_ratio * return_b
    risk_vol = (
        spread_return.rolling(
            vol_window,
            min_periods=max(5, vol_window // 2),
        )
        .std()
        .shift(1)
        / (1.0 + abs(hedge_ratio))
    )
    reference_vol = risk_vol.rolling(
        reference_window,
        min_periods=max(20, reference_window // 4),
    ).median().shift(1)
    risk_ratio = reference_vol / risk_vol.where(risk_vol > 0)
    multiplier = risk_ratio.pow(risk_exponent).clip(0.5, risk_multiplier_cap)
    out["pair_risk_per_unit"] = risk_vol
    out["pair_risk_reference"] = reference_vol
    out["pair_risk_ratio"] = risk_ratio
    out["pair_risk_multiplier"] = multiplier.fillna(1.0)
    return out


def pool_checkpoints_low_vol_boost(
    context: dict,
    vix_scales: dict[int, float],
    risk_exponent: float = 1.0,
    risk_multiplier_cap: float = 1.5,
    vol_window: int = 60,
    reference_window: int = 504,
) -> pd.DataFrame:
    """Build causal classifier training episodes using V6 low-vol sizing."""
    if vol_window < 5 or reference_window < vol_window:
        raise ValueError("reference_window must be at least vol_window >= 5.")
    if not math.isfinite(risk_exponent) or not 0.0 <= risk_exponent <= 1.0:
        raise ValueError("risk_exponent must be finite and between 0 and 1.")
    if not math.isfinite(risk_multiplier_cap) or risk_multiplier_cap < 1.0:
        raise ValueError("risk_multiplier_cap must be finite and at least 1.")
    stocks, dates = context["stocks"], context["dates"]
    pairs = [
        (a, b)
        for tickers in UNIVERSE.values()
        for i, a in enumerate(sorted(tickers))
        for b in sorted(tickers)[i + 1 :]
        if a in stocks and b in stocks
    ]
    rows = []
    for a, b in pairs:
        features = context["arrays"](a, b)
        z = features["z"]
        spread_return = features["spread_ret"]
        min_vol_periods = max(5, vol_window // 2)
        risk = (
            pd.Series(spread_return)
            .rolling(vol_window, min_periods=min_vol_periods)
            .std()
            .shift(1)
        )
        reference = risk.rolling(
            reference_window,
            min_periods=max(20, reference_window // 4),
        ).median().shift(1)
        risk_ratio = reference / risk.where(risk > 0)
        risk_ratio = risk_ratio.to_numpy(dtype=float)
        risk_ratio = np.nan_to_num(risk_ratio, nan=1.0, posinf=100.0, neginf=0.0)
        risk_ratio = np.clip(risk_ratio, 0.0, 100.0)
        pair_multiplier = np.clip(
            np.power(risk_ratio, risk_exponent),
            1.0,
            risk_multiplier_cap,
        )
        years = pd.Index(dates.year)
        vix_scale = np.array([vix_scales.get(int(year), 0.0) for year in years])
        vix_pct = np.nan_to_num(features["vix_pct"], nan=0.5)
        vix_multiplier = np.clip(
            1.0 + 2.0 * vix_scale * (vix_pct - 0.5),
            0.5,
            1.5,
        )
        target = np.clip(-z / 2.0, -1.0, 1.0) * vix_multiplier * pair_multiplier
        target = np.nan_to_num(np.clip(target, -1.0, 1.0), nan=0.0)

        position = np.zeros(len(target))
        current = 0.0
        for index, wanted in enumerate(target):
            if abs(wanted - current) > 0.2:
                current = wanted
            position[index] = current

        pnl = np.zeros(len(target))
        pnl[1:] = (
            position[:-1] * np.nan_to_num(spread_return[1:])
            - 5e-4 * np.abs(np.diff(position))
        )
        side = np.sign(position)
        index = 0
        while index < len(side):
            if side[index] != 0 and (index == 0 or side[index - 1] != side[index]):
                end = index
                while end + 1 < len(side) and side[end + 1] == side[index]:
                    end += 1
                if end + 1 < len(side) and dates[index] >= pd.Timestamp("2006-01-01"):
                    rows.extend(
                        checkpoint_rows(
                            index,
                            np.arange(index + 1, end + 2),
                            pnl[index + 1 : end + 2],
                            features,
                            dates[end + 1],
                            {
                                "ep_key": f"{a}_{b}_{index}",
                                "source": "v6_low_vol_pool",
                                "episode_pnl": float(pnl[index + 1 : end + 2].sum()),
                            },
                        )
                    )
                index = end + 1
            else:
                index += 1
    if not rows:
        raise ValueError("No resolved V6 low-vol pool episodes were generated.")
    return pd.DataFrame(rows)


class VolScaledNaivePairsTradingModelV6(VolScaledNaivePairsTradingModel):
    """V5 sizing with a learned, pair-specific volatility multiplier."""

    def __init__(
        self,
        vix_scale: float = 0.0,
        band: float = 0.0,
        band_vix_scale: float = 0.0,
        zscore_halflife: float = 120.0,
        pair_vol_scale: float = 0.0,
        pair_vol_mode: str = "percentile",
        risk_exponent: float = 1.0,
        risk_multiplier_cap: float = 1.5,
        pair_vol_window: int = 60,
        pair_vol_rank_window: int = 504,
        vix_window: int = 504,
        lookback_years: int = 2,
        refit_interval_months: int = 6,
        min_train_observations: int = 120,
    ) -> None:
        if not math.isfinite(pair_vol_scale) or abs(pair_vol_scale) > 0.5:
            raise ValueError("pair_vol_scale must be finite and between -0.5 and 0.5.")
        if pair_vol_mode not in {"percentile", "risk_target", "low_vol_boost"}:
            raise ValueError(
                "pair_vol_mode must be 'percentile', 'risk_target', or 'low_vol_boost'."
            )
        if pair_vol_mode != "percentile" and pair_vol_scale != 0:
            raise ValueError("pair_vol_scale must be zero for direct risk-based modes.")
        if not math.isfinite(risk_exponent) or not 0.0 <= risk_exponent <= 1.0:
            raise ValueError("risk_exponent must be finite and between 0 and 1.")
        if (
            not math.isfinite(risk_multiplier_cap)
            or not 1.0 <= risk_multiplier_cap <= 2.0
        ):
            raise ValueError("risk_multiplier_cap must be finite and between 1 and 2.")
        if pair_vol_window < 5 or pair_vol_rank_window < pair_vol_window:
            raise ValueError(
                "pair_vol_rank_window must be at least pair_vol_window, "
                "and pair_vol_window must be >= 5."
            )
        self.pair_vol_scale = float(pair_vol_scale)
        self.pair_vol_mode = pair_vol_mode
        self.risk_exponent = float(risk_exponent)
        self.risk_multiplier_cap = float(risk_multiplier_cap)
        self.pair_vol_window = pair_vol_window
        self.pair_vol_rank_window = pair_vol_rank_window
        super().__init__(
            vix_scale=vix_scale,
            band=band,
            band_vix_scale=band_vix_scale,
            zscore_halflife=zscore_halflife,
            full_exposure_zscore=2.0,
            max_gross_exposure=1.0,
            vix_window=vix_window,
            lookback_years=lookback_years,
            refit_interval_months=refit_interval_months,
            min_train_observations=min_train_observations,
        )

    def _features(self, frame: pd.DataFrame, hedge: _Hedge) -> pd.DataFrame:
        out = super()._features(frame, hedge)
        out = add_pair_vol_features(
            out,
            hedge.hedge_ratio,
            self.pair_vol_window,
            self.pair_vol_rank_window,
        )
        if self.pair_vol_mode != "percentile":
            out = add_pair_risk_features(
                out,
                hedge.hedge_ratio,
                self.pair_vol_window,
                self.pair_vol_rank_window,
                self.risk_exponent,
                self.risk_multiplier_cap,
            )
        return out

    def _pair_vol_multiplier(self, segment: pd.DataFrame) -> np.ndarray:
        if self.pair_vol_mode == "risk_target":
            return segment["pair_risk_multiplier"].to_numpy(dtype=float)
        if self.pair_vol_mode == "low_vol_boost":
            risk_ratio = segment["pair_risk_ratio"].to_numpy(dtype=float)
            risk_ratio = np.nan_to_num(
                risk_ratio, nan=1.0, posinf=100.0, neginf=0.0
            )
            risk_ratio = np.clip(risk_ratio, 0.0, 100.0)
            return np.clip(
                np.power(risk_ratio, self.risk_exponent),
                1.0,
                self.risk_multiplier_cap,
            )
        pair_vol_pct = segment["pair_vol_percentile"].fillna(0.5).to_numpy(dtype=float)
        return np.clip(
            1.0 + 2.0 * self.pair_vol_scale * (pair_vol_pct - 0.5),
            0.5,
            1.5,
        )

    def _positions(self, segment: pd.DataFrame, hedge: _Hedge) -> pd.DataFrame:
        z = segment["spread_zscore"].to_numpy(dtype=float)
        vix_pct = segment["vix_pct"].fillna(0.5).to_numpy(dtype=float)
        base = np.clip(-z / self.full_exposure_zscore, -1.0, 1.0)
        vix_multiplier = np.clip(
            1.0 + 2.0 * self.vix_scale * (vix_pct - 0.5),
            0.5,
            1.5,
        )
        pair_vol_multiplier = self._pair_vol_multiplier(segment)
        target = np.clip(base * vix_multiplier * pair_vol_multiplier, -1.0, 1.0)
        target = np.nan_to_num(target, nan=0.0)
        band = self.band * np.clip(
            1.0 + 2.0 * self.band_vix_scale * (0.5 - vix_pct),
            0.0,
            2.0,
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
        segment["vix_multiplier"] = vix_multiplier
        segment["pair_vol_multiplier"] = pair_vol_multiplier
        segment["band_width"] = band
        segment["target_signal"] = target
        segment["rebalanced"] = traded
        segment["signal"] = position
        segment["target_weight_a"] = position / scale
        segment["target_weight_b"] = -hedge.hedge_ratio * position / scale
        segment["target_gross_exposure"] = np.abs(position)
        return segment


def _read_v5_parameters(path: Path, years: range) -> pd.DataFrame:
    params = pd.read_csv(path)
    required = {"test_year", "vix_scale", "band", "band_vix_scale", "pairs"}
    missing = required.difference(params.columns)
    if missing:
        raise ValueError(f"{path} is missing V5 parameter columns: {sorted(missing)}")
    params = params.loc[params["test_year"].isin(years)].copy()
    absent = sorted(set(years).difference(params["test_year"].astype(int)))
    if absent:
        raise ValueError(f"{path} has no V5-selected parameters for years {absent}.")
    if params["test_year"].duplicated().any():
        raise ValueError(f"{path} contains duplicate yearly V5 parameter rows.")
    return params.sort_values("test_year").reset_index(drop=True)


def _portfolio_sharpe(daily: pd.DataFrame) -> float:
    returns = portfolio_returns(daily)
    volatility = float(returns.std())
    return (
        float(returns.mean() / volatility * math.sqrt(252.0))
        if volatility > 0
        else -math.inf
    )


def select_pair_vol_scale(
    year_data: pd.DataFrame,
    test_year: int,
    v5_params: dict[str, float],
    train_years: int = 4,
    inner_test_years: int = 2,
    pair_vol_scales: tuple[float, ...] = PAIR_VOL_SCALE_GRIDS["signed"],
    pair_vol_window: int = 60,
    pair_vol_rank_window: int = 504,
) -> tuple[float, pd.DataFrame]:
    """Choose the signed pair-vol overlay using only pre-test-year data."""
    training = year_data.loc[
        (year_data["date"] >= pd.Timestamp(year=test_year - train_years, month=1, day=1))
        & (year_data["date"] < pd.Timestamp(year=test_year, month=1, day=1))
    ]
    scores = []
    for pair_vol_scale in pair_vol_scales:
        params = {
            **v5_params,
            "pair_vol_scale": pair_vol_scale,
            "pair_vol_window": pair_vol_window,
            "pair_vol_rank_window": pair_vol_rank_window,
        }
        daily = run_walk_forward_backtest(
            training,
            model_factory=lambda p=params: VolScaledNaivePairsTradingModelV6(**p),
            start_year=test_year - inner_test_years,
            end_year=test_year - 1,
            train_years=train_years - inner_test_years,
            transaction_cost_bps=0.0,
        )
        scores.append(
            {"pair_vol_scale": pair_vol_scale, "inner_sharpe": _portfolio_sharpe(daily)}
        )
    score_table = pd.DataFrame(scores)
    best = score_table.loc[score_table["inner_sharpe"].idxmax(), "pair_vol_scale"]
    return float(best), score_table


def select_low_vol_boost_vix_scale(
    year_data: pd.DataFrame,
    test_year: int,
    v5_params: dict[str, float],
    train_years: int,
    inner_test_years: int,
    risk_exponent: float,
    risk_multiplier_cap: float,
    pair_vol_window: int = 60,
    pair_vol_rank_window: int = 504,
) -> tuple[float, pd.DataFrame]:
    """Tune low-vol-boost VIX sizing on pre-test data only."""
    training = year_data.loc[
        (year_data["date"] >= pd.Timestamp(year=test_year - train_years, month=1, day=1))
        & (year_data["date"] < pd.Timestamp(year=test_year, month=1, day=1))
    ]
    scores = []
    for vix_scale in VIX_SCALE_GRID:
        params = {
            **v5_params,
            "vix_scale": vix_scale,
            "pair_vol_scale": 0.0,
            "pair_vol_mode": "low_vol_boost",
            "risk_exponent": risk_exponent,
            "risk_multiplier_cap": risk_multiplier_cap,
            "pair_vol_window": pair_vol_window,
            "pair_vol_rank_window": pair_vol_rank_window,
        }
        daily = run_walk_forward_backtest(
            training,
            model_factory=lambda p=params: VolScaledNaivePairsTradingModelV6(**p),
            start_year=test_year - inner_test_years,
            end_year=test_year - 1,
            train_years=train_years - inner_test_years,
            transaction_cost_bps=0.0,
        )
        scores.append(
            {"vix_scale": vix_scale, "inner_sharpe": _portfolio_sharpe(daily)}
        )
    score_table = pd.DataFrame(scores)
    best = score_table.loc[score_table["inner_sharpe"].idxmax(), "vix_scale"]
    return float(best), score_table


def select_low_vol_boost_exponent(
    year_data: pd.DataFrame,
    test_year: int,
    v5_params: dict[str, float],
    train_years: int,
    inner_test_years: int,
    risk_multiplier_cap: float,
    pair_vol_window: int = 60,
    pair_vol_rank_window: int = 504,
) -> tuple[float, pd.DataFrame]:
    """Tune low-vol boost strength on pre-test years only."""
    training = year_data.loc[
        (year_data["date"] >= pd.Timestamp(year=test_year - train_years, month=1, day=1))
        & (year_data["date"] < pd.Timestamp(year=test_year, month=1, day=1))
    ]
    scores = []
    for risk_exponent in RISK_EXPONENT_GRID:
        params = {
            **v5_params,
            "pair_vol_scale": 0.0,
            "pair_vol_mode": "low_vol_boost",
            "risk_exponent": risk_exponent,
            "risk_multiplier_cap": risk_multiplier_cap,
            "pair_vol_window": pair_vol_window,
            "pair_vol_rank_window": pair_vol_rank_window,
        }
        daily = run_walk_forward_backtest(
            training,
            model_factory=lambda p=params: VolScaledNaivePairsTradingModelV6(**p),
            start_year=test_year - inner_test_years,
            end_year=test_year - 1,
            train_years=train_years - inner_test_years,
            transaction_cost_bps=0.0,
        )
        scores.append(
            {"risk_exponent": risk_exponent, "inner_sharpe": _portfolio_sharpe(daily)}
        )
    score_table = pd.DataFrame(scores)
    best = score_table.loc[score_table["inner_sharpe"].idxmax(), "risk_exponent"]
    return float(best), score_table


def run_v6(
    data: pd.DataFrame,
    selection: pd.DataFrame,
    v5_params: pd.DataFrame,
    years: range,
    train_years: int,
    inner_test_years: int,
    pair_vol_scales: tuple[float, ...] = PAIR_VOL_SCALE_GRIDS["signed"],
    sizing_method: str = "percentile",
    risk_exponent: float = 1.0,
    risk_multiplier_cap: float = 1.5,
    retune_vix_scale: bool = False,
    tune_risk_exponent: bool = False,
    pair_vol_window: int = 60,
    pair_vol_rank_window: int = 504,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    if sizing_method not in {"percentile", "risk_target", "low_vol_boost"}:
        raise ValueError(
            "sizing_method must be 'percentile', 'risk_target', or 'low_vol_boost'."
        )
    if retune_vix_scale and tune_risk_exponent:
        raise ValueError("Retune only one low-vol-boost parameter at a time.")
    if (retune_vix_scale or tune_risk_exponent) and sizing_method != "low_vol_boost":
        raise ValueError("Low-vol-boost tuning requires sizing_method='low_vol_boost'.")
    if not math.isfinite(risk_exponent) or not 0.0 <= risk_exponent <= 1.0:
        raise ValueError("risk_exponent must be finite and between 0 and 1.")
    frames = []
    chosen = []
    score_tables = []
    for fold_id, year in enumerate(years, start=1):
        row = v5_params.loc[v5_params["test_year"] == year]
        if len(row) != 1:
            raise ValueError(f"Expected one V5 parameter row for test year {year}.")
        row = row.iloc[0]
        pairs = selection.loc[selection["test_year"] == year, "pair_id"].astype(str).unique()
        if not pairs.size:
            raise ValueError(f"No point-in-time selected pairs for test year {year}.")
        if set(str(row["pairs"]).split(",")) != set(pairs):
            raise ValueError(
                f"V5 parameter pairs do not match point-in-time selection for {year}."
            )

        v5_settings = {
            "vix_scale": float(row["vix_scale"]),
            "band": float(row["band"]),
            "band_vix_scale": float(row["band_vix_scale"]),
        }
        year_data = data.loc[data["pair_id"].isin(pairs)]
        if sizing_method == "low_vol_boost" and tune_risk_exponent:
            pair_vol_scale = 0.0
            risk_exponent, scores = select_low_vol_boost_exponent(
                year_data,
                year,
                v5_settings,
                train_years,
                inner_test_years,
                risk_multiplier_cap,
                pair_vol_window,
                pair_vol_rank_window,
            )
            scores["test_year"] = year
            score_tables.append(scores)
        elif sizing_method == "low_vol_boost" and retune_vix_scale:
            pair_vol_scale = 0.0
            vix_scale, scores = select_low_vol_boost_vix_scale(
                year_data,
                year,
                v5_settings,
                train_years,
                inner_test_years,
                risk_exponent,
                risk_multiplier_cap,
                pair_vol_window,
                pair_vol_rank_window,
            )
            v5_settings["vix_scale"] = vix_scale
            scores["test_year"] = year
            score_tables.append(scores)
        elif sizing_method in {"risk_target", "low_vol_boost"}:
            pair_vol_scale = 0.0
            scores = pd.DataFrame()
        else:
            pair_vol_scale, scores = select_pair_vol_scale(
                year_data,
                year,
                v5_settings,
                train_years=train_years,
                inner_test_years=inner_test_years,
                pair_vol_scales=pair_vol_scales,
                pair_vol_window=pair_vol_window,
                pair_vol_rank_window=pair_vol_rank_window,
            )
            scores["test_year"] = year
            score_tables.append(scores)
        model_params = {
            **v5_settings,
            "pair_vol_scale": pair_vol_scale,
            "pair_vol_mode": sizing_method,
            "risk_exponent": risk_exponent,
            "risk_multiplier_cap": risk_multiplier_cap,
            "pair_vol_window": pair_vol_window,
            "pair_vol_rank_window": pair_vol_rank_window,
        }
        daily = run_walk_forward_backtest(
            year_data,
            model_factory=lambda p=model_params: VolScaledNaivePairsTradingModelV6(**p),
            start_year=year,
            end_year=year,
            train_years=train_years,
            transaction_cost_bps=0.0,
            model_name=MODEL_NAME,
        )
        daily["fold_id"] = fold_id
        frames.append(daily)
        selected = {
            "test_year": year,
            **model_params,
            "pairs": ",".join(pairs),
        }
        if sizing_method == "percentile":
            selected["pair_vol_inner_sharpe"] = float(
                scores.loc[
                    scores["pair_vol_scale"] == pair_vol_scale, "inner_sharpe"
                ].iloc[0]
            )
        elif sizing_method == "low_vol_boost" and retune_vix_scale:
            selected["vix_inner_sharpe"] = float(
                scores.loc[
                    scores["vix_scale"] == v5_settings["vix_scale"],
                    "inner_sharpe",
                ].iloc[0]
            )
        elif sizing_method == "low_vol_boost" and tune_risk_exponent:
            selected["risk_exponent_inner_sharpe"] = float(
                scores.loc[
                    scores["risk_exponent"] == risk_exponent,
                    "inner_sharpe",
                ].iloc[0]
            )
        chosen.append(selected)
        print(
            f"  {year}: sizing_method={sizing_method}, "
            f"pair_vol_scale={pair_vol_scale:g}, V5 settings={v5_settings}",
            flush=True,
        )
    return (
        pd.concat(frames, ignore_index=True),
        pd.DataFrame(chosen),
        (
            pd.concat(score_tables, ignore_index=True)
            if score_tables
            else pd.DataFrame()
        ),
    )


def seed_returns(
    data: pd.DataFrame,
    selection: pd.DataFrame,
    params: dict,
    first_year: int,
    train_years: int,
    inner_test_years: int,
) -> pd.DataFrame:
    pairs = selection.loc[selection["test_year"] == first_year, "pair_id"].unique()
    year_data = data.loc[data["pair_id"].isin(pairs)]
    training = year_data.loc[
        (year_data["date"] >= pd.Timestamp(year=first_year - train_years, month=1, day=1))
        & (year_data["date"] < pd.Timestamp(year=first_year, month=1, day=1))
    ]
    return run_walk_forward_backtest(
        training,
        model_factory=lambda: VolScaledNaivePairsTradingModelV6(**params),
        start_year=first_year - inner_test_years,
        end_year=first_year - 1,
        train_years=train_years - inner_test_years,
        transaction_cost_bps=0.0,
        model_name=MODEL_NAME,
    )


def levered(
    daily: pd.DataFrame,
    seed: pd.DataFrame,
    name: str,
    cost: float,
    target_vol: float,
    max_leverage: float,
    vol_halflife: float,
) -> pd.DataFrame:
    history = pd.concat(
        [
            portfolio_returns(apply_positions_to_returns(seed, transaction_cost_bps=cost)),
            portfolio_returns(daily),
        ]
    ).sort_index()
    history = history[~history.index.duplicated(keep="last")]
    dates = pd.Index(sorted(daily["date"].unique()))
    leverage = dynamic_leverage(
        history, target_vol, max_leverage, vol_halflife
    ).reindex(dates)
    if leverage.isna().any():
        raise ValueError("Dynamic leverage is unavailable for some V6 test dates.")
    return lever(daily, leverage, name, cost)


def with_borrow(daily: pd.DataFrame, borrow_bps: float) -> pd.DataFrame:
    out = daily.copy()
    borrow = out["position_gross_exposure"] * 0.5 * borrow_bps / 10_000.0 / 252.0
    out["strategy_return"] = out["strategy_return"] - borrow
    return out


def summarize(daily: pd.DataFrame) -> dict[str, float]:
    returns = portfolio_returns(daily)
    volatility = float(returns.std())
    return {
        "total_return": float((1.0 + returns).prod() - 1.0),
        "sharpe": (
            float(returns.mean() / volatility * math.sqrt(252.0))
            if volatility > 0
            else float("nan")
        ),
        "annualized_volatility": float(volatility * math.sqrt(252.0)),
        "average_gross_exposure": float(daily["position_gross_exposure"].mean()),
        "average_daily_turnover": float(daily["turnover"].mean()),
        "average_leverage": float(daily["leverage"].mean()),
    }


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument(
        "--input",
        type=Path,
        default=Path("data") / "pit_2008_2026" / "pit_pair_prices.csv",
    )
    parser.add_argument(
        "--selection",
        type=Path,
        default=Path("data") / "pit_2008_2026" / "pit_pair_selection.csv",
    )
    parser.add_argument(
        "--v5-parameters",
        type=Path,
        default=Path("backtest_results")
        / "v5_extended_2008_2026"
        / "v3_selected_parameters.csv",
    )
    parser.add_argument(
        "--exit-weights",
        type=Path,
        default=Path("models") / "exit_ridge_weights_pit_2008_2026.json",
    )
    parser.add_argument(
        "--retrain-exits",
        action="store_true",
        help="Train expanding-year exit weights on V6-sized pool episodes.",
    )
    parser.add_argument(
        "--v6-exit-weights",
        type=Path,
        default=V6_EXIT_WEIGHTS_PATH,
        help="Separate output path for weights created by --retrain-exits.",
    )
    parser.add_argument("--start-year", type=int, default=2008)
    parser.add_argument("--end-year", type=int, default=2017)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--inner-test-years", type=int, default=2)
    parser.add_argument(
        "--sizing-method",
        choices=("percentile", "risk_target", "low_vol_boost"),
        default="percentile",
        help=(
            "Use percentile-tuned sizing, direct inverse-vol risk targeting, "
            "or a low-volatility-only size boost."
        ),
    )
    parser.add_argument(
        "--risk-exponent",
        type=float,
        default=1.0,
        help="Power for direct risk scaling, between 0 and 1.",
    )
    parser.add_argument(
        "--risk-multiplier-cap",
        type=float,
        default=1.5,
        help="Upper cap for direct volatility sizing multipliers, between 1 and 2.",
    )
    parser.add_argument("--pair-vol-window", type=int, default=60)
    parser.add_argument("--pair-vol-rank-window", type=int, default=504)
    parser.add_argument(
        "--retune-vix-scale",
        action="store_true",
        help="Retune VIX sizing annually for low-vol boost using only prior data.",
    )
    parser.add_argument(
        "--tune-risk-exponent",
        action="store_true",
        help="Select low-vol boost exponent annually from prior data.",
    )
    parser.add_argument(
        "--pair-vol-grid",
        choices=tuple(PAIR_VOL_SCALE_GRIDS),
        default="signed",
        help=(
            "Scale candidates: signed {0, ±0.25, ±0.5}, "
            "VIX-style {0, +0.25}, or off {0}."
        ),
    )
    parser.add_argument(
        "--disable-vix-overlay",
        action="store_true",
        help="Set VIX sizing and VIX-dependent band scales to zero.",
    )
    parser.add_argument("--target-vol", type=float, default=0.08)
    parser.add_argument("--max-leverage", type=float, default=4.0)
    parser.add_argument("--vol-halflife", type=float, default=60.0)
    parser.add_argument("--net-cost-bps", type=float, default=5.0)
    parser.add_argument("--borrow-bps", type=float, default=50.0)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
    )
    args = parser.parse_args()
    if args.retune_vix_scale and args.sizing_method != "low_vol_boost":
        parser.error("--retune-vix-scale requires --sizing-method low_vol_boost.")
    if args.tune_risk_exponent and args.sizing_method != "low_vol_boost":
        parser.error("--tune-risk-exponent requires --sizing-method low_vol_boost.")
    if args.retune_vix_scale and args.tune_risk_exponent:
        parser.error("Choose either --retune-vix-scale or --tune-risk-exponent.")
    if args.retrain_exits and args.sizing_method != "low_vol_boost":
        parser.error("--retrain-exits requires --sizing-method low_vol_boost.")
    if args.retrain_exits and (args.retune_vix_scale or args.tune_risk_exponent):
        parser.error("--retrain-exits requires fixed, previously selected V6 sizing.")
    if args.retrain_exits and args.v6_exit_weights.resolve() == args.exit_weights.resolve():
        parser.error("--v6-exit-weights must not overwrite --exit-weights.")
    if args.output_dir is None:
        if args.sizing_method in {"risk_target", "low_vol_boost"}:
            exponent_suffix = (
                ""
                if args.risk_exponent == 1.0
                else f"_gamma_{str(args.risk_exponent).replace('.', 'p')}"
            )
            cap_suffix = (
                ""
                if args.risk_multiplier_cap == 1.5
                else f"_cap_{str(args.risk_multiplier_cap).replace('.', 'p')}"
            )
            model_suffix = f"pair_{args.sizing_method}{exponent_suffix}{cap_suffix}"
            if args.retune_vix_scale:
                model_suffix += "_retuned_vix"
            if args.tune_risk_exponent:
                model_suffix += "_tuned_gamma"
        else:
            model_suffix = "pair_vol"
        if args.retrain_exits:
            model_suffix += "_v6_exits"
        if args.pair_vol_window != 60 or args.pair_vol_rank_window != 504:
            model_suffix += (
                f"_vol{args.pair_vol_window}_rank{args.pair_vol_rank_window}"
            )
        args.output_dir = (
            Path("backtest_results")
            / f"v6_{model_suffix}_{args.start_year}_{args.end_year}"
        )
    years = range(args.start_year, args.end_year + 1)

    data = pd.read_csv(args.input, parse_dates=["date"])
    selection = pd.read_csv(args.selection)
    v5_params = _read_v5_parameters(args.v5_parameters, years)
    if args.disable_vix_overlay:
        v5_params = v5_params.copy()
        v5_params["vix_scale"] = 0.0
        v5_params["band_vix_scale"] = 0.0
    v6_gross, chosen, scores = run_v6(
        data,
        selection,
        v5_params,
        years,
        args.train_years,
        args.inner_test_years,
        PAIR_VOL_SCALE_GRIDS[args.pair_vol_grid],
        args.sizing_method,
        args.risk_exponent,
        args.risk_multiplier_cap,
        args.retune_vix_scale,
        args.tune_risk_exponent,
        args.pair_vol_window,
        args.pair_vol_rank_window,
    )
    first_params = chosen.iloc[0][
        [
            "vix_scale",
            "band",
            "band_vix_scale",
            "pair_vol_scale",
            "pair_vol_mode",
            "risk_exponent",
            "risk_multiplier_cap",
            "pair_vol_window",
            "pair_vol_rank_window",
        ]
    ].to_dict()
    for key in (
        "vix_scale",
        "band",
        "band_vix_scale",
        "pair_vol_scale",
        "risk_exponent",
        "risk_multiplier_cap",
    ):
        first_params[key] = float(first_params[key])
    for key in ("pair_vol_window", "pair_vol_rank_window"):
        first_params[key] = int(first_params[key])
    seed = seed_returns(
        data,
        selection,
        first_params,
        args.start_year,
        args.train_years,
        args.inner_test_years,
    )

    if args.retrain_exits:
        context = load_context(
            end=pd.Timestamp(year=args.end_year, month=12, day=31)
        )
        vix_scales = {
            int(row.test_year): float(row.vix_scale)
            for row in chosen.itertuples(index=False)
        }
        pool_checkpoints = pool_checkpoints_low_vol_boost(
            context,
            vix_scales,
            risk_exponent=args.risk_exponent,
            risk_multiplier_cap=args.risk_multiplier_cap,
            vol_window=args.pair_vol_window,
            reference_window=args.pair_vol_rank_window,
        )
        exit_models = train_ridge_weights(pool_checkpoints, years)
        save_ridge_weights(exit_models, args.v6_exit_weights)
        print(
            f"Saved V6-trained annual exit weights to {args.v6_exit_weights}",
            flush=True,
        )
        weights_path = args.v6_exit_weights
    else:
        exit_models = load_ridge_weights(args.exit_weights)
        context = load_context()
        weights_path = args.exit_weights
    missing = [year for year in years if str(year) not in exit_models]
    if missing:
        raise RuntimeError(f"{weights_path} has no exit-classifier weights for {missing}.")

    args.output_dir.mkdir(parents=True, exist_ok=True)
    chosen.to_csv(args.output_dir / "selected_parameters.csv", index=False)
    if not scores.empty:
        score_name = (
            "low_vol_boost_vix_scores.csv"
            if args.retune_vix_scale
            else (
                "low_vol_boost_exponent_scores.csv"
                if args.tune_risk_exponent
                else "pair_vol_scale_scores.csv"
            )
        )
        scores.to_csv(args.output_dir / score_name, index=False)
    runs = (
        ("gross", 0.0, 0.0),
        (
            f"net_{args.net_cost_bps:g}bps_borrow_{args.borrow_bps:g}",
            args.net_cost_bps,
            args.borrow_bps,
        ),
    )
    summaries = []
    for label, cost, borrow in runs:
        v6_costed = apply_positions_to_returns(v6_gross, transaction_cost_bps=cost)
        traded_cp, _ = traded_checkpoints(context, v6_costed)
        decisions = ridge_decisions(exit_models, traded_cp, years)
        filtered = apply_positions_to_returns(
            apply_exit_targets(v6_costed, decisions, EXIT_RULE),
            transaction_cost_bps=cost,
        )
        daily = levered(
            filtered,
            seed,
            MODEL_NAME,
            cost,
            args.target_vol,
            args.max_leverage,
            args.vol_halflife,
        )
        daily = with_borrow(daily, borrow)
        daily.to_csv(args.output_dir / f"daily_results_{label}.csv", index=False)
        decisions.to_csv(args.output_dir / f"exit_decisions_{label}.csv", index=False)
        summaries.append({"scenario": label, **summarize(daily)})
        print(
            f"{label}: {summarize(daily)} "
            f"(target vol={args.target_vol:.1%}, max leverage={args.max_leverage:g}x)",
            flush=True,
        )
    pd.DataFrame(summaries).to_csv(args.output_dir / "summary.csv", index=False)


if __name__ == "__main__":
    main()
