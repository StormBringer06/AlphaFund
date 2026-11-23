"""V7 adds a lagged VIX-floor trading gate to the consistent V6 strategy."""

from __future__ import annotations

import argparse
import math
from pathlib import Path

import numpy as np
import pandas as pd

from backtest import apply_positions_to_returns
from improved_models.vol_scaled_naive_model_v6 import (
    levered,
    seed_returns,
    summarize,
    with_borrow,
)


MODEL_NAME = "v7_vix_floor"
DEFAULT_INPUT = Path("data") / "pit_2008_2026" / "pit_pair_prices.csv"
DEFAULT_SELECTION = Path("data") / "pit_2008_2026" / "pit_pair_selection.csv"
V6_RUNS = (
    (
        2008,
        2017,
        Path("backtest_results") / "v6_low_vol_boost_gamma_1_2008_2017",
    ),
    (
        2018,
        2026,
        Path("backtest_results")
        / "v6_holdout_low_vol_boost_gamma_1_frozen_v5_exits_2018_2026",
    ),
)
MODEL_PARAMETER_COLUMNS = (
    "vix_scale",
    "band",
    "band_vix_scale",
    "pair_vol_scale",
    "pair_vol_mode",
    "risk_exponent",
    "risk_multiplier_cap",
    "pair_vol_window",
    "pair_vol_rank_window",
)


def apply_vix_floor_to_targets(
    daily: pd.DataFrame,
    vix_floor: float = 10.0,
    remove_leverage: bool = False,
) -> pd.DataFrame:
    """Zero target weights when close-t VIX is below the floor.

    The backtest engine applies each date's targets to the next date's returns,
    so using the close-t VIX here makes the filter available without look-ahead.
    """
    if not math.isfinite(vix_floor) or vix_floor < 0:
        raise ValueError("vix_floor must be finite and non-negative.")
    required = {"vix", "target_weight_a", "target_weight_b"}
    missing = required.difference(daily.columns)
    if missing:
        raise ValueError(
            "Daily positions are missing columns: " + ", ".join(sorted(missing)) + "."
        )

    out = daily.copy()
    vix = pd.to_numeric(out["vix"], errors="coerce")
    if vix.isna().any() or not np.isfinite(vix.to_numpy()).all():
        raise ValueError("VIX values must be finite.")
    if remove_leverage:
        if "leverage" not in out:
            raise ValueError("leverage is required to remove prior leverage.")
        leverage = pd.to_numeric(out["leverage"], errors="coerce")
        if leverage.isna().any() or not np.isfinite(leverage.to_numpy()).all():
            raise ValueError("leverage values must be finite.")
        if (leverage <= 0).any():
            raise ValueError("leverage values must be positive.")
        out["target_weight_a"] = out["target_weight_a"] / leverage
        out["target_weight_b"] = out["target_weight_b"] / leverage

    gate_open = vix >= vix_floor
    out.loc[~gate_open, ["target_weight_a", "target_weight_b"]] = 0.0
    out["vix_gate_open"] = gate_open
    for column in ("target_signal", "signal", "target_gross_exposure"):
        if column in out:
            out.loc[~gate_open, column] = 0.0
    return out


def _first_year_parameters(selected_path: Path, first_year: int) -> dict:
    selected = pd.read_csv(selected_path)
    row = selected.loc[selected["test_year"] == first_year]
    if len(row) != 1:
        raise ValueError(
            f"Expected one selected-parameter row for {first_year} in {selected_path}."
        )
    defaults = {
        "risk_multiplier_cap": 1.5,
        "pair_vol_window": 60,
        "pair_vol_rank_window": 504,
    }
    values = row.iloc[0].to_dict()
    params = {
        key: values.get(key, defaults.get(key))
        for key in MODEL_PARAMETER_COLUMNS
    }
    for key in (
        "vix_scale",
        "band",
        "band_vix_scale",
        "pair_vol_scale",
        "risk_exponent",
        "risk_multiplier_cap",
    ):
        params[key] = float(params[key])
    for key in ("pair_vol_window", "pair_vol_rank_window"):
        params[key] = int(params[key])
    params["pair_vol_mode"] = str(params["pair_vol_mode"])
    return params


def run_period(
    v6_run_dir: Path,
    output_dir: Path,
    first_year: int,
    last_year: int,
    data: pd.DataFrame,
    selection: pd.DataFrame,
    vix_floor: float = 10.0,
    train_years: int = 4,
    inner_test_years: int = 2,
    target_vol: float = 0.08,
    max_leverage: float = 4.0,
    vol_halflife: float = 60.0,
    net_cost_bps: float = 5.0,
    borrow_bps: float = 50.0,
) -> pd.DataFrame:
    v6_daily_path = v6_run_dir / "daily_results_gross.csv"
    selected_path = v6_run_dir / "selected_parameters.csv"
    if not v6_daily_path.exists() or not selected_path.exists():
        raise FileNotFoundError(f"Missing V6 source outputs under {v6_run_dir}.")

    params = _first_year_parameters(selected_path, first_year)
    seed = seed_returns(
        data,
        selection,
        params,
        first_year,
        train_years,
        inner_test_years,
    )
    seed = apply_vix_floor_to_targets(seed, vix_floor)
    output_dir.mkdir(parents=True, exist_ok=True)
    low_vix_days = None
    rows = []
    for label, cost, borrow in (
        ("gross", 0.0, 0.0),
        (f"net_{net_cost_bps:g}bps_borrow_{borrow_bps:g}", net_cost_bps, borrow_bps),
    ):
        source_label = (
            "gross" if cost == 0.0
            else f"net_{net_cost_bps:g}bps_borrow_{borrow_bps:g}"
        )
        source_path = v6_run_dir / f"daily_results_{source_label}.csv"
        if not source_path.exists():
            raise FileNotFoundError(f"Missing V6 source results: {source_path}.")
        v6_daily = pd.read_csv(source_path, parse_dates=["date"], low_memory=False)
        v7_unlevered = apply_vix_floor_to_targets(
            v6_daily,
            vix_floor,
            remove_leverage=True,
        )
        if low_vix_days is None:
            low_vix_days = int(
                v7_unlevered.loc[
                    ~v7_unlevered["vix_gate_open"], "date"
                ].nunique()
            )
        prelevered = apply_positions_to_returns(v7_unlevered, cost)
        daily = levered(
            prelevered,
            seed,
            MODEL_NAME,
            cost,
            target_vol,
            max_leverage,
            vol_halflife,
        )
        daily = with_borrow(daily, borrow)
        daily.to_csv(output_dir / f"daily_results_{label}.csv", index=False)
        rows.append({"scenario": label, **summarize(daily)})
    summary = pd.DataFrame(rows)
    summary.to_csv(output_dir / "summary.csv", index=False)
    pd.DataFrame(
        [{"first_year": first_year, "last_year": last_year, "vix_floor": vix_floor,
          "low_vix_days": low_vix_days}]
    ).to_csv(output_dir / "filter_summary.csv", index=False)
    print(
        f"{first_year}-{last_year}: VIX<{vix_floor:g} on {low_vix_days} target dates; "
        f"{summary.to_dict(orient='records')}",
        flush=True,
    )
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=DEFAULT_INPUT)
    parser.add_argument("--selection", type=Path, default=DEFAULT_SELECTION)
    parser.add_argument("--vix-floor", type=float, default=10.0)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--inner-test-years", type=int, default=2)
    parser.add_argument("--target-vol", type=float, default=0.08)
    parser.add_argument("--max-leverage", type=float, default=4.0)
    parser.add_argument("--vol-halflife", type=float, default=60.0)
    parser.add_argument("--net-cost-bps", type=float, default=5.0)
    parser.add_argument("--borrow-bps", type=float, default=50.0)
    args = parser.parse_args()

    if not math.isfinite(args.vix_floor) or args.vix_floor < 0:
        parser.error("--vix-floor must be finite and non-negative.")
    data = pd.read_csv(args.input, parse_dates=["date"])
    selection = pd.read_csv(args.selection)
    floor_tag = f"{args.vix_floor:g}".replace(".", "p")
    summaries = []
    for first_year, last_year, v6_dir in V6_RUNS:
        output_dir = (
            Path("backtest_results")
            / f"v7_vix_floor_{floor_tag}_{first_year}_{last_year}"
        )
        summary = run_period(
            v6_dir,
            output_dir,
            first_year,
            last_year,
            data,
            selection,
            args.vix_floor,
            args.train_years,
            args.inner_test_years,
            args.target_vol,
            args.max_leverage,
            args.vol_halflife,
            args.net_cost_bps,
            args.borrow_bps,
        )
        summary.insert(0, "period", f"{first_year}-{last_year}")
        summaries.append(summary)
    pd.concat(summaries, ignore_index=True).to_csv(
        Path("backtest_results") / f"v7_vix_floor_{floor_tag}_comparison.csv",
        index=False,
    )


if __name__ == "__main__":
    main()
