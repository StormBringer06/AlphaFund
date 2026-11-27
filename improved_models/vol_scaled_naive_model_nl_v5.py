"""V5: ciaranMaxModel (V3 + walk-forward exit classifier) with V4's dynamic leverage.

1. Signal: ``ciaranMaxModel`` unchanged - BD's V3 (EWMA z-score, VIX overlay,
   band 0.2, top 5 pairs from the 10-year point-in-time screen) plus Ciaran's
   ridge exit classifier, which closes an open episode at a checkpoint (days
   5-120) when its predicted remaining P&L is negative. The yearly classifier
   weights are loaded from Ciaran's ``models/exit_ridge_weights.json`` (each
   year trained on pool trades resolved before 1 Jan of that year); V5 never
   refits them. See ``improved_models/ciaranMaxModel.py`` and
   ``improved_models/exit_model.py``.
2. Leverage: V4's dynamic portfolio volatility target. Every weight is scaled
   by ``min(target_vol / ewma_vol, max_leverage)``, with ``ewma_vol`` the EWMA
   (``--vol-halflife``) volatility of V5's own unlevered equal-weight portfolio
   returns up to that close, applied to the next day. The EWMA is seeded with
   V3's inner walk-forward returns for the two years before the first test year
   (the classifier has no out-of-sample decisions there), which only affects
   leverage in the first weeks of 2014.

Reported for comparison on the same pairs: V3, ciaranMaxModel (unlevered) and
V3 with the same dynamic leverage (= ``v4_dynamic_vol``). Gross of fees first;
the net table uses ``--net-cost-bps`` trading costs, with exit decisions made on
net P&L as in ciaranMaxModel, plus ``--borrow-bps`` per year on the short leg.
"""

from __future__ import annotations

import argparse
from pathlib import Path
import warnings

import pandas as pd

from backtest import apply_positions_to_returns, run_walk_forward_backtest
from improved_models.ciaranMaxModel import EXIT_RULE, apply_exit_targets, run_v3
from improved_models.exit_model import (
    RIDGE_WEIGHTS_PATH, load_context, load_ridge_weights, ridge_decisions, traded_checkpoints,
)
from improved_models.regression_model import portfolio_metrics
from improved_models.vol_scaled_naive_model_nl import VolScaledNaivePairsTradingModel, robustness_table
from improved_models.vol_scaled_naive_model_nl_v4 import dynamic_leverage, lever, portfolio_returns


def seed_returns(
    data: pd.DataFrame, selection: pd.DataFrame, params: dict, first_year: int,
    train_years: int, inner_test_years: int,
) -> pd.DataFrame:
    """V3's inner walk-forward daily results for the years before ``first_year``."""
    pairs = selection.loc[selection["test_year"] == first_year, "pair_id"].unique()
    year_data = data.loc[data["pair_id"].isin(pairs)]
    training = year_data.loc[
        (year_data["date"] >= pd.Timestamp(year=first_year - train_years, month=1, day=1))
        & (year_data["date"] < pd.Timestamp(year=first_year, month=1, day=1))
    ]
    return run_walk_forward_backtest(
        training,
        model_factory=lambda: VolScaledNaivePairsTradingModel(**params),
        start_year=first_year - inner_test_years,
        end_year=first_year - 1,
        train_years=train_years - inner_test_years,
        transaction_cost_bps=0.0,
    )


def levered(daily: pd.DataFrame, seed: pd.DataFrame, name: str, cost: float, args) -> pd.DataFrame:
    """Apply dynamic leverage computed from ``daily``'s own unlevered returns."""
    history = pd.concat([
        portfolio_returns(apply_positions_to_returns(seed, transaction_cost_bps=cost)),
        portfolio_returns(daily),
    ]).sort_index()
    history = history[~history.index.duplicated(keep="last")]
    dates = pd.Index(sorted(daily["date"].unique()))
    leverage = dynamic_leverage(
        history, args.target_vol, args.max_leverage, args.vol_halflife
    ).reindex(dates)
    return lever(daily, leverage, name, cost)


def with_borrow(daily: pd.DataFrame, borrow_bps: float) -> pd.DataFrame:
    out = daily.copy()
    borrow = out["position_gross_exposure"] * 0.5 * borrow_bps / 10_000.0 / 252.0
    out["strategy_return"] = out["strategy_return"] - borrow
    return out


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    parser.add_argument("--input", type=Path, default=Path("data") / "pit_pair_prices.csv")
    parser.add_argument("--selection", type=Path, default=Path("data") / "pit_pair_selection.csv")
    parser.add_argument("--start-year", type=int, default=2014)
    parser.add_argument("--end-year", type=int, default=2023)
    parser.add_argument("--train-years", type=int, default=4)
    parser.add_argument("--inner-test-years", type=int, default=2)
    parser.add_argument("--target-vol", type=float, default=0.08)
    parser.add_argument("--max-leverage", type=float, default=4.0)
    parser.add_argument("--vol-halflife", type=float, default=60.0)
    parser.add_argument("--net-cost-bps", type=float, default=5.0)
    parser.add_argument("--borrow-bps", type=float, default=50.0, help="Per year, on the short leg.")
    parser.add_argument(
        "--freeze-exit-model", type=int, default=None, metavar="YEAR",
        help="Use the saved exit model of YEAR for every test year (e.g. 2023 for a "
             "2024+ holdout, so no weights are refit or chosen after the holdout starts).",
    )
    parser.add_argument("--output-dir", type=Path, default=Path("backtest_results") / "vol_scaled_naive_v5")
    args = parser.parse_args()
    warnings.simplefilter("ignore")

    data = pd.read_csv(args.input, parse_dates=["date"])
    selection = pd.read_csv(args.selection)
    years = range(args.start_year, args.end_year + 1)

    print("Running V3 ...", flush=True)
    v3_gross, params = run_v3(data, selection, years, args.train_years)
    first_params = params.iloc[0][["vix_scale", "band", "band_vix_scale"]].astype(float).to_dict()
    seed = seed_returns(
        data, selection, first_params, args.start_year, args.train_years, args.inner_test_years
    )

    # Ciaran's trained exit models; never refit here.
    exit_models = load_ridge_weights()
    if args.freeze_exit_model is not None:
        frozen = str(args.freeze_exit_model)
        if frozen not in exit_models:
            raise RuntimeError(f"{RIDGE_WEIGHTS_PATH} has no exit model for {frozen}.")
        exit_models = {str(y): exit_models[frozen] for y in years}
        print(f"Using the {frozen} exit model for every year", flush=True)
    missing = [y for y in years if str(y) not in exit_models]
    if missing:
        raise RuntimeError(
            f"{RIDGE_WEIGHTS_PATH} has no exit model for {missing}. Train them with "
            "`python -m improved_models.ciaranMaxModel` (it appends missing years)."
        )
    print(f"Loaded exit-classifier weights from {RIDGE_WEIGHTS_PATH}", flush=True)
    ctx = load_context()

    args.output_dir.mkdir(parents=True, exist_ok=True)
    tables = {}
    runs = (("gross", 0.0, 0.0), (f"net_{args.net_cost_bps:g}bps_borrow_{args.borrow_bps:g}",
                                   args.net_cost_bps, args.borrow_bps))
    for label, cost, borrow in runs:
        print(f"Exit decisions and leverage ({label}) ...", flush=True)
        v3 = apply_positions_to_returns(v3_gross, transaction_cost_bps=cost)
        traded_cp, _ = traded_checkpoints(ctx, v3)
        decisions = ridge_decisions(exit_models, traded_cp, years)
        filtered = apply_positions_to_returns(
            apply_exit_targets(v3, decisions, EXIT_RULE), transaction_cost_bps=cost
        )
        daily = pd.concat([
            v3.assign(model_name="v3"),
            filtered.assign(model_name="ciaranMaxModel"),
            levered(v3, seed, "v3_dynamic_vol (=v4)", cost, args),
            levered(filtered, seed, "v5_ciaranMax_dynamic_vol", cost, args),
        ], ignore_index=True)
        daily = with_borrow(daily, borrow)
        table = portfolio_metrics(daily)
        lev = daily.groupby("model_name")["leverage"].mean()
        table["average_leverage"] = table["model_name"].map(lev).fillna(1.0)
        robust = robustness_table(daily, baseline="v3")
        table = table.merge(
            robust[["model_name", "sharpe_ex_2020_2022", "sharpe_p05", "sharpe_p95",
                    "prob_beats_baseline"]],
            on="model_name",
        )
        tables[label] = table
        table.to_csv(args.output_dir / f"comparison_{label}.csv", index=False)
        daily.to_csv(args.output_dir / f"daily_results_{label}.csv", index=False)
        decisions.to_csv(args.output_dir / f"exit_decisions_{label}.csv", index=False)
        yearly = daily.groupby(["model_name", daily["date"].dt.year])["strategy_return"].apply(
            lambda r: r.groupby(daily.loc[r.index, "date"]).mean().add(1).prod() - 1
        ).unstack(0)
        yearly.to_csv(args.output_dir / f"yearly_{label}.csv")
    params.to_csv(args.output_dir / "v3_selected_parameters.csv", index=False)

    with pd.option_context("display.width", 260, "display.precision", 3):
        for label, table in tables.items():
            print(f"\n== {label} (target vol {args.target_vol:.0%}, max leverage {args.max_leverage:g}x) ==")
            print(table.to_string(index=False))


if __name__ == "__main__":
    main()
