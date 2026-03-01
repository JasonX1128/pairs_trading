#!/usr/bin/env python3
"""Recompute gross returns (zero fees) and export trade-level PnL tables."""
import sys
from pathlib import Path
import numpy as np
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hmm import CrudeOilArbitrageHMM


def generate_synthetic_prices(T=400, seed=0):
    np.random.seed(seed)
    base = np.cumsum(np.random.normal(0, 0.5, size=T)) + 50
    p1 = base + np.random.normal(0, 0.1, size=T)
    p2 = 2.0 * base + np.random.normal(0, 0.2, size=T)
    p3 = 0.5 * base + np.random.normal(0, 0.15, size=T)
    df = pd.DataFrame({'Brent': p1, 'Shanghai': p2, 'WTI': p3})
    return df


def compute_returns_no_fees(signals, prices, lambda_vec, lambda_0):
    daily_rets = []
    for t in range(1, len(signals)):
        g_t_prev = abs(lambda_0) + np.sum(np.abs(lambda_vec) * prices.iloc[t-1])
        p_delta = prices.iloc[t] - prices.iloc[t-1]
        pnl = signals[t-1] * np.sum(lambda_vec * p_delta)
        fee = 0.0
        daily_rets.append((pnl - fee) / g_t_prev)
    return pd.Series(daily_rets)


def extract_trades(signals, prices, lambda_vec):
    trades = []
    pos = signals[0]
    entry_idx = None
    entry_price = None
    for t in range(1, len(signals)):
        if pos == 0 and signals[t] != 0:
            # entry
            entry_idx = t
            entry_price = prices.iloc[t]
            pos = signals[t]
        elif pos != 0 and signals[t] == 0:
            # exit
            exit_idx = t
            exit_price = prices.iloc[t]
            pnl = pos * np.sum(lambda_vec * (exit_price - entry_price))
            trades.append({'entry': entry_idx, 'exit': exit_idx, 'pnl': float(pnl), 'duration': exit_idx - entry_idx})
            pos = 0
            entry_idx = None
            entry_price = None
    return pd.DataFrame(trades)


def main():
    out = Path('artifacts')
    out.mkdir(exist_ok=True)

    prices = generate_synthetic_prices(T=400)
    # Save synthetic prices for inspection
    prices.to_csv(out / 'prices.csv', index=True)
    model = CrudeOilArbitrageHMM()

    strategies = ['PV', 'ProbI', 'PredI', 'RI', 'PI']

    for strat in strategies:
        try:
            signals, spread = model.run_backtest(strat, prices)
        except Exception as e:
            print(f"{strat} error: {e}")
            continue

        rets_gross = compute_returns_no_fees(signals, prices, model.lambda_vec, model.lambda_0)
        rets_gross.to_csv(out / f"rets_gross_{strat}.csv", index=True)

        trades_df = extract_trades(signals, prices, model.lambda_vec)
        trades_df.to_csv(out / f"trades_{strat}.csv", index=False)

        # print quick summary
        print(f"{strat}: gross_cum={rets_gross.add(1).cumprod().iloc[-1]-1:.4f}, n_trades={len(trades_df)}")


if __name__ == '__main__':
    main()
