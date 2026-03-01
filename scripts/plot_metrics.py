#!/usr/bin/env python3
"""Create visualizations from artifacts produced by the test scripts.

Produces PNG files into `artifacts/figs/`:
- equity_curve_{strategy}_net.png  (net returns)
- equity_curve_{strategy}_gross.png
- daily_hist_{strategy}.png
- trades_pnl_{strategy}.png
- spread_with_trades_{strategy}.png
"""
import json
from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt
import matplotlib.ticker as mtick
import numpy as np
import seaborn as sns


ART = Path('artifacts')
FIGS = ART / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)


def load_series(path):
    s = pd.read_csv(path, index_col=0, header=None, squeeze=True)
    s.index = pd.to_datetime(s.index, errors='ignore')
    return s


def plot_equity(ret_series, outpath, title):
    eq = (1 + ret_series).cumprod()
    plt.figure(figsize=(8,4))
    sns.lineplot(x=eq.index, y=eq.values)
    plt.title(title)
    plt.ylabel('Equity (gross)')
    # show y-axis as percent change from start for easier comparison
    start = eq.iloc[0] if len(eq) > 0 else 1.0
    rel = (eq / start - 1) * 100
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    # replace plotted line with percent-relative values for clearer scaling
    plt.clf()
    plt.figure(figsize=(8,4))
    sns.lineplot(x=rel.index, y=rel.values)
    plt.title(title)
    plt.ylabel('Cumulative return (%)')
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_hist(ret_series, outpath, title):
    # Plot daily returns in percent, excluding exact zeros for clarity
    vals = (ret_series.dropna() * 100)
    if vals.empty:
        return

    is_zero = np.isclose(vals, 0.0)
    pct_zeros = 100.0 * np.sum(is_zero) / len(vals)
    nonzero_vals = vals[~is_zero]

    if nonzero_vals.size == 0:
        # all zeros — show a simple message plot
        plt.figure(figsize=(6,4))
        plt.text(0.5, 0.5, 'All daily returns are 0%', ha='center', va='center')
        plt.title(f"{title} (zeros={pct_zeros:.1f}%)")
        plt.tight_layout()
        plt.savefig(outpath)
        plt.close()
        return

    low, high = np.percentile(nonzero_vals, [1, 99])
    margin = max(0.1 * (high - low), 0.01)
    x_min, x_max = low - margin, high + margin

    plt.figure(figsize=(6,4))
    sns.histplot(nonzero_vals.clip(x_min, x_max), bins=50, kde=True)
    plt.title(f"{title} (zeros={pct_zeros:.1f}%)")
    plt.xlabel('Daily return (%)')
    plt.xlim(x_min, x_max)
    plt.gca().xaxis.set_major_formatter(mtick.PercentFormatter())
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_trades_pnl(trades_df, outpath, title):
    if trades_df.empty:
        return
    plt.figure(figsize=(6,4))
    sns.histplot(trades_df['pnl'], bins=40)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def plot_spread_with_trades(spread_series, trades_df, outpath, title):
    plt.figure(figsize=(10,4))
    plt.plot(spread_series.index, spread_series.values, label='spread')
    if not trades_df.empty:
        for _, r in trades_df.iterrows():
            entry = int(r['entry'])
            exit = int(r['exit'])
            plt.axvline(spread_series.index[entry], color='g', alpha=0.6)
            plt.axvline(spread_series.index[exit], color='r', alpha=0.6)
    plt.title(title)
    plt.tight_layout()
    plt.savefig(outpath)
    plt.close()


def main():
    with open(ART / 'metrics_summary.json') as f:
        metrics = json.load(f)

    strategies = list(metrics.keys())

    for strat in strategies:
        # net returns
        net_path = ART / f"rets_{strat}.csv"
        gross_path = ART / f"rets_gross_{strat}.csv"
        trades_path = ART / f"trades_{strat}.csv"
        spread_path = ART / f"spread_{strat}.csv"

        if net_path.exists():
            net = pd.read_csv(net_path, index_col=0, header=0).iloc[:, 0]
        else:
            net = pd.Series(dtype=float)

        if gross_path.exists():
            gross = pd.read_csv(gross_path, index_col=0, header=0).iloc[:, 0]
        else:
            gross = pd.Series(dtype=float)

        trades_df = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()

        # spread
        if spread_path.exists():
            spread = pd.read_csv(spread_path, index_col=0, header=None).iloc[:, 0]
            spread.index = pd.RangeIndex(start=0, stop=len(spread))
        else:
            spread = pd.Series(dtype=float)

        # Equity curves
        if not net.empty:
            plot_equity(net, FIGS / f"equity_net_{strat}.png", f"Equity (net) - {strat}")
            plot_hist(net, FIGS / f"daily_hist_{strat}.png", f"Daily returns - {strat}")
        if not gross.empty:
            plot_equity(gross, FIGS / f"equity_gross_{strat}.png", f"Equity (gross) - {strat}")

        # Trades PnL
        plot_trades_pnl(trades_df, FIGS / f"trades_pnl_{strat}.png", f"Trades PnL - {strat}")

        # Spread with trade markers
        plot_spread_with_trades(spread, trades_df, FIGS / f"spread_trades_{strat}.png", f"Spread and trades - {strat}")

        print(f"Plotted {strat}")


if __name__ == '__main__':
    main()
