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
import matplotlib.dates as mdates
import numpy as np
import seaborn as sns


ART = Path('artifacts')
FIGS = ART / 'figs'
FIGS.mkdir(parents=True, exist_ok=True)


def load_series(path):
    s = pd.read_csv(path, index_col=0, header=0, parse_dates=True).iloc[:, 0]
    return s


def plot_equity(ret_series, outpath, title):
    if ret_series.empty:
        return

    eq = (1 + ret_series).cumprod()
    start = eq.iloc[0] if len(eq) > 0 else 1.0
    rel = (eq / start - 1) * 100

    plt.figure(figsize=(10, 5))
    sns.lineplot(x=rel.index, y=rel.values, color='#2b83ba')
    plt.title(title, fontweight='bold')
    plt.ylabel('Cumulative return (%)')
    plt.xlabel('Date')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
    # format x-axis as dates
    try:
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    except Exception:
        pass
    plt.xticks(rotation=45)
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
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
    if spread_series.empty:
        return
    plt.figure(figsize=(12,5))
    plt.plot(spread_series.index, spread_series.values, label='HMM Spread', color='black', alpha=0.6)
    if not trades_df.empty:
        for _, r in trades_df.iterrows():
            try:
                entry = int(r['entry'])
                exit = int(r['exit'])
                entry_dt = spread_series.index[entry]
                exit_dt = spread_series.index[exit]
                plt.axvspan(entry_dt, exit_dt, color='green' if r.get('pnl', 0) > 0 else 'red', alpha=0.2)
            except Exception:
                continue
    plt.title(title, fontweight='bold')
    plt.xlabel('Date')
    plt.ylabel('Spread')
    try:
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    except Exception:
        pass
    plt.xticks(rotation=45)
    plt.legend()
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    with open(ART / 'metrics_summary.json') as f:
        metrics = json.load(f)

    strategies = list(metrics.keys())

    for strat in strategies:
        # paths: these CSVs should have a datetime index saved
        net_path = ART / f"rets_{strat}.csv"
        gross_path = ART / f"rets_gross_{strat}.csv"
        trades_path = ART / f"trades_{strat}.csv"
        spread_path = ART / f"spread_{strat}.csv"

        if net_path.exists():
            net = pd.read_csv(net_path, index_col=0, header=0, parse_dates=True).iloc[:, 0]
        else:
            net = pd.Series(dtype=float)

        if gross_path.exists():
            gross = pd.read_csv(gross_path, index_col=0, header=0, parse_dates=True).iloc[:, 0]
        else:
            gross = pd.Series(dtype=float)

        trades_df = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()

        # spread (keep original dates if present)
        if spread_path.exists():
            spread = pd.read_csv(spread_path, index_col=0, header=0, parse_dates=True).iloc[:, 0]
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
