#!/usr/bin/env python3
"""Create visualizations from artifacts produced by the test scripts.

Produces PNG files into `artifacts/figs/`:
- equity_curve_{strategy}_net.png
- equity_curve_{strategy}_gross.png
- daily_hist_{strategy}.png
- trades_pnl_{strategy}.png
- spread_with_trades_{strategy}.png
- drawdown_{strategy}.png
- lambda_hist_{strategy}.png
- hmm_probs_{strategy}.png
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


def plot_drawdown(ret_series, outpath, title):
    """Plots the underwater curve to show max drawdowns and durations."""
    if ret_series.empty:
        return
    
    cum = (1 + ret_series).cumprod()
    running_max = cum.cummax()
    dd = (cum / running_max) - 1

    plt.figure(figsize=(10, 4))
    plt.fill_between(dd.index, dd.values * 100, 0, color='#d7191c', alpha=0.5)
    plt.plot(dd.index, dd.values * 100, color='#d7191c', linewidth=1)
    plt.title(title, fontweight='bold')
    plt.ylabel('Drawdown (%)')
    plt.xlabel('Date')
    plt.gca().yaxis.set_major_formatter(mtick.PercentFormatter())
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
    vals = (ret_series.dropna() * 100)
    if vals.empty:
        return

    is_zero = np.isclose(vals, 0.0)
    pct_zeros = 100.0 * np.sum(is_zero) / len(vals)
    nonzero_vals = vals[~is_zero]

    if nonzero_vals.size == 0:
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


def plot_lambda_history(lambda_df, outpath, title):
    """Plots the rolling cointegration weights over time."""
    if lambda_df.empty:
        return
    plt.figure(figsize=(10, 5))
    
    # Exclude the intercept for visibility of the actual asset weights
    cols_to_plot = [c for c in lambda_df.columns if 'Intercept' not in c]
    for col in cols_to_plot:
        plt.plot(lambda_df.index, lambda_df[col], label=col)
        
    plt.title(title, fontweight='bold')
    plt.ylabel('Cointegration Weight')
    plt.xlabel('Date')
    try:
        plt.gca().xaxis.set_major_locator(mdates.AutoDateLocator())
        plt.gca().xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    except Exception:
        pass
    plt.xticks(rotation=45)
    plt.legend()
    plt.grid(True, alpha=0.25)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def plot_hmm_probs(spread_series, xhat_df, outpath, title):
    """Plots the spread in the top panel and the HMM state probabilities in the bottom panel."""
    if spread_series.empty or xhat_df.empty:
        return
    
    fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 7), sharex=True, gridspec_kw={'height_ratios': [2, 1]})
    
    # Top Panel: Spread
    ax1.plot(spread_series.index, spread_series.values, color='black', alpha=0.7)
    ax1.set_title(title, fontweight='bold')
    ax1.set_ylabel('Spread')
    ax1.grid(True, alpha=0.25)
    
    # Bottom Panel: Probability of State 0
    state0 = xhat_df.iloc[:, 0]
    ax2.fill_between(state0.index, state0.values, 0, color='#2b83ba', alpha=0.5, label='State 0 Prob')
    ax2.plot(state0.index, state0.values, color='#2b83ba', linewidth=1)
    ax2.set_ylabel('Probability')
    ax2.set_xlabel('Date')
    ax2.set_ylim(0, 1)
    ax2.grid(True, alpha=0.25)
    ax2.legend(loc='upper right')
    
    try:
        ax2.xaxis.set_major_locator(mdates.AutoDateLocator())
        ax2.xaxis.set_major_formatter(mdates.ConciseDateFormatter(mdates.AutoDateLocator()))
    except Exception:
        pass
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(outpath, dpi=300)
    plt.close()


def main():
    with open(ART / 'metrics_summary.json') as f:
        metrics = json.load(f)

    strategies = list(metrics.keys())

    for strat in strategies:
        net_path = ART / f"rets_{strat}.csv"
        gross_path = ART / f"rets_gross_{strat}.csv"
        trades_path = ART / f"trades_{strat}.csv"
        spread_path = ART / f"spread_{strat}.csv"
        lambda_path = ART / f"lambda_{strat}.csv"
        xhat_path = ART / f"xhat_{strat}.csv"

        net = pd.read_csv(net_path, index_col=0, header=0, parse_dates=True).iloc[:, 0] if net_path.exists() else pd.Series(dtype=float)
        gross = pd.read_csv(gross_path, index_col=0, header=0, parse_dates=True).iloc[:, 0] if gross_path.exists() else pd.Series(dtype=float)
        trades_df = pd.read_csv(trades_path) if trades_path.exists() else pd.DataFrame()
        spread = pd.read_csv(spread_path, index_col=0, header=0, parse_dates=True).iloc[:, 0] if spread_path.exists() else pd.Series(dtype=float)
        lambda_df = pd.read_csv(lambda_path, index_col=0, header=0, parse_dates=True) if lambda_path.exists() else pd.DataFrame()
        xhat_df = pd.read_csv(xhat_path, index_col=0, header=0, parse_dates=True) if xhat_path.exists() else pd.DataFrame()

        # Core Plots
        if not net.empty:
            plot_equity(net, FIGS / f"equity_net_{strat}.png", f"Equity (net) - {strat}")
            plot_drawdown(net, FIGS / f"drawdown_{strat}.png", f"Underwater Curve - {strat}")
            plot_hist(net, FIGS / f"daily_hist_{strat}.png", f"Daily returns - {strat}")
            
        if not gross.empty:
            plot_equity(gross, FIGS / f"equity_gross_{strat}.png", f"Equity (gross) - {strat}")

        plot_trades_pnl(trades_df, FIGS / f"trades_pnl_{strat}.png", f"Trades PnL - {strat}")
        plot_spread_with_trades(spread, trades_df, FIGS / f"spread_trades_{strat}.png", f"Spread and trades - {strat}")

        # HMM and Cointegration Analytics
        if not lambda_df.empty:
            plot_lambda_history(lambda_df, FIGS / f"lambda_hist_{strat}.png", f"Rolling Cointegration Weights - {strat}")
            
        if not spread.empty and not xhat_df.empty:
            plot_hmm_probs(spread, xhat_df, FIGS / f"hmm_probs_{strat}.png", f"HMM Regime Probabilities - {strat}")

        print(f"Plotted {strat}")

if __name__ == '__main__':
    main()