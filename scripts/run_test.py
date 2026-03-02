#!/usr/bin/env python3
"""Run smoke tests for `hmm.py` strategies and optionally produce metrics files.

Usage:
  python3 scripts/run_test.py --metrics
"""
import argparse
import json
import sys
import numpy as np
import pandas as pd
from pathlib import Path

# Ensure project root is on sys.path so we can import `hmm` when running from scripts/
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

from hmm import CrudeOilArbitrageHMM, evaluate_performance

def generate_synthetic_prices(T=300, seed=0):
    np.random.seed(seed)
    base = np.cumsum(np.random.normal(0, 0.5, size=T)) + 50
    p1 = base + np.random.normal(0, 0.1, size=T)
    p2 = 2.0 * base + np.random.normal(0, 0.2, size=T)
    p3 = 0.5 * base + np.random.normal(0, 0.15, size=T)
    dates = pd.date_range(end=pd.Timestamp.today(), periods=T, freq='D')
    df = pd.DataFrame({'Brent': p1, 'Shanghai': p2, 'WTI': p3}, index=dates)
    return df

def summarize_returns(ret_series):
    if ret_series.empty:
        return {}
    mean_daily = ret_series.mean()
    vol_daily = ret_series.std()
    ann_ret = mean_daily * 252
    ann_vol = vol_daily * np.sqrt(252)
    sharpe = ann_ret / ann_vol if ann_vol > 0 else np.nan
    cum = (1 + ret_series).cumprod().iloc[-1] - 1
    return {
        'cumulative_return': float(cum),
        'annualized_return': float(ann_ret),
        'annualized_vol': float(ann_vol),
        'sharpe': float(sharpe),
        'mean_daily': float(mean_daily),
        'vol_daily': float(vol_daily),
        'n_obs': int(len(ret_series)),
    }

def main(metrics=False, outdir='artifacts', prices_path=None):
    out = Path(outdir)
    out.mkdir(parents=True, exist_ok=True)

    prices = None
    if prices_path:
        try:
            df = pd.read_csv(prices_path, parse_dates=['date'])
            if 'Shanghai_USD' in df.columns:
                sh = df['Shanghai_USD']
            elif 'Shanghai_CNY' in df.columns and 'USDCNY' in df.columns:
                sh = df['Shanghai_CNY'] / df['USDCNY']
            else:
                raise ValueError('No Shanghai price column (Shanghai_USD or Shanghai_CNY+USDCNY)')

            if 'Brent' not in df.columns or 'WTI' not in df.columns:
                raise ValueError('Missing Brent or WTI columns')

            prices = pd.DataFrame({'Brent': df['Brent'].values, 'Shanghai': sh.values, 'WTI': df['WTI'].values}, index=df['date'])
            print(f"Loaded prices from {prices_path}")
        except Exception as e:
            print(f"Failed to load prices from {prices_path}: {e}. Falling back to synthetic data.")

    if prices is None:
        prices = generate_synthetic_prices(T=400)

    prices.to_csv(out / 'prices.csv', index=True)

    strategies = ['PV', 'ProbI', 'PredI', 'RI', 'PI']
    results = {}

    do_gross = getattr(main, '_do_gross', False)
    do_trades = getattr(main, '_do_trades', False)
    do_plot = getattr(main, '_do_plot', False)

    for strat in strategies:
        model = CrudeOilArbitrageHMM()
        try:
            signals, spread, lambda_history, x_hat_history = model.run_backtest(strat, prices)
        except Exception as e:
            import traceback
            traceback.print_exc()
            results[strat] = {'error': repr(e)}
            print(f"{strat}: error {repr(e)}")
            continue

        if metrics:
            rets = evaluate_performance(signals, prices, lambda_history)
            try:
                if isinstance(prices.index, pd.DatetimeIndex):
                    rets = pd.Series(rets)
                    if len(prices.index) >= len(rets) + 1:
                        rets.index = prices.index[1:1+len(rets)]
                    elif len(prices.index) == len(rets):
                        rets.index = prices.index
            except Exception:
                rets = pd.Series(rets)

            summary = summarize_returns(rets)
            trades = sum(1 for i in range(1, len(signals)) if signals[i] != signals[i-1])
            wins = (rets > 0).sum()
            win_rate = int(wins) / len(rets) if len(rets) > 0 else None
            summary.update({'total_trades': int(trades), 'win_rate': win_rate})

            rets.to_csv(out / f"rets_{strat}.csv", index=True)
            
            spread_series = pd.Series(spread)
            try:
                if isinstance(prices.index, pd.DatetimeIndex) and len(prices.index) >= len(spread_series):
                    spread_series.index = prices.index[:len(spread_series)]
            except Exception:
                pass
            spread_series.to_csv(out / f"spread_{strat}.csv", index=True)

            # --- NEW DATA EXPORTS ---
            # Save lambda history
            lambda_cols = [f'Weight_{i}' for i in range(len(lambda_history[0][0]))] + ['Intercept']
            lambda_df = pd.DataFrame([np.append(l[0], l[1]) for l in lambda_history], columns=lambda_cols)
            if isinstance(prices.index, pd.DatetimeIndex) and len(prices.index) >= len(lambda_df):
                lambda_df.index = prices.index[:len(lambda_df)]
            lambda_df.to_csv(out / f"lambda_{strat}.csv", index=True)
            
            # Save x_hat history
            xhat_df = pd.DataFrame(x_hat_history, columns=[f'State_{i}' for i in range(len(x_hat_history[0]))])
            if isinstance(prices.index, pd.DatetimeIndex) and len(prices.index) >= len(xhat_df):
                xhat_df.index = prices.index[:len(xhat_df)]
            xhat_df.to_csv(out / f"xhat_{strat}.csv", index=True)

            results[strat] = {'summary': summary}
            print(f"{strat}: trades={trades}, cum_return={summary['cumulative_return']:.4f}, sharpe={summary['sharpe']:.2f}")
        else:
            nonzero = sum(1 for x in signals if x != 0)
            results[strat] = {'signals_count': int(nonzero)}
            print(f"{strat}: signals_count={nonzero}")

        if do_gross:
            from scripts.gross_and_trades import compute_returns_no_fees
            rets_gross = compute_returns_no_fees(signals, prices, model.lambda_vec, model.lambda_0)
            rets_gross.to_csv(out / f"rets_gross_{strat}.csv", index=True)

        if do_trades:
            from scripts.gross_and_trades import extract_trades
            trades_df = extract_trades(signals, prices, model.lambda_vec)
            trades_df.to_csv(out / f"trades_{strat}.csv", index=False)

        if do_plot:
            pass

    with open(out / 'metrics_summary.json', 'w') as f:
        json.dump(results, f, indent=2)

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--metrics', action='store_true', help='Compute and save metrics (daily returns + summary)')
    parser.add_argument('--outdir', default='artifacts', help='Output directory for metrics')
    parser.add_argument('--prices', default=None, help='Path to CSV file with real prices (optional)')
    parser.add_argument('--save-gross', action='store_true', help='Also compute and save gross (no-fee) returns')
    parser.add_argument('--save-trades', action='store_true', help='Also extract and save per-trade PnL tables')
    parser.add_argument('--plot', action='store_true', help='Generate plots using artifacts/ files')
    args = parser.parse_args()
    
    setattr(main, '_do_gross', args.save_gross)
    setattr(main, '_do_trades', args.save_trades)
    setattr(main, '_do_plot', args.plot)
    main(metrics=args.metrics, outdir=args.outdir, prices_path=args.prices)

    if args.plot:
        try:
            from scripts.plot_metrics import main as plot_main
            plot_main()
        except Exception as e:
            print(f"Plotting failed: {e}")