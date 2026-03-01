import sys
from pathlib import Path
PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT))

import pandas as pd
import numpy as np
from hmm import CrudeOilArbitrageHMM

DATA='raw_data/crude_data_robust.csv'
df = pd.read_csv(DATA, parse_dates=['date'])
if 'Shanghai_USD' in df.columns:
    sh = df['Shanghai_USD']
else:
    sh = df['Shanghai_CNY'] / df['USDCNY']
prices = pd.DataFrame({'Brent': df['Brent'].values, 'Shanghai': sh.values, 'WTI': df['WTI'].values}, index=df['date'])

strategies=['PV','ProbI','PredI','RI','PI']

for strat in strategies:
    model = CrudeOilArbitrageHMM()
    signals, spread = model.run_backtest(strat, prices)
    lambda_vec = model.lambda_vec
    lambda_0 = model.lambda_0

    # compute gross returns as in compute_returns_no_fees
    rets = []
    dates = []
    for t in range(1, len(signals)):
        g_t_prev = abs(lambda_0) + np.sum(np.abs(lambda_vec) * prices.iloc[t-1])
        p_delta = prices.iloc[t] - prices.iloc[t-1]
        pnl = signals[t-1] * np.sum(lambda_vec * p_delta)
        rets.append((pnl) / g_t_prev)
        dates.append(prices.index[t])
    rets = pd.Series(rets, index=pd.DatetimeIndex(dates))
    eq = (1+rets).cumprod()
    ratio = eq / eq.shift(1)
    # find largest single-day jump
    idx = ratio.idxmax()
    pos = ratio.idxmax() and ratio.idxmax()
    pos_i = ratio.values.argmax()
    # gather info for that t: date, gross ret, pnl, fee (should be zero here), exposure
    date = idx
    gross_ret = rets.loc[date]
    # find t index in original price index
    t = list(prices.index).index(date)
    p_delta = prices.iloc[t] - prices.iloc[t-1]
    pnl = signals[t-1] * np.sum(lambda_vec * p_delta)
    # compute fee despite gross
    costs = np.array([5.80,53.71,20.24])/10000
    fee = np.sum(np.abs(signals[t] - signals[t-1]) * costs * prices.iloc[t] * np.abs(lambda_vec))
    g_prev = abs(lambda_0) + np.sum(np.abs(lambda_vec) * prices.iloc[t-1])

    print('---', strat, '---')
    print('Date of max gross equity jump:', date)
    print('Gross return on that date (%):', float(gross_ret)*100)
    print('Computed pnl (absolute):', float(pnl))
    print('Fee on that date (abs):', float(fee))
    print('Fee as pct of exposure (%):', float(-fee/g_prev*100))
    print('Gross exposure previous day:', float(g_prev))
    print('Signal before:', int(signals[t-1]), 'Signal after:', int(signals[t]))
    print('Spread at t-1, t:', float(spread[t-1]), float(spread[t]))
    print('Price delta:', p_delta.to_dict())
    print('Cumulative gross equity before:', float(((1+rets).cumprod().shift(1).loc[date])))
    print('Cumulative gross equity after:', float(((1+rets).cumprod().loc[date])))
    print('\n')
