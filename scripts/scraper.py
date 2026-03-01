import akshare as ak
import yfinance as yf
import pandas as pd
import numpy as np

def get_shanghai_crude_robust():
    print("Fetching Shanghai Crude (SC0) via Sina...")
    df = ak.futures_main_sina(symbol="SC0")
    df = df.iloc[:, [0, 4]] 
    df.columns = ['date', 'Shanghai_CNY']
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def get_western_data():
    print("Fetching WTI, Brent, and FX...")
    # Using 'max' period to ensure enough history for HMM training
    tickers = ["CL=F", "BZ=F", "CNY=X"]
    data = yf.download(tickers, period='max', multi_level_index=False, auto_adjust=True)
    data = data['Close'].rename(columns={"CL=F": "WTI", "BZ=F": "Brent", "CNY=X": "USDCNY"})
    return data

def build_robust_dataset():
    sh_df = get_shanghai_crude_robust()
    west_df = get_western_data()

    # 1. Standardize Index
    for d in [sh_df, west_df]:
        if d.index.tz is not None:
            d.index = d.index.tz_localize(None)

    # 2. Shift Western Data to align with Shanghai's earlier close
    # Shanghai closes while the US is still in the previous night. 
    # To avoid look-ahead bias, we use the US settlement from the 'night before'
    # relative to the Shanghai trading day.
    west_df = west_df.shift(1)

    # 3. Outer Join + Limited F-Fill (Handles differing holidays)
    # We allow a 2-day fill to bridge minor holidays without introducing stale data
    combined = sh_df.join(west_df, how="outer").sort_index()
    combined = combined.ffill(limit=2)

    # 4. Currency Conversion
    combined["Shanghai_USD"] = combined["Shanghai_CNY"] / combined["USDCNY"]

    # 5. Roll-Over Detection (Simple Discontinuity Check)
    # Detects if any asset jumped more than 3 standard deviations in one day
    # These days often represent contract rolls rather than market moves.
    for col in ['Shanghai_USD', 'WTI', 'Brent']:
        returns = combined[col].pct_change()
        combined[f'{col}_Roll_Flag'] = (returns.abs() > (returns.std() * 3)).astype(int)

    # 6. Final Clean up
    combined = combined.dropna()
    return combined

if __name__ == "__main__":
    df = build_robust_dataset()
    df.to_csv("crude_data_robust.csv")
    print(f"Dataset generated: {df.shape[0]} aligned trading days.")