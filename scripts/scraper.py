import akshare as ak
import yfinance as yf
import pandas as pd

def get_shanghai_crude_robust():
    """Fetches Shanghai Crude (SC) Main Contract via AKShare."""
    print("Fetching Shanghai Crude from Sina/INE...")
    # 'SC0' is the continuous main contract. 
    # Removed 'sample' argument to fix the 'unexpected keyword argument' error.
    df = ak.futures_main_sina(symbol="SC0")
    
    # Standardizing the columns: AKShare usually returns ['日期', '开盘价', '最高价', '最低价', '收盘价', '成交量', '持仓量']
    # We only need Date and Close.
    df = df.iloc[:, [0, 4]] 
    df.columns = ['date', 'Shanghai_CNY']
    
    df['date'] = pd.to_datetime(df['date'])
    df.set_index('date', inplace=True)
    return df

def get_western_data():
    """Fetches WTI, Brent, and USDCNY via yfinance."""
    print("Fetching WTI, Brent, and FX...")
    tickers = ["CL=F", "BZ=F", "CNY=X"]
    
    # multi_level_index=False is required for yfinance >= 0.2.50 to keep columns simple strings
    data = yf.download(tickers, period='max', multi_level_index=False, auto_adjust=True)
    
    # Extract only 'Close' and rename
    data = data['Close'].rename(columns={
        "CL=F": "WTI",
        "BZ=F": "Brent",
        "CNY=X": "USDCNY"
    })
    return data

def build_robust_dataset():
    # 1. Get Data
    sh_df = get_shanghai_crude_robust()
    west_df = get_western_data()

    # 2. Align Timezones
    # Remove timezone awareness so they merge on the date string
    if west_df.index.tz is not None:
        west_df.index = west_df.index.tz_localize(None)
    if sh_df.index.tz is not None:
        sh_df.index = sh_df.index.tz_localize(None)
    
    # 3. Join & Align
    # Using 'inner' join keeps only days where all markets (CN and US) were open.
    # If you want more data points, use 'outer' followed by .ffill()
    combined = sh_df.join(west_df, how="inner")
    
    # 4. Currency Conversion
    combined["Shanghai_USD"] = combined["Shanghai_CNY"] / combined["USDCNY"]
    
    # 5. Clean up
    combined = combined.sort_index().dropna()
    return combined

if __name__ == "__main__":
    try:
        df = build_robust_dataset()
        print("\n--- Success ---")
        print(f"Dataset Shape: {df.shape}")
        print(df.tail())
        
        # Save output
        df.to_csv("raw_data/crude_data_robust.csv")
        print("\nFile saved: crude_data_robust.csv")
        
    except Exception as e:
        print(f"\nCritical Failure: {e}")
        print("Tip: Ensure 'akshare' and 'yfinance' are updated to the latest versions.")