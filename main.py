import requests
import streamlit as st
import pandas as pd
import numpy as np
import yahooquery as yq
from sodapy import Socrata
import talib
import plotly.graph_objects as go
from datetime import datetime

st.set_page_config(
    page_title="HealthGauge Algorithm Backtest",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.title("HealthGauge Algorithm Backtest")
st.markdown("""
This app backtests the HealthGauge algorithm using data from Yahoo Finance and Socrata (COT data).
It runs 5 iterations with different parameters and displays performance metrics.
""")

COT_MARKET_MAP = {
    "GC=F": "114131",
    "SI=F": "114133",
    "HG=F": "114132",
    "PL=F": "114134",
    "PA=F": "114135",
    "CL=F": "114132",
    "BZ=F": "114136",
    "NG=F": "114137",
    "EURUSD=X": "096742",
    "JPY=X":    "097741",
    "GBPUSD=X": "094741",
    "AUDUSD=X": "092741",
    "USDCAD=X": "093741",
    "USDCHF=X": "095741",
    "NZDUSD=X": "098741",
    "^GSPC":  None,
    "^DJI":   None,
    "^IXIC":  None,
    "^RUT":   None,
    "^FTSE":  None,
    "^N225":  None,
    "^GDAXI": None
}

ASSET_TREE = {
    "metals": {
        "GOLD - COMMODITY EXCHANGE INC.": {
            "name": "Gold",
            "yahoo_ticker": "GC=F",
            "asset_group": "Precious Metals"
        },
        "SILVER - COMMODITY EXCHANGE INC.": {
            "name": "Silver",
            "yahoo_ticker": "SI=F",
            "asset_group": "Precious Metals"
        },
        "COPPER - COMMODITY EXCHANGE INC.": {
            "name": "Copper",
            "yahoo_ticker": "HG=F",
            "asset_group": "Base Metals"
        },
        "PLATINUM - NEW YORK MERCANTILE EXCHANGE": {
            "name": "Platinum",
            "yahoo_ticker": "PL=F",
            "asset_group": "Precious Metals"
        },
        "PALLADIUM - NEW YORK MERCANTILE EXCHANGE": {
            "name": "Palladium",
            "yahoo_ticker": "PA=F",
            "asset_group": "Precious Metals"
        }
    },
    "energy": {
        "WTI FINANCIAL CRUDE OIL - NEW YORK MERCANTILE EXCHANGE": {
            "name": "Crude Oil (WTI)",
            "yahoo_ticker": "CL=F",
            "asset_group": "Energy"
        },
        "BRENT LAST DAY - NEW YORK MERCANTILE EXCHANGE": {
            "name": "Brent Crude Oil",
            "yahoo_ticker": "BZ=F",
            "asset_group": "Energy"
        },
        "E-MINI NATURAL GAS - NEW YORK MERCANTILE EXCHANGE": {
            "name": "Natural Gas",
            "yahoo_ticker": "NG=F",
            "asset_group": "Energy"
        }
    },
    "currencies": {
        "EURO FX - CHICAGO MERCANTILE EXCHANGE": {
            "name": "EUR/USD",
            "yahoo_ticker": "EURUSD=X",
            "asset_group": "Forex"
        },
        "JAPANESE YEN - CHICAGO MERCANTILE EXCHANGE": {
            "name": "USD/JPY",
            "yahoo_ticker": "JPY=X",
            "asset_group": "Forex"
        },
        "BRITISH POUND - CHICAGO MERCANTILE EXCHANGE": {
            "name": "GBP/USD",
            "yahoo_ticker": "GBPUSD=X",
            "asset_group": "Forex"
        },
        "AUSTRALIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": {
            "name": "AUD/USD",
            "yahoo_ticker": "AUDUSD=X",
            "asset_group": "Forex"
        },
        "CANADIAN DOLLAR - CHICAGO MERCANTILE EXCHANGE": {
            "name": "USD/CAD",
            "yahoo_ticker": "USDCAD=X",
            "asset_group": "Forex"
        },
        "SWISS FRANC - CHICAGO MERCANTILE EXCHANGE": {
            "name": "USD/CHF",
            "yahoo_ticker": "USDCHF=X",
            "asset_group": "Forex"
        },
        "NZ DOLLAR - CHICAGO MERCANTILE EXCHANGE": {
            "name": "NZD/USD",
            "yahoo_ticker": "NZDUSD=X",
            "asset_group": "Forex"
        }
    },
    "indices": {
        "E-MINI S&P 500 - CHICAGO MERCANTILE EXCHANGE": {
            "name": "S&P 500",
            "yahoo_ticker": "^GSPC",
            "asset_group": "Equity Indices"
        },
        "DOW JONES INDUSTRIAL AVERAGE - CHICAGO BOARD OF TRADE": {
            "name": "Dow Jones",
            "yahoo_ticker": "^DJI",
            "asset_group": "Equity Indices"
        },
        "NASDAQ MINI - CHICAGO MERCANTILE EXCHANGE": {
            "name": "NASDAQ",
            "yahoo_ticker": "^IXIC",
            "asset_group": "Equity Indices"
        },
        "RUSSELL 2000 STOCK INDEX - ICE FUTURES U.S.": {
            "name": "Russell 2000",
            "yahoo_ticker": "^RUT",
            "asset_group": "Equity Indices"
        },
        "FTSE 100 Index": {
            "name": "FTSE 100",
            "yahoo_ticker": "^FTSE",
            "asset_group": "Equity Indices"
        },
        "NIKKEI STOCK AVERAGE - CHICAGO MERCANTILE EXCHANGE": {
            "name": "Nikkei 225",
            "yahoo_ticker": "^N225",
            "asset_group": "Equity Indices"
        },
        "DAX Performance Index": {
            "name": "DAX",
            "yahoo_ticker": "^GDAXI",
            "asset_group": "Equity Indices"
        }
    }
}

st.sidebar.header("Backtest Parameters")
asset_groups = list(ASSET_TREE.keys())
selected_group = st.sidebar.selectbox("Select Asset Group:", asset_groups)
selected_asset = st.sidebar.selectbox("Select Asset:", list(ASSET_TREE[selected_group].keys()))
ticker = ASSET_TREE[selected_group][selected_asset]["yahoo_ticker"]

start_date = st.sidebar.date_input("Start Date:", datetime(2020, 1, 1))
end_date   = st.sidebar.date_input("End Date:",   datetime(2023, 12, 31))
use_cot = st.sidebar.checkbox("Use COT Data (if available)", True)

@st.cache_data
def fetch_ohlcv_data(ticker, start_date, end_date):
    try:
        data = yq.Ticker(ticker)
        df = data.history(start=start_date, end=end_date)
        df.reset_index(inplace=True)
        df['date'] = pd.to_datetime(df['date']).dt.date
        return df
    except Exception as e:
        st.error(f"Error fetching OHLCV data: {e}")
        return pd.DataFrame()

@st.cache_data
def fetch_cot_data(ticker, start_date, end_date):
    market_code = COT_MARKET_MAP.get(ticker)
    if not market_code:
        st.warning(f"No COT data available for {ticker}.")
        return pd.DataFrame()

    try:
        client = Socrata("publicreporting.cftc.gov", None)
        
        start_str = start_date.strftime('%Y-%m-%dT00:00:00')
        end_str = end_date.strftime('%Y-%m-%dT00:00:00')

        where_clause = (
            f"cftc_market_code='{market_code}' AND "
            f"report_date_as_yyyy_mm_dd >= '{start_str}' AND "
            f"report_date_as_yyyy_mm_dd <= '{end_str}'"
        )

        results = client.get(
            "72hh-2qvh",
            where=where_clause,
            limit=5000,
            select=(
                "report_date_as_yyyy_mm_dd, "
                "noncomm_positions_long_all, noncomm_positions_short_all, "
                "comm_positions_long_all, comm_positions_short_all"
            )
        )

        if not results:
            st.warning(f"No COT records returned for {ticker}.")
            return pd.DataFrame()

        cot_df = pd.DataFrame.from_records(results)
        cot_df.columns = [col.lower() for col in cot_df.columns]

        date_col = 'report_date_as_yyyy_mm_dd'
        if date_col not in cot_df.columns:
            return pd.DataFrame()

        cot_df[date_col] = pd.to_datetime(cot_df[date_col])
        cot_df = cot_df.sort_values(date_col)

        cot_df.rename(columns={
            date_col:                       'report_date_as_yyyymmdd',
            'comm_positions_long_all':      'commercial_long_all',
            'comm_positions_short_all':     'commercial_short_all',
            'noncomm_positions_long_all':   'noncommercial_long_all',
            'noncomm_positions_short_all':  'noncommercial_short_all',
        }, inplace=True)

        for col in ['commercial_long_all', 'commercial_short_all',
                    'noncommercial_long_all', 'noncommercial_short_all']:
            if col in cot_df.columns:
                cot_df[col] = pd.to_numeric(cot_df[col], errors='coerce')

        return cot_df

    except Exception as e:
        st.error(f"Error fetching COT data: {e}")
        return pd.DataFrame()



def calculate_technical_indicators(df):
    if df.empty:
        return df
    df = df.copy()
    df['sma20']  = talib.SMA(df['close'], timeperiod=20)
    df['sma50']  = talib.SMA(df['close'], timeperiod=50)
    df['sma200'] = talib.SMA(df['close'], timeperiod=200)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)
    df['vol_sma20'] = talib.SMA(df['volume'], timeperiod=20)
    df['rvol'] = df['volume'] / df['vol_sma20']
    df['atr'] = talib.ATR(df['high'], df['low'], df['close'], timeperiod=14)
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']
    return df

def calculate_cot_indicators(df, cot_df, window=52):
    if df.empty or cot_df.empty:
        return df
    df = df.copy()
    cot_df['date'] = pd.to_datetime(cot_df['report_date_as_yyyymmdd']).dt.date
    df = df.merge(cot_df, on='date', how='left')
    if 'commercial_long_all' in df.columns and 'commercial_short_all' in df.columns:
        df['commercial_net'] = df['commercial_long_all'] - df['commercial_short_all']
        df['commercial_net_zscore'] = (
            df['commercial_net'] - df['commercial_net'].rolling(window=window).mean()
        ) / df['commercial_net'].rolling(window=window).std()
    if 'noncommercial_long_all' in df.columns and 'noncommercial_short_all' in df.columns:
        df['non_commercial_net'] = df['noncommercial_long_all'] - df['noncommercial_short_all']
        df['non_commercial_net_zscore'] = (
            df['non_commercial_net'] - df['non_commercial_net'].rolling(window=window).mean()
        ) / df['non_commercial_net'].rolling(window=window).std()
    return df

def calculate_market_health_gauge(df):
    if df.empty:
        return df
    df = df.copy()
    health = pd.Series(0.0, index=df.index)
    health += np.where(df['close'] > df['sma20'],  1.0, 0)
    health += np.where(df['close'] > df['sma50'],  1.5, 0)
    health += np.where(df['close'] > df['sma200'], 1.5, 0)
    health += np.where(df['rvol'] > 1.0, 1.5, 0)
    health += np.where(df['rvol'] > 1.5, 1.5, 0)
    bb_width_sma = df['bb_width'].rolling(window=20).mean()
    health += np.where(df['bb_width'] < bb_width_sma, 3.0, 0)
    df['health_gauge'] = health
    return df

def classify_market_structure(df):
    if df.empty:
        return df
    df = df.copy()
    df['market_structure'] = 'ranging'
    bullish = (df['close'] > df['sma200']) & (df['sma50'] > df['sma200']) & (df['atr'] > 1.5 * df['atr'].rolling(20).mean())
    df.loc[bullish, 'market_structure'] = 'bullish'
    bearish = (df['close'] < df['sma200']) & (df['sma50'] < df['sma200']) & (df['atr'] > 1.5 * df['atr'].rolling(20).mean())
    df.loc[bearish, 'market_structure'] = 'bearish'
    return df

def calculate_fibonacci_levels(df):
    if df.empty:
        return df
    df = df.copy()
    df['swing_high'] = df['high'].rolling(50, center=True).max()
    df['swing_low']  = df['low'].rolling(50,  center=True).min()
    df['fib_23.6'] = df['swing_high'] - 0.236 * (df['swing_high'] - df['swing_low'])
    df['fib_38.2'] = df['swing_high'] - 0.382 * (df['swing_high'] - df['swing_low'])
    df['fib_50']   = df['swing_high'] - 0.5   * (df['swing_high'] - df['swing_low'])
    df['fib_61.8'] = df['swing_high'] - 0.618 * (df['swing_high'] - df['swing_low'])
    return df

def generate_trading_signals(df, rsi_oversold=30, rsi_overbought=70, zscore_threshold=2.0, health_threshold=7.0, use_commercial=True, fib_retracement=0.382, continuation_days=3):
    if df.empty:
        return df
    df = df.copy()
    df['signal'] = 'HOLD'
    df['fib_retracement_level'] = None
    df['continuation_days'] = 0
    z_col = 'commercial_net_zscore' if use_commercial else 'non_commercial_net_zscore'
    if z_col not in df.columns:
        z_col = None
    
    for i in range(1, len(df)):
        buy_cond = (df['rsi'].iloc[i] < rsi_oversold) and (df['health_gauge'].iloc[i] >= health_threshold)
        if z_col: buy_cond = buy_cond and (df[z_col].iloc[i] < -zscore_threshold)
        
        sell_cond = (df['rsi'].iloc[i] > rsi_overbought) and (df['health_gauge'].iloc[i] >= health_threshold)
        if z_col: sell_cond = sell_cond and (df[z_col].iloc[i] > zscore_threshold)

        if buy_cond:
            if df['market_structure'].iloc[i] == 'bullish' and df['close'].iloc[i] <= df['fib_38.2'].iloc[i]:
                df.at[df.index[i], 'signal'] = 'BUY'
            elif i >= continuation_days and not df['signal'].iloc[i-continuation_days:i].eq('BUY').any():
                df.at[df.index[i], 'signal'] = 'BUY'

        if sell_cond:
            if df['market_structure'].iloc[i] == 'bearish' and df['close'].iloc[i] >= df['fib_61.8'].iloc[i]:
                df.at[df.index[i], 'signal'] = 'SELL'
            elif i >= continuation_days and not df['signal'].iloc[i-continuation_days:i].eq('SELL').any():
                df.at[df.index[i], 'signal'] = 'SELL'
    return df

def backtest_strategy(df, initial_capital=10000):
    if df.empty: return pd.DataFrame(), {}
    df = df.copy()
    pos, cash = 0, initial_capital
    vals = [initial_capital] * len(df)
    log = []
    for i in range(1, len(df)):
        if df['signal'].iloc[i] == 'BUY' and pos == 0:
            pos, cash = cash / df['close'].iloc[i], 0
            log.append({'date': df['date'].iloc[i], 'signal': 'BUY', 'price': df['close'].iloc[i]})
        elif df['signal'].iloc[i] == 'SELL' and pos > 0:
            cash, pos = pos * df['close'].iloc[i], 0
            log.append({'date': df['date'].iloc[i], 'signal': 'SELL', 'price': df['close'].iloc[i]})
        vals[i] = cash + (pos * df['close'].iloc[i])
    df['portfolio_value'] = vals
    ret = (vals[-1] / initial_capital - 1) * 100
    mdd = ((df['portfolio_value'].cummax() - df['portfolio_value']).max() / df['portfolio_value'].cummax().max()) * 100
    metrics = {'Total Return (%)': ret, 'Max Drawdown (%)': mdd, 'Number of Trades': len(log) // 2}
    return df, metrics

def main():
    ohlcv_df = fetch_ohlcv_data(ticker, start_date, end_date)
    if ohlcv_df.empty: return
    cot_df = fetch_cot_data(ticker, start_date, end_date) if use_cot else pd.DataFrame()
    ohlcv_df = calculate_technical_indicators(ohlcv_df)
    if not cot_df.empty: ohlcv_df = calculate_cot_indicators(ohlcv_df, cot_df)
    ohlcv_df = calculate_market_health_gauge(ohlcv_df)
    ohlcv_df = classify_market_structure(ohlcv_df)
    ohlcv_df = calculate_fibonacci_levels(ohlcv_df)

    iterations = [
        {"rsi_oversold": 30, "rsi_overbought": 70, "zscore_threshold": 2.0, "health_threshold": 7.0},
        {"rsi_oversold": 25, "rsi_overbought": 75, "zscore_threshold": 1.5, "health_threshold": 6.0},
        {"rsi_oversold": 35, "rsi_overbought": 65, "zscore_threshold": 2.5, "health_threshold": 8.0},
        {"rsi_oversold": 20, "rsi_overbought": 80, "zscore_threshold": 3.0, "health_threshold": 5.0},
        {"rsi_oversold": 40, "rsi_overbought": 60, "zscore_threshold": 1.0, "health_threshold": 9.0}
    ]

    for i, params in enumerate(iterations):
        st.write(f"### Iteration {i + 1}")
        df_sig = generate_trading_signals(ohlcv_df.copy(), **params)
        res, met = backtest_strategy(df_sig)
        c1, c2, c3 = st.columns(3)
        c1.metric("Return", f"{met.get('Total Return (%)', 0):.2f}%")
        c2.metric("Max DD", f"{met.get('Max Drawdown (%)', 0):.2f}%")
        c3.metric("Trades", met.get('Number of Trades', 0))
        if not res.empty:
            fig = go.Figure()
            fig.add_trace(go.Scatter(x=res['date'], y=res['portfolio_value'], name='Portfolio'))
            st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
    
