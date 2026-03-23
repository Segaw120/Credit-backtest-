
import streamlit as st
import pandas as pd
import numpy as np
import yahooquery as yq
from sodapy import Socrata
import talib
import matplotlib.pyplot as plt
import plotly.graph_objects as go
from datetime import datetime, timedelta

# Set page title and layout
st.set_page_config(
    page_title="HealthGauge Algorithm Backtest",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Title and description
st.title("HealthGauge Algorithm Backtest")
st.markdown("""
This app backtests the HealthGauge algorithm using data from Yahoo Finance and Socrata (COT data).
It runs 5 iterations with different parameters and displays performance metrics.
""")

# --- COT Market Mapping ---
COT_MARKET_MAP = {
    # Metals
    "GC=F": "114131",  # Gold
    "SI=F": "114133",  # Silver
    "HG=F": "114132",  # Copper
    "PL=F": "114134",  # Platinum
    "PA=F": "114135",  # Palladium

    # Energy
    "CL=F": "114132",  # WTI Crude Oil
    "BZ=F": "114136",  # Brent Crude Oil
    "NG=F": "114137",  # Natural Gas

    # Currencies
    "EURUSD=X": "096742",  # Euro FX
    "JPY=X": "097741",     # Japanese Yen
    "GBPUSD=X": "094741",  # British Pound
    "AUDUSD=X": "092741",  # Australian Dollar
    "USDCAD=X": "093741",  # Canadian Dollar
    "USDCHF=X": "095741",  # Swiss Franc
    "NZDUSD=X": "098741",  # New Zealand Dollar

    # Indices (Note: Most indices do not have COT data)
    "^GSPC": None,  # S&P 500 (no COT)
    "^DJI": None,   # Dow Jones (no COT)
    "^IXIC": None,  # NASDAQ (no COT)
    "^RUT": None,   # Russell 2000 (no COT)
    "^FTSE": None,  # FTSE 100 (no COT)
    "^N225": None,  # Nikkei 225 (no COT)
    "^GDAXI": None  # DAX (no COT)
}

# --- Asset Tree Structure ---
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

# --- Sidebar for User Inputs ---
st.sidebar.header("Backtest Parameters")

# Asset selection dropdown
asset_groups = list(ASSET_TREE.keys())
selected_group = st.sidebar.selectbox("Select Asset Group:", asset_groups)
selected_asset = st.sidebar.selectbox(
    "Select Asset:",
    list(ASSET_TREE[selected_group].keys())
)
ticker = ASSET_TREE[selected_group][selected_asset]["yahoo_ticker"]

# Date range selection
start_date = st.sidebar.date_input("Start Date:", datetime(2020, 1, 1))
end_date = st.sidebar.date_input("End Date:", datetime(2023, 12, 31))

# COT data toggle (only available for certain assets)
use_cot = st.sidebar.checkbox("Use COT Data (if available)", True)

# --- Data Fetching Functions ---
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
        client = Socrata("data.cftc.gov", None)
        results = client.get(
            "6dca-aqww",
            where=f"market_code='{market_code}'",
            limit=5000
        )
        cot_df = pd.DataFrame.from_records(results)

        # Convert and filter dates
        cot_df['report_date_as_yyyymmdd'] = pd.to_datetime(cot_df['report_date_as_yyyymmdd'])
        cot_df = cot_df[
            (cot_df['report_date_as_yyyymmdd'] >= pd.to_datetime(start_date)) &
            (cot_df['report_date_as_yyyymmdd'] <= pd.to_datetime(end_date))
        ]
        cot_df = cot_df.sort_values('report_date_as_yyyymmdd')

        return cot_df
    except Exception as e:
        st.error(f"Error fetching COT data: {e}")
        return pd.DataFrame()

# --- Indicator Calculation Functions ---
def calculate_technical_indicators(df):
    if df.empty:
        return df

    df = df.copy()

    # Trend Indicators
    df['sma20'] = talib.SMA(df['close'], timeperiod=20)
    df['sma50'] = talib.SMA(df['close'], timeperiod=50)
    df['sma200'] = talib.SMA(df['close'], timeperiod=200)

    # Momentum (RSI)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)

    # Liquidity (RVOL)
    df['vol_sma20'] = talib.SMA(df['volume'], timeperiod=20)
    df['rvol'] = df['volume'] / df['vol_sma20']

    # Volatility (ATR and Bollinger Bands)
    df['atr'] = talib.ATR(
        df['high'], df['low'], df['close'], timeperiod=14
    )
    df['bb_upper'], df['bb_middle'], df['bb_lower'] = talib.BBANDS(
        df['close'], timeperiod=20, nbdevup=2, nbdevdn=2, matype=0
    )
    df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / df['bb_middle']

    return df

def calculate_cot_indicators(df, cot_df, window=52):
    if df.empty or cot_df.empty:
        return df

    df = df.copy()

    # Merge COT data with OHLCV data on date
    cot_df['date'] = pd.to_datetime(cot_df['report_date_as_yyyymmdd']).dt.date
    df = df.merge(cot_df, on='date', how='left')

    # Calculate net positions and z-scores
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

    # Initialize health score
    health = pd.Series(0.0, index=df.index)

    # Trend Component (Max 4 pts)
    health += np.where(df['close'] > df['sma20'], 1.0, 0)
    health += np.where(df['close'] > df['sma50'], 1.5, 0)
    health += np.where(df['close'] > df['sma200'], 1.5, 0)

    # Liquidity Component (Max 3 pts)
    health += np.where(df['rvol'] > 1.0, 1.5, 0)
    health += np.where(df['rvol'] > 1.5, 1.5, 0)

    # Volatility/Stability Component (Max 3 pts)
    bb_width_sma = df['bb_width'].rolling(window=20).mean()
    health += np.where(df['bb_width'] < bb_width_sma, 3.0, 0)

    df['health_gauge'] = health
    return df

def classify_market_structure(df):
    if df.empty:
        return df

    df = df.copy()

    # Initialize market structure
    df['market_structure'] = 'ranging'

    # Bullish conditions
    bullish = (
        (df['close'] > df['sma200']) &
        (df['sma50'] > df['sma200']) &
        (df['atr'] > 1.5 * df['atr'].rolling(20).mean())
    )
    df.loc[bullish, 'market_structure'] = 'bullish'

    # Bearish conditions
    bearish = (
        (df['close'] < df['sma200']) &
        (df['sma50'] < df['sma200']) &
        (df['atr'] > 1.5 * df['atr'].rolling(20).mean())
    )
    df.loc[bearish, 'market_structure'] = 'bearish'

    return df

def calculate_fibonacci_levels(df):
    if df.empty:
        return df

    df = df.copy()

    # Find swing highs and lows (simplified)
    df['swing_high'] = df['high'].rolling(50, center=True).max()
    df['swing_low'] = df['low'].rolling(50, center=True).min()

    # Calculate Fibonacci levels from swing high to low
    df['fib_23.6'] = df['swing_high'] - 0.236 * (df['swing_high'] - df['swing_low'])
    df['fib_38.2'] = df['swing_high'] - 0.382 * (df['swing_high'] - df['swing_low'])
    df['fib_50'] = df['swing_high'] - 0.5 * (df['swing_high'] - df['swing_low'])
    df['fib_61.8'] = df['swing_high'] - 0.618 * (df['swing_high'] - df['swing_low'])

    return df

# --- Signal Generation and Backtesting ---
def generate_trading_signals(
    df,
    rsi_oversold=30,
    rsi_overbought=70,
    zscore_threshold=2.0,
    health_threshold=7.0,
    use_commercial=True,
    fib_retracement=0.382,
    continuation_days=3
):
    if df.empty:
        return df

    df = df.copy()
    df['signal'] = 'HOLD'
    df['fib_retracement_level'] = None
    df['continuation_days'] = 0

    z_col = 'commercial_net_zscore' if use_commercial else 'non_commercial_net_zscore'

    if z_col not in df.columns:
        st.warning(f"COT data not available. Using technical indicators only.")
        z_col = None

    # Buy: Oversold + Institutional Backing (Low Z-Score) + High Market Health
    if z_col:
        buy_condition = (
            (df['rsi'] < rsi_oversold) &
            (df[z_col] < -zscore_threshold) &
            (df['health_gauge'] >= health_threshold)
        )
    else:
        buy_condition = (
            (df['rsi'] < rsi_oversold) &
            (df['health_gauge'] >= health_threshold)
        )

    # Sell: Overbought + Institutional Distribution (High Z-Score) + High Market Health
    if z_col:
        sell_condition
