import requests
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
    "JPY=X":    "097741",  # Japanese Yen
    "GBPUSD=X": "094741",  # British Pound
    "AUDUSD=X": "092741",  # Australian Dollar
    "USDCAD=X": "093741",  # Canadian Dollar
    "USDCHF=X": "095741",  # Swiss Franc
    "NZDUSD=X": "098741",  # New Zealand Dollar

    # Indices (no COT data)
    "^GSPC":  None,
    "^DJI":   None,
    "^IXIC":  None,
    "^RUT":   None,
    "^FTSE":  None,
    "^N225":  None,
    "^GDAXI": None
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

asset_groups = list(ASSET_TREE.keys())
selected_group = st.sidebar.selectbox("Select Asset Group:", asset_groups)
selected_asset = st.sidebar.selectbox(
    "Select Asset:",
    list(ASSET_TREE[selected_group].keys())
)
ticker = ASSET_TREE[selected_group][selected_asset]["yahoo_ticker"]

start_date = st.sidebar.date_input("Start Date:", datetime(2020, 1, 1))
end_date   = st.sidebar.date_input("End Date:",   datetime(2023, 12, 31))

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
        client = Socrata("publicreporting.cftc.gov", None)

        where_clause = (
            f"cftc_market_code='{market_code}' AND "
            f"report_date_as_yyyy_mm_dd>='{start_date.strftime('%Y-%m-%d')}' AND "
            f"report_date_as_yyyy_mm_dd<='{end_date.strftime('%Y-%m-%d')}'"
        )

        results = client.get(
            "6dca-aqww",
            where=where_clause,
            limit=5000,
            select=(
                "report_date_as_yyyy_mm_dd, "
                "noncomm_positions_long_all, noncomm_positions_short_all, "
                "comm_positions_long_all, comm_positions_short_all"
            )
        )

        # FIX 1: guard against empty results list BEFORE building the DataFrame.
        # pd.DataFrame.from_records([]) creates a column-less frame, so any
        # column access afterwards raises a KeyError.
        if not results:
            st.warning(f"No COT records returned for {ticker}.")
            return pd.DataFrame()

        cot_df = pd.DataFrame.from_records(results)

        if cot_df.empty:
            st.warning(f"Empty COT DataFrame for {ticker}.")
            return pd.DataFrame()

        # FIX 2: normalise column names to lowercase to handle any API casing
        # variation that could also cause the KeyError.
        cot_df.columns = [col.lower() for col in cot_df.columns]

        date_col = 'report_date_as_yyyy_mm_dd'
        if date_col not in cot_df.columns:
            st.warning(
                f"Expected date column '{date_col}' not found. "
                f"Available columns: {list(cot_df.columns)}"
            )
            return pd.DataFrame()

        cot_df[date_col] = pd.to_datetime(cot_df[date_col])
        cot_df = cot_df.sort_values(date_col)

        # FIX 3: rename ALL position columns so they match the names that
        # calculate_cot_indicators() checks for.  Previously the fetched names
        # (comm_positions_long_all / noncomm_positions_long_all) never matched
        # the expected names (commercial_long_all / noncommercial_long_all),
        # so COT z-scores were silently never calculated.
        cot_df.rename(columns={
            date_col:                       'report_date_as_yyyymmdd',
            'comm_positions_long_all':      'commercial_long_all',
            'comm_positions_short_all':     'commercial_short_all',
            'noncomm_positions_long_all':   'noncommercial_long_all',
            'noncomm_positions_short_all':  'noncommercial_short_all',
        }, inplace=True)

        # Ensure position columns are numeric
        for col in ['commercial_long_all', 'commercial_short_all',
                    'noncommercial_long_all', 'noncommercial_short_all']:
            if col in cot_df.columns:
                cot_df[col] = pd.to_numeric(cot_df[col], errors='coerce')

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
    df['sma20']  = talib.SMA(df['close'], timeperiod=20)
    df['sma50']  = talib.SMA(df['close'], timeperiod=50)
    df['sma200'] = talib.SMA(df['close'], timeperiod=200)

    # Momentum (RSI)
    df['rsi'] = talib.RSI(df['close'], timeperiod=14)

    # Liquidity (RVOL)
    df['vol_sma20'] = talib.SMA(df['volume'], timeperiod=20)
    df['rvol'] = df['volume'] / df['vol_sma20']

    # Volatility (ATR and Bollinger Bands)
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




        # Plot comparative metrics
        st.write("### Comparative Performance")
        fig = go.Figure()
        for metric in ['Total Return (%)', 'Sharpe Ratio', 'Max Drawdown (%)', 'Win Rate (%)', 'Profit Factor']:
            fig.add_trace(go.Bar(
                x=comparative_df.index,
                y=comparative_df[metric],
                name=metric
            ))

        fig.update_layout(
            title="Comparative Performance Metrics",
            xaxis_title="Iteration",
            yaxis_title="Value",
            barmode='group',
            hovermode="x unified"
        )
        st.plotly_chart(fig, use_container_width=True)

if __name__ == "__main__":
    main()
