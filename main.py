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

def calculate_market_health_gauge(df):
    if df.empty:
        return df

    df = df.copy()

    # Initialize health score
    health = pd.Series(0.0, index=df.index)

    # Trend Component (Max 4 pts)
    health += np.where(df['close'] > df['sma20'],  1.0, 0)
    health += np.where(df['close'] > df['sma50'],  1.5, 0)
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
        (df['sma50']  > df['sma200']) &
        (df['atr'] > 1.5 * df['atr'].rolling(20).mean())
    )
    df.loc[bullish, 'market_structure'] = 'bullish'

    # Bearish conditions
    bearish = (
        (df['close'] < df['sma200']) &
        (df['sma50']  < df['sma200']) &
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
    df['swing_low']  = df['low'].rolling(50,  center=True).min()

    # Calculate Fibonacci levels from swing high to low
    df['fib_23.6'] = df['swing_high'] - 0.236 * (df['swing_high'] - df['swing_low'])
    df['fib_38.2'] = df['swing_high'] - 0.382 * (df['swing_high'] - df['swing_low'])
    df['fib_50']   = df['swing_high'] - 0.5   * (df['swing_high'] - df['swing_low'])
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
        sell_condition = (
            (df['rsi'] > rsi_overbought) &
            (df[z_col] > zscore_threshold) &
            (df['health_gauge'] >= health_threshold)
        )
    else:
        sell_condition = (
            (df['rsi'] > rsi_overbought) &
            (df['health_gauge'] >= health_threshold)
        )

    # Apply Fibonacci and continuation logic
    for i in range(1, len(df)):
        if buy_condition.iloc[i]:
            if df['market_structure'].iloc[i] == 'bullish':
                if df['close'].iloc[i] <= df['fib_38.2'].iloc[i]:
                    df.loc[df.index[i], 'fib_retracement_level'] = '38.2%'
                    df.loc[df.index[i], 'signal'] = 'BUY'
            elif df['market_structure'].iloc[i] in ['bearish', 'ranging']:
                if i >= continuation_days:
                    prev_signals = df['signal'].iloc[i - continuation_days:i]
                    if not prev_signals.eq('BUY').any():
                        df.loc[df.index[i], 'continuation_days'] = continuation_days
                        df.loc[df.index[i], 'signal'] = 'BUY'

        if sell_condition.iloc[i]:
            if df['market_structure'].iloc[i] == 'bearish':
                if df['close'].iloc[i] >= df['fib_61.8'].iloc[i]:
                    df.loc[df.index[i], 'fib_retracement_level'] = '61.8%'
                    df.loc[df.index[i], 'signal'] = 'SELL'
            elif df['market_structure'].iloc[i] in ['bullish', 'ranging']:
                if i >= continuation_days:
                    prev_signals = df['signal'].iloc[i - continuation_days:i]
                    if not prev_signals.eq('SELL').any():
                        df.loc[df.index[i], 'continuation_days'] = continuation_days
                        df.loc[df.index[i], 'signal'] = 'SELL'

    return df

def backtest_strategy(df, initial_capital=10000):
    if df.empty:
        return pd.DataFrame(), {}

    df = df.copy()
    position = 0
    cash = initial_capital
    portfolio_value = [initial_capital]
    trade_log = []

    for i in range(1, len(df)):
        if df['signal'].iloc[i] == 'BUY' and position == 0:
            shares = cash / df['close'].iloc[i]
            position = shares
            cash = 0
            entry_price = df['close'].iloc[i]
            trade_log.append({
                'date': df['date'].iloc[i],
                'signal': 'BUY',
                'price': entry_price,
                'fib_level': df['fib_retracement_level'].iloc[i],
                'continuation': df['continuation_days'].iloc[i]
            })

        elif df['signal'].iloc[i] == 'SELL' and position > 0:
            cash = position * df['close'].iloc[i]
            position = 0
            exit_price = df['close'].iloc[i]
            trade_log.append({
                'date': df['date'].iloc[i],
                'signal': 'SELL',
                'price': exit_price,
                'fib_level': df['fib_retracement_level'].iloc[i],
                'continuation': df['continuation_days'].iloc[i]
            })

        current_value = cash + (position * df['close'].iloc[i])
        portfolio_value.append(current_value)

    # Calculate returns
    df['portfolio_value'] = portfolio_value[:-1]  # Align with df length
    total_return = (portfolio_value[-1] / initial_capital - 1) * 100

    # Calculate performance metrics
    daily_returns = df['portfolio_value'].pct_change().dropna()
    sharpe_ratio  = np.sqrt(252) * daily_returns.mean() / daily_returns.std()
    max_drawdown  = (
        (df['portfolio_value'].cummax() - df['portfolio_value']).max()
        / df['portfolio_value'].cummax().max()
    )

    # Win rate and profit factor
    trades_df = pd.DataFrame(trade_log)
    if not trades_df.empty and len(trades_df) > 1:
        buy_trades  = trades_df[trades_df['signal'] == 'BUY']
        sell_trades = trades_df[trades_df['signal'] == 'SELL']
        if len(buy_trades) == len(sell_trades):
            trades_df['return'] = (
                sell_trades['price'].values / buy_trades['price'].values - 1
            )
            winning_trades = trades_df[trades_df['return'] > 0]
            win_rate       = len(winning_trades) / len(trades_df) * 100
            profit_factor  = (
                trades_df['return'][trades_df['return'] > 0].sum()
                / abs(trades_df['return'][trades_df['return'] < 0].sum())
            )
        else:
            win_rate, profit_factor = np.nan, np.nan
    else:
        win_rate, profit_factor = np.nan, np.nan

    # Compile metrics
    metrics = {
        'Total Return (%)': total_return,
        'Sharpe Ratio':     sharpe_ratio,
        'Max Drawdown (%)': max_drawdown * 100,
        'Win Rate (%)':     win_rate,
        'Profit Factor':    profit_factor,
        'Number of Trades': len(trades_df) // 2 if not trades_df.empty else 0
    }

    return df, metrics


def main():
    # Fetch OHLCV data
    with st.spinner("Fetching OHLCV data..."):
        ohlcv_df = fetch_ohlcv_data(ticker, start_date, end_date)

    if ohlcv_df.empty:
        st.error("Failed to fetch OHLCV data. Please check the ticker and date range.")
        return

    # Fetch COT data if requested and available
    cot_df = pd.DataFrame()
    cot_code = COT_MARKET_MAP.get(ticker)
    if use_cot and cot_code:
        with st.spinner("Fetching COT data..."):
            cot_df = fetch_cot_data(ticker, start_date, end_date)

    # Build indicator stack
    ohlcv_df = calculate_technical_indicators(ohlcv_df)
    if not cot_df.empty:
        ohlcv_df = calculate_cot_indicators(ohlcv_df, cot_df)
    ohlcv_df = calculate_market_health_gauge(ohlcv_df)
    ohlcv_df = classify_market_structure(ohlcv_df)
    ohlcv_df = calculate_fibonacci_levels(ohlcv_df)

    # 5 iterations with different parameter sets
    iterations = [
        {"rsi_oversold": 30, "rsi_overbought": 70, "zscore_threshold": 2.0,
         "health_threshold": 7.0, "use_commercial": True,  "fib_retracement": 0.382, "continuation_days": 3},
        {"rsi_oversold": 25, "rsi_overbought": 75, "zscore_threshold": 1.5,
         "health_threshold": 6.0, "use_commercial": True,  "fib_retracement": 0.5,   "continuation_days": 5},
        {"rsi_oversold": 35, "rsi_overbought": 65, "zscore_threshold": 2.5,
         "health_threshold": 8.0, "use_commercial": False, "fib_retracement": 0.618, "continuation_days": 2},
        {"rsi_oversold": 20, "rsi_overbought": 80, "zscore_threshold": 3.0,
         "health_threshold": 5.0, "use_commercial": True,  "fib_retracement": 0.236, "continuation_days": 7},
        {"rsi_oversold": 40, "rsi_overbought": 60, "zscore_threshold": 1.0,
         "health_threshold": 9.0, "use_commercial": False, "fib_retracement": 0.382, "continuation_days": 1},
    ]

    all_metrics = []

    def _fmt(val, pct=False):
        if isinstance(val, float) and np.isnan(val):
            return "N/A"
        return f"{val:.2f}{'%' if pct else ''}"

    for i, params in enumerate(iterations):
        st.write(f"### Iteration {i + 1}")
        st.write(
            f"**RSI Oversold:** {params['rsi_oversold']} | "
            f"**RSI Overbought:** {params['rsi_overbought']} | "
            f"**Z-Score Threshold:** {params['zscore_threshold']} | "
            f"**Health Threshold:** {params['health_threshold']} | "
            f"**Use Commercial:** {params['use_commercial']} | "
            f"**Continuation Days:** {params['continuation_days']}"
        )

        df_signals = generate_trading_signals(ohlcv_df.copy(), **params)
        result_df, metrics = backtest_strategy(df_signals)
        all_metrics.append(metrics)

        col1, col2, col3, col4, col5, col6 = st.columns(6)
        col1.metric("Total Return",  _fmt(metrics.get('Total Return (%)', np.nan), pct=True))
        col2.metric("Sharpe Ratio",  _fmt(metrics.get('Sharpe Ratio',     np.nan)))
        col3.metric("Max Drawdown",  _fmt(metrics.get('Max Drawdown (%)', np.nan), pct=True))
        col4.metric("Win Rate",      _fmt(metrics.get('Win Rate (%)',     np.nan), pct=True))
        col5.metric("Profit Factor", _fmt(metrics.get('Profit Factor',    np.nan)))
        col6.metric("# Trades",      str(metrics.get('Number of Trades', 0)))

        # Portfolio value chart with signal markers
        if not result_df.empty and 'portfolio_value' in result_df.columns:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=result_df['date'], y=result_df['portfolio_value'],
                mode='lines', name='Portfolio Value'
            ))
            buys  = result_df[result_df['signal'] == 'BUY']
            sells = result_df[result_df['signal'] == 'SELL']
            fig.add_trace(go.Scatter(
                x=buys['date'], y=buys['portfolio_value'],
                mode='markers',
                marker=dict(symbol='triangle-up', color='green', size=10),
                name='BUY'
            ))
            fig.add_trace(go.Scatter(
                x=sells['date'], y=sells['portfolio_value'],
                mode='markers',
                marker=dict(symbol='triangle-down', color='red', size=10),
                name='SELL'
            ))
            fig.update_layout(
                title=f"Iteration {i + 1} – Portfolio Value",
                xaxis_title="Date",
                yaxis_title="Portfolio Value ($)",
                hovermode="x unified"
            )
            st.plotly_chart(fig, use_container_width=True)

        st.divider()

    # Build comparative DataFrame
    comparative_df = pd.DataFrame(
        all_metrics,
        index=[f"Iteration {i + 1}" for i in range(len(iterations))]
    )

    st.write("### Comparative Performance Table")
    st.dataframe(comparative_df.style.format("{:.2f}"))

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
