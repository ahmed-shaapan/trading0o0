import streamlit as st
import pandas as pd
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import os
import subprocess
from datetime import datetime, date
from data_utils import load_stock_data, load_benchmark_data

# Page config
st.set_page_config(
    page_title="Financial Analysis Tool",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for Dash-like styling
st.markdown("""
<style>
    .main > div {
        padding-top: 1rem;
    }
    .stButton > button {
        border-radius: 10px;
        font-weight: bold;
        padding: 0.5rem 1rem;
        border: none;
        transition: all 0.3s ease;
    }
    .buy-button {
        background-color: #26a69a !important;
        color: white !important;
    }
    .sell-button {
        background-color: #ef5350 !important;
        color: white !important;
    }
    .secondary-button {
        background-color: #6c757d !important;
        color: white !important;
    }
    .primary-button {
        background-color: #007bff !important;
        color: white !important;
    }
    .stDataFrame {
        border-radius: 18px;
        box-shadow: 0 2px 12px rgba(0,0,0,0.07);
    }
    .metric-container {
        background: linear-gradient(90deg, #f8f9fa 0%, #ffffff 100%);
        padding: 1rem;
        border-radius: 10px;
        border-left: 4px solid #007bff;
        margin: 0.5rem 0;
    }
    h1 {
        text-align: center;
        color: #2c3e50;
        margin-bottom: 2rem;
    }
    .stSelectbox > div > div > div {
        background-color: #f8f9fa;
        border-radius: 8px;
    }
</style>
""", unsafe_allow_html=True)

# Auto-refresh data daily
@st.cache_data(ttl=24*60*60)  # Cache for 24 hours
def load_data():
    """Load stock and benchmark data with daily refresh"""
    return load_stock_data('stock_data'), load_benchmark_data('stock_data')

# Load data
try:
    stock_data, benchmark_data = load_data()
    if stock_data is None or benchmark_data is None:
        st.error("❌ Error loading data. Please check your data files.")
        st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {str(e)}")
    st.stop()

# Helper functions (same logic as your Dash app)
def calculate_profitable_trades(signals_df, benchmark_df):
    """Calculate profitable trades from a DataFrame of signals."""
    if signals_df.empty:
        return pd.DataFrame()

    trades = []
    signals_df = signals_df.sort_values(by=['ticker', 'Date'])

    for ticker, group in signals_df.groupby('ticker'):
        buy_signal = None
        for index, row in group.iterrows():
            if row['signal'] == 'buy':
                buy_signal = row
            elif row['signal'] == 'sell' and buy_signal is not None:
                buy_date = pd.to_datetime(buy_signal['Date'])
                sell_date = pd.to_datetime(row['Date'])

                benchmark_buy_price = benchmark_df.loc[benchmark_df['Date'] == buy_date, 'Close'].values[0]
                benchmark_sell_price = benchmark_df.loc[benchmark_df['Date'] == sell_date, 'Close'].values[0]
                
                return_pct = ((row['Close'] - buy_signal['Close']) / buy_signal['Close']) * 100
                benchmark_return_pct = ((benchmark_sell_price - benchmark_buy_price) / benchmark_buy_price) * 100

                if return_pct > benchmark_return_pct:
                    trades.append({
                        'Ticker': ticker,
                        'buy_date': buy_date.strftime('%Y-%m-%d'),
                        'price_at_buy': buy_signal['Close'],
                        'sell_date': sell_date.strftime('%Y-%m-%d'),
                        'price_at_sell': row['Close'],
                        'return_pct': return_pct,
                        'NSDAQ100etf_return_pct': benchmark_return_pct
                    })
                
                buy_signal = None

    return pd.DataFrame(trades)

def can_remove_signal(ticker_signals, signal_to_remove):
    """Check if a signal can be removed based on trading logic."""
    signal_index = -1
    for i, s in enumerate(ticker_signals):
        if (pd.to_datetime(s['Date']).strftime('%Y-%m-%d') == pd.to_datetime(signal_to_remove['Date']).strftime('%Y-%m-%d') and 
            s['ticker'] == signal_to_remove['ticker'] and 
            s['signal'] == signal_to_remove['signal']):
            signal_index = i
            break
            
    if signal_index == -1:
        return False, "Signal not found."

    if signal_to_remove['signal'] == 'buy':
        if signal_index < len(ticker_signals) - 1:
            next_signal = ticker_signals[signal_index + 1]
            if next_signal['signal'] == 'sell':
                return False, "Cannot remove a 'buy' that has a corresponding 'sell'. Remove the 'sell' first."
    
    return True, ""

def update_data():
    """Update stock data by running tech_data.py script"""
    try:
        result = subprocess.run(['python', 'tech_data.py'], 
                              capture_output=True, text=True, timeout=300)
        if result.returncode == 0:
            st.cache_data.clear()
            return True, "✅ Data updated successfully!"
        else:
            return False, f"❌ Error updating data: {result.stderr}"
    except subprocess.TimeoutExpired:
        return False, "❌ Data update timed out (5 minutes)"
    except Exception as e:
        return False, f"❌ Error updating data: {str(e)}"

# Initialize session state
if 'signals' not in st.session_state:
    st.session_state.signals = []
if 'profitable_trades' not in st.session_state:
    st.session_state.profitable_trades = []
if 'selected_point' not in st.session_state:
    st.session_state.selected_point = None
if 'last_click_data' not in st.session_state:
    st.session_state.last_click_data = None
if 'warning_message' not in st.session_state:
    st.session_state.warning_message = ""
if 'success_message' not in st.session_state:
    st.session_state.success_message = ""
if 'selected_date_from_chart' not in st.session_state:
    st.session_state.selected_date_from_chart = None
if 'chart_click_info' not in st.session_state:
    st.session_state.chart_click_info = None

# Clear messages after displaying
def clear_messages():
    st.session_state.warning_message = ""
    st.session_state.success_message = ""

# Title
st.title("📈 Financial Analysis Tool")

# Instructions
with st.expander("📋 How to Use This App", expanded=False):
    st.markdown("""
    ### 🚀 Two Ways to Add Signals:
    
    **Method 1: Chart Click (Faster) ⚡**
    1. Click directly on any candlestick or point on the chart
    2. The date will be automatically selected in the sidebar
    3. Use the "Quick Buy" or "Quick Sell" buttons that appear
    
    **Method 2: Manual Date Selection 📅**
    1. Use the date picker in the sidebar to choose a date
    2. Click "Buy Signal" or "Sell Signal" buttons
    
    ### 📊 Features:
    - **Technical Indicators**: Toggle various indicators on/off
    - **Signal Management**: Add, remove, and save buy/sell signals
    - **Profitable Trades**: Automatic calculation of profitable trade pairs
    - **Data Export**: Download signals and trades as CSV files
    - **Daily Updates**: Stock data updates automatically via GitHub Actions
    
    ### 📈 Trading Rules:
    - Must start with a Buy signal
    - Cannot have consecutive Buy or Sell signals
    - Each Buy must be followed by a Sell to complete a trade pair
    """)

# Display messages
if st.session_state.warning_message:
    st.error(st.session_state.warning_message)
if st.session_state.success_message:
    st.success(st.session_state.success_message)

# Sidebar for controls
with st.sidebar:
    st.header("🎛️ Controls")
    
    # Ticker selection
    selected_ticker = st.selectbox(
        "📊 Select Ticker",
        options=sorted(stock_data['ticker'].unique()),
        index=0
    )
    
    # Technical indicators
    st.subheader("📈 Technical Indicators")
    
    # Create columns for better layout
    col1, col2 = st.columns(2)
    
    with col1:
        bb = st.checkbox("Bollinger Bands", key="bb")
        rsi = st.checkbox("RSI", key="rsi")
        macd = st.checkbox("MACD", key="macd")
        ichimoku = st.checkbox("Ichimoku Cloud", key="ichimoku")
        adx = st.checkbox("ADX", key="adx")
    
    with col2:
        psar = st.checkbox("Parabolic SAR", key="psar")
        donchian = st.checkbox("Donchian Channels", key="donchian")
        roc = st.checkbox("Rate of Change", key="roc")
        ewo = st.checkbox("Elliott Wave Oscillator", key="ewo")
    
    selected_indicators = []
    if bb: selected_indicators.append('bb')
    if rsi: selected_indicators.append('rsi')
    if macd: selected_indicators.append('macd')
    if ichimoku: selected_indicators.append('ichimoku')
    if adx: selected_indicators.append('adx')
    if psar: selected_indicators.append('psar')
    if donchian: selected_indicators.append('donchian')
    if roc: selected_indicators.append('roc')
    if ewo: selected_indicators.append('ewo')
    
    st.divider()
    
    # Signal management
    st.subheader("🎯 Signal Management")
    
    # Show chart click information if available
    if st.session_state.chart_click_info:
        st.success(f"📍 Chart Click: {st.session_state.chart_click_info}")
    
    # Date picker for signals - use chart selected date if available
    default_date = datetime.now().date()
    if st.session_state.selected_date_from_chart:
        default_date = st.session_state.selected_date_from_chart
    
    signal_date = st.date_input(
        "📅 Select Date for Signal", 
        default_date,
        help="Choose a date to add buy/sell signals (or click on the chart above)"
    )
    
    # Convert to datetime for comparison
    signal_datetime = pd.to_datetime(signal_date)
    
    # Check if data exists for selected date
    date_data = stock_data[
        (stock_data['ticker'] == selected_ticker) & 
        (stock_data['Date'].dt.date == signal_date)
    ]
    
    if not date_data.empty:
        st.success(f"✅ Data available for {signal_date}")
        price = date_data['Close'].iloc[0]
        high_price = date_data['High'].iloc[0]
        low_price = date_data['Low'].iloc[0]
        open_price = date_data['Open'].iloc[0]
        volume = date_data['Volume'].iloc[0]
        
        # Show detailed price info
        st.markdown(f"""
        **� Price Data for {signal_date}:**
        - �💰 **Close:** ${price:.2f}
        - 📈 **High:** ${high_price:.2f}
        - 📉 **Low:** ${low_price:.2f}
        - 🔓 **Open:** ${open_price:.2f}
        - 📊 **Volume:** {volume:,}
        """)
        
        # Check if signal already exists for this date
        existing_signal = any(
            pd.to_datetime(s['Date']).date() == signal_date and s['ticker'] == selected_ticker 
            for s in st.session_state.signals
        )
        if existing_signal:
            st.warning("⚠️ Signal already exists for this date")
    else:
        st.error(f"❌ No data available for {signal_date}")
    
    # Quick action buttons for chart clicks
    if st.session_state.selected_date_from_chart and st.session_state.selected_date_from_chart == signal_date:
        st.markdown("**🚀 Quick Actions (from chart click):**")
        
        col_quick_buy, col_quick_sell = st.columns(2)
        
        with col_quick_buy:
            if st.button("⚡ Quick Buy", key="quick_buy", type="primary", disabled=date_data.empty):
                # Same logic as regular buy button
                clear_messages()
                
                existing_signal = any(
                    pd.to_datetime(s['Date']).date() == signal_date and s['ticker'] == selected_ticker 
                    for s in st.session_state.signals
                )
                if existing_signal:
                    st.session_state.warning_message = "Signal already exists for this date."
                    st.rerun()
                
                ticker_signals = [s for s in st.session_state.signals if s['ticker'] == selected_ticker]
                ticker_signals.sort(key=lambda x: x['Date'])
                last_signal_type = ticker_signals[-1]['signal'] if ticker_signals else None
                
                if last_signal_type == 'buy':
                    st.session_state.warning_message = "Cannot add a 'buy' after another 'buy'. Must be a 'sell'."
                    st.rerun()
                else:
                    new_signal = {
                        **date_data.to_dict('records')[0],
                        'signal': 'buy'
                    }
                    st.session_state.signals.append(new_signal)
                    st.session_state.success_message = "⚡ Quick Buy signal added!"
                    # Clear chart selection after adding signal
                    st.session_state.selected_date_from_chart = None
                    st.session_state.chart_click_info = None
                    st.rerun()
        
        with col_quick_sell:
            if st.button("⚡ Quick Sell", key="quick_sell", disabled=date_data.empty):
                # Same logic as regular sell button
                clear_messages()
                
                existing_signal = any(
                    pd.to_datetime(s['Date']).date() == signal_date and s['ticker'] == selected_ticker 
                    for s in st.session_state.signals
                )
                if existing_signal:
                    st.session_state.warning_message = "Signal already exists for this date."
                    st.rerun()
                
                ticker_signals = [s for s in st.session_state.signals if s['ticker'] == selected_ticker]
                ticker_signals.sort(key=lambda x: x['Date'])
                last_signal_type = ticker_signals[-1]['signal'] if ticker_signals else None
                
                if last_signal_type != 'buy':
                    st.session_state.warning_message = "Cannot add a 'sell' without a preceding 'buy'."
                    st.rerun()
                else:
                    new_signal = {
                        **date_data.to_dict('records')[0],
                        'signal': 'sell'
                    }
                    st.session_state.signals.append(new_signal)
                    
                    signals_df = pd.DataFrame(st.session_state.signals)
                    st.session_state.profitable_trades = calculate_profitable_trades(signals_df, benchmark_data).to_dict('records')
                    
                    st.session_state.success_message = "⚡ Quick Sell signal added!"
                    # Clear chart selection after adding signal
                    st.session_state.selected_date_from_chart = None
                    st.session_state.chart_click_info = None
                    st.rerun()
        
        st.divider()
    
    # Clear chart selection button
    if st.session_state.selected_date_from_chart:
        if st.button("🗑️ Clear Chart Selection", key="clear_chart_selection"):
            st.session_state.selected_date_from_chart = None
            st.session_state.chart_click_info = None
            st.rerun()
    
    # Regular signal buttons
    st.markdown("**📅 Manual Signal Entry:**")
    col_buy, col_sell = st.columns(2)
    
    with col_buy:
        if st.button("🟢 Buy Signal", key="buy_btn", type="primary", disabled=date_data.empty):
            clear_messages()
            
            # Check if signal for this date already exists
            existing_signal = any(
                pd.to_datetime(s['Date']).date() == signal_date and s['ticker'] == selected_ticker 
                for s in st.session_state.signals
            )
            if existing_signal:
                st.session_state.warning_message = "Signal already exists for this date."
                st.rerun()
            
            # Check trading logic
            ticker_signals = [s for s in st.session_state.signals if s['ticker'] == selected_ticker]
            ticker_signals.sort(key=lambda x: x['Date'])
            last_signal_type = ticker_signals[-1]['signal'] if ticker_signals else None
            
            if last_signal_type == 'buy':
                st.session_state.warning_message = "Cannot add a 'buy' after another 'buy'. Must be a 'sell'."
                st.rerun()
            else:
                new_signal = {
                    **date_data.to_dict('records')[0],
                    'signal': 'buy'
                }
                st.session_state.signals.append(new_signal)
                st.session_state.success_message = "✅ Buy signal added!"
                st.rerun()
    
    with col_sell:
        if st.button("🔴 Sell Signal", key="sell_btn", disabled=date_data.empty):
            clear_messages()
            
            # Check if signal for this date already exists
            existing_signal = any(
                pd.to_datetime(s['Date']).date() == signal_date and s['ticker'] == selected_ticker 
                for s in st.session_state.signals
            )
            if existing_signal:
                st.session_state.warning_message = "Signal already exists for this date."
                st.rerun()
            
            # Check trading logic
            ticker_signals = [s for s in st.session_state.signals if s['ticker'] == selected_ticker]
            ticker_signals.sort(key=lambda x: x['Date'])
            last_signal_type = ticker_signals[-1]['signal'] if ticker_signals else None
            
            if last_signal_type != 'buy':
                st.session_state.warning_message = "Cannot add a 'sell' without a preceding 'buy'."
                st.rerun()
            else:
                new_signal = {
                    **date_data.to_dict('records')[0],
                    'signal': 'sell'
                }
                st.session_state.signals.append(new_signal)
                
                # Recalculate profitable trades
                signals_df = pd.DataFrame(st.session_state.signals)
                st.session_state.profitable_trades = calculate_profitable_trades(signals_df, benchmark_data).to_dict('records')
                
                st.session_state.success_message = "✅ Sell signal added!"
                st.rerun()
    
    # Remove last signal button
    if st.button("🗑️ Remove Last Signal", type="secondary", key="remove_btn"):
        clear_messages()
        ticker_signals = [s for s in st.session_state.signals if s['ticker'] == selected_ticker]
        if ticker_signals:
            # Find the last signal for current ticker
            ticker_signals.sort(key=lambda x: x['Date'])
            last_signal = ticker_signals[-1]
            
            # Remove from main signals list
            st.session_state.signals = [s for s in st.session_state.signals 
                                      if not (s['Date'] == last_signal['Date'] and 
                                            s['ticker'] == last_signal['ticker'] and 
                                            s['signal'] == last_signal['signal'])]
            
            # Recalculate profitable trades
            if st.session_state.signals:
                signals_df = pd.DataFrame(st.session_state.signals)
                st.session_state.profitable_trades = calculate_profitable_trades(signals_df, benchmark_data).to_dict('records')
            else:
                st.session_state.profitable_trades = []
            
            st.session_state.success_message = "✅ Last signal removed!"
            st.rerun()
        else:
            st.session_state.warning_message = "No signals to remove for this ticker."
            st.rerun()
    
    # Save signals button
    if st.button("💾 Save Signals", key="save_btn", type="primary"):
        clear_messages()
        if st.session_state.signals:
            signals_df = pd.DataFrame(st.session_state.signals)
            signals_df.to_csv('signals.csv', index=False)
            
            profitable_trades_df = pd.DataFrame(st.session_state.profitable_trades)
            if not profitable_trades_df.empty:
                profitable_trades_df.to_csv('profitable_trades.csv', index=False)
            
            st.session_state.success_message = "💾 Signals and profitable trades saved successfully!"
            st.rerun()
        else:
            st.session_state.warning_message = "No signals to save."
            st.rerun()
    
    st.divider()
    
    # Data management
    st.subheader("💾 Data Management")
    
    if st.button("🔄 Update Stock Data", key="update_data_btn", type="primary"):
        clear_messages()
        with st.spinner("Updating stock data... This may take a few minutes..."):
            success, message = update_data()
            if success:
                st.session_state.success_message = message
                st.rerun()
            else:
                st.session_state.warning_message = message
                st.rerun()

# Main content area
df = stock_data[stock_data['ticker'] == selected_ticker].copy()
df.sort_values('Date', inplace=True)

# Create chart with TradingView style
INCREASING_COLOR = '#26a69a'
DECREASING_COLOR = '#ef5350'
GRID_COLOR = '#EAEAEA'

# Determine rows needed for indicators
indicator_rows = [ind for ind in ['rsi', 'macd', 'adx', 'roc', 'ewo'] if ind in selected_indicators]
num_rows = 2 + len(indicator_rows)
row_heights = [0.7] + [0.15] * (len(indicator_rows) + 1)

specs = [[{"secondary_y": False}], [{"secondary_y": False}]]
for _ in indicator_rows:
    specs.append([{"secondary_y": False}])

fig = make_subplots(
    rows=num_rows, cols=1, shared_xaxes=True,
    vertical_spacing=0.02,
    row_heights=row_heights,
    specs=specs,
    subplot_titles=[f"{selected_ticker} Stock Chart", "Volume"] + [ind.upper() for ind in indicator_rows]
)

# Add candlestick
fig.add_trace(go.Candlestick(
    x=df['Date'],
    open=df['Open'],
    high=df['High'],
    low=df['Low'],
    close=df['Close'],
    name='Price',
    increasing_fillcolor=INCREASING_COLOR,
    increasing_line_color=INCREASING_COLOR,
    decreasing_fillcolor=DECREASING_COLOR,
    decreasing_line_color=DECREASING_COLOR
), row=1, col=1)

# Add price-based indicators
if 'bb' in selected_indicators and all(c in df.columns for c in ['bb_upper', 'bb_lower', 'bb_middle']):
    fig.add_trace(go.Scatter(x=df['Date'], y=df['bb_upper'], mode='lines', name='BB Upper', 
                           line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['bb_lower'], mode='lines', name='BB Lower', 
                           line=dict(color='gray', width=1, dash='dash')), row=1, col=1)

if 'ichimoku' in selected_indicators and all(c in df.columns for c in ['ichimoku_senkou_a', 'ichimoku_senkou_b']):
    fig.add_trace(go.Scatter(x=df['Date'], y=df['ichimoku_senkou_a'], mode='lines', name='Ichimoku A', 
                           line=dict(color='rgba(0, 255, 0, 0.2)')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['ichimoku_senkou_b'], mode='lines', name='Ichimoku B', 
                           line=dict(color='rgba(255, 0, 0, 0.2)'), fill='tonexty'), row=1, col=1)

if 'psar' in selected_indicators and 'psar' in df.columns:
    fig.add_trace(go.Scatter(x=df['Date'], y=df['psar'], mode='markers', name='Parabolic SAR', 
                           marker=dict(color='purple', size=4)), row=1, col=1)

if 'donchian' in selected_indicators and all(c in df.columns for c in ['donchian_upper', 'donchian_lower']):
    fig.add_trace(go.Scatter(x=df['Date'], y=df['donchian_upper'], mode='lines', name='Donchian Upper', 
                           line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)
    fig.add_trace(go.Scatter(x=df['Date'], y=df['donchian_lower'], mode='lines', name='Donchian Lower', 
                           line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)

# Add volume
volume_colors = [INCREASING_COLOR if row['Close'] >= row['Open'] else DECREASING_COLOR 
                for index, row in df.iterrows()]
fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'], marker_color=volume_colors, name='Volume'), row=2, col=1)

# Add indicator traces
current_row = 3
for indicator in indicator_rows:
    if indicator == 'rsi' and 'rsi' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['rsi'], mode='lines', name='RSI'), row=current_row, col=1)
        fig.update_yaxes(title_text="RSI", side='right', gridcolor=GRID_COLOR, row=current_row, col=1)
    elif indicator == 'macd' and all(c in df.columns for c in ['macd_line', 'macd_signal', 'macd_histogram']):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['macd_line'], mode='lines', name='MACD Line'), row=current_row, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['macd_signal'], mode='lines', name='MACD Signal'), row=current_row, col=1)
        fig.add_trace(go.Bar(x=df['Date'], y=df['macd_histogram'], name='MACD Hist'), row=current_row, col=1)
        fig.update_yaxes(title_text="MACD", side='right', gridcolor=GRID_COLOR, row=current_row, col=1)
    elif indicator == 'adx' and 'adx' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['adx'], mode='lines', name='ADX'), row=current_row, col=1)
        fig.update_yaxes(title_text="ADX", side='right', gridcolor=GRID_COLOR, row=current_row, col=1)
    elif indicator == 'roc' and 'roc' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['roc'], mode='lines', name='ROC'), row=current_row, col=1)
        fig.update_yaxes(title_text="ROC", side='right', gridcolor=GRID_COLOR, row=current_row, col=1)
    elif indicator == 'ewo' and 'elliott_wave_oscillator' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['elliott_wave_oscillator'], mode='lines', name='EWO'), row=current_row, col=1)
        fig.update_yaxes(title_text="EWO", side='right', gridcolor=GRID_COLOR, row=current_row, col=1)
    current_row += 1

# Add signal annotations and chart selection indicator
for signal in st.session_state.signals:
    if signal['ticker'] == selected_ticker:
        signal_date = pd.to_datetime(signal['Date'])
        signal_df = df[df['Date'] == signal_date]
        if not signal_df.empty:
            if signal['signal'] == 'buy':
                fig.add_annotation(
                    x=signal_date, y=signal_df.iloc[0]['Low'],
                    text="B", showarrow=True, arrowhead=2,
                    ax=0, ay=20, bgcolor=INCREASING_COLOR, 
                    font=dict(color="white", size=12), bordercolor="white"
                )
            elif signal['signal'] == 'sell':
                fig.add_annotation(
                    x=signal_date, y=signal_df.iloc[0]['High'],
                    text="S", showarrow=True, arrowhead=2,
                    ax=0, ay=-20, bgcolor=DECREASING_COLOR,
                    font=dict(color="white", size=12), bordercolor="white"
                )

# Add indicator for currently selected date from chart click
if st.session_state.selected_date_from_chart:
    selected_date_dt = pd.to_datetime(st.session_state.selected_date_from_chart)
    selected_df = df[df['Date'].dt.date == st.session_state.selected_date_from_chart]
    if not selected_df.empty:
        # Add a highlighted marker for the selected date
        fig.add_annotation(
            x=selected_date_dt, 
            y=selected_df.iloc[0]['High'] * 1.02,  # Slightly above the high
            text="📍", 
            showarrow=True, 
            arrowhead=2,
            ax=0, 
            ay=-10, 
            bgcolor="orange", 
            font=dict(color="white", size=14), 
            bordercolor="white",
            opacity=0.8
        )

# Display chart with enhanced interaction
try:
    from streamlit_plotly_events import plotly_events
    HAS_PLOTLY_EVENTS = True
except ImportError:
    HAS_PLOTLY_EVENTS = False
    st.warning("📌 For enhanced chart interaction (click to select dates), install: `pip install streamlit-plotly-events`")

# Configure the chart for better click detection
fig.update_layout(
    height=800,
    xaxis_rangeslider_visible=False,
    showlegend=False,
    plot_bgcolor='white',
    paper_bgcolor='white',
    margin=dict(l=10, r=50, t=50, b=10),
    # Enable click events
    clickmode='event+select'
)

# Update axes
fig.update_yaxes(side='right', tickfont=dict(size=12, color='#333'), gridcolor=GRID_COLOR, row=1, col=1)
fig.update_yaxes(showticklabels=False, gridcolor=GRID_COLOR, row=2, col=1)
fig.update_xaxes(gridcolor=GRID_COLOR, tickfont=dict(size=12, color='#787878'), showticklabels=True)

# Hide x-axis labels on all but bottom chart
for i in range(1, num_rows):
    fig.update_xaxes(showticklabels=False, row=i, col=1)

# Display chart and capture click events
if HAS_PLOTLY_EVENTS:
    selected_points = plotly_events(fig, click_event=True, hover_event=False, select_event=False, key="chart_events")
    
    # Process chart click events
    if selected_points:
        try:
            point_data = selected_points[0]
            if 'x' in point_data:
                clicked_date = pd.to_datetime(point_data['x']).date()
                
                # Update session state with clicked date
                if clicked_date != st.session_state.selected_date_from_chart:
                    st.session_state.selected_date_from_chart = clicked_date
                    
                    # Get price info for the clicked date
                    clicked_data = stock_data[
                        (stock_data['ticker'] == selected_ticker) & 
                        (stock_data['Date'].dt.date == clicked_date)
                    ]
                    
                    if not clicked_data.empty:
                        price = clicked_data['Close'].iloc[0]
                        st.session_state.chart_click_info = f"Date: {clicked_date}, Price: ${price:.2f}"
                    else:
                        st.session_state.chart_click_info = f"Date: {clicked_date} (No data)"
                    
                    st.rerun()
        except Exception as e:
            pass  # Ignore click parsing errors
else:
    # Fallback to regular plotly chart
    st.plotly_chart(fig, use_container_width=True, key="main_chart")
    st.info("💡 **Tip:** You can manually select dates using the calendar in the sidebar, or install `streamlit-plotly-events` for chart clicking.")

# Display tables
col1, col2 = st.columns(2)

with col1:
    st.subheader("🎯 Selected Signals")
    if st.session_state.signals:
        signals_df = pd.DataFrame(st.session_state.signals)
        signals_df['Date'] = pd.to_datetime(signals_df['Date']).dt.strftime('%Y-%m-%d')
        display_df = signals_df[['Date', 'ticker', 'Close', 'signal']].copy()
        display_df['Close'] = display_df['Close'].apply(lambda x: f"${x:.2f}")
        
        # Add signal deletion interface
        st.write("**Remove Specific Signal:**")
        
        # Create a selection list for signals to delete
        signal_options = []
        for i, row in display_df.iterrows():
            signal_options.append(f"{row['Date']} - {row['ticker']} - {row['signal']} - {row['Close']}")
        
        if signal_options:
            selected_signal_to_delete = st.selectbox(
                "Select signal to remove:",
                options=[""] + signal_options,
                key="signal_delete_select"
            )
            
            if selected_signal_to_delete and st.button("🗑️ Remove Selected Signal", key="remove_selected"):
                clear_messages()
                
                # Parse the selected signal
                parts = selected_signal_to_delete.split(" - ")
                date_str, ticker, signal_type, price_str = parts
                
                # Find the corresponding signal in storage
                signal_to_remove = None
                for signal in st.session_state.signals:
                    if (pd.to_datetime(signal['Date']).strftime('%Y-%m-%d') == date_str and
                        signal['ticker'] == ticker and
                        signal['signal'] == signal_type):
                        signal_to_remove = signal
                        break
                
                if signal_to_remove:
                    # Check if removal is allowed
                    ticker_signals = [s for s in st.session_state.signals if s['ticker'] == ticker]
                    ticker_signals.sort(key=lambda x: x['Date'])
                    
                    can_remove, warning_msg = can_remove_signal(ticker_signals, signal_to_remove)
                    
                    if can_remove:
                        # Remove the signal
                        st.session_state.signals = [s for s in st.session_state.signals 
                                                  if not (s['Date'] == signal_to_remove['Date'] and
                                                        s['ticker'] == signal_to_remove['ticker'] and
                                                        s['signal'] == signal_to_remove['signal'])]
                        
                        # Recalculate profitable trades
                        if st.session_state.signals:
                            signals_df = pd.DataFrame(st.session_state.signals)
                            st.session_state.profitable_trades = calculate_profitable_trades(signals_df, benchmark_data).to_dict('records')
                        else:
                            st.session_state.profitable_trades = []
                        
                        st.session_state.success_message = "✅ Signal removed successfully!"
                        st.rerun()
                    else:
                        st.session_state.warning_message = warning_msg
                        st.rerun()
        
        st.divider()
        
        # Create styled dataframe
        def style_signals(df):
            def color_signal(val):
                if val == 'buy':
                    return 'background-color: #d4edda; color: #155724; font-weight: bold'
                elif val == 'sell':
                    return 'background-color: #f8d7da; color: #721c24; font-weight: bold'
                return 'font-weight: bold'
            
            return df.style.applymap(color_signal, subset=['signal']).set_properties(**{
                'text-align': 'center',
                'font-size': '16px',
                'padding': '10px'
            }).set_table_styles([
                {'selector': 'thead th', 'props': [
                    ('background-color', '#fff'),
                    ('font-weight', 'bold'),
                    ('font-size', '20px'),
                    ('text-align', 'center'),
                    ('border', 'none')
                ]},
                {'selector': 'tbody td', 'props': [
                    ('border', 'none')
                ]},
                {'selector': '', 'props': [
                    ('border-radius', '18px'),
                    ('box-shadow', '0 2px 12px rgba(0,0,0,0.07)')
                ]}
            ])
        
        st.dataframe(
            style_signals(display_df), 
            use_container_width=True, 
            hide_index=True,
            height=300
        )
        
        # Download signals button
        csv = signals_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Signals CSV",
            data=csv,
            file_name=f"signals_export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            type="secondary"
        )
        
        # Show signal count
        buy_count = len([s for s in st.session_state.signals if s['signal'] == 'buy'])
        sell_count = len([s for s in st.session_state.signals if s['signal'] == 'sell'])
        st.info(f"📊 Total: {buy_count} Buy signals, {sell_count} Sell signals")
        
    else:
        st.info("No signals added yet. Select a date and use the signal buttons.")

with col2:
    st.subheader("💰 Profitable Trades")
    if st.session_state.profitable_trades:
        trades_df = pd.DataFrame(st.session_state.profitable_trades)
        
        # Format the display
        display_trades = trades_df.copy()
        display_trades['price_at_buy'] = display_trades['price_at_buy'].apply(lambda x: f"${x:.2f}")
        display_trades['price_at_sell'] = display_trades['price_at_sell'].apply(lambda x: f"${x:.2f}")
        display_trades['return_pct'] = display_trades['return_pct'].apply(lambda x: f"{x:.2f}%")
        display_trades['NSDAQ100etf_return_pct'] = display_trades['NSDAQ100etf_return_pct'].apply(lambda x: f"{x:.2f}%")
        
        # Rename columns to match Dash version
        display_trades.columns = [
            'Ticker', 'Buy Date', 'Buy Price', 'Sell Date', 'Sell Price', 'Return %', 'Benchmark Return %'
        ]
        
        st.dataframe(
            display_trades.style.set_properties(**{
                'text-align': 'center',
                'font-weight': 'bold',
                'font-size': '16px',
                'padding': '10px'
            }).set_table_styles([
                {'selector': 'thead th', 'props': [
                    ('background-color', '#fff'),
                    ('font-weight', 'bold'),
                    ('font-size', '20px'),
                    ('text-align', 'center'),
                    ('border', 'none')
                ]},
                {'selector': 'tbody td', 'props': [
                    ('border', 'none')
                ]},
                {'selector': '', 'props': [
                    ('border-radius', '18px'),
                    ('box-shadow', '0 2px 12px rgba(0,0,0,0.07)')
                ]}
            ]), 
            use_container_width=True, 
            hide_index=True,
            height=300
        )
        
        # Download profitable trades button
        csv = trades_df.to_csv(index=False)
        st.download_button(
            label="📥 Download Profitable Trades CSV",
            data=csv,
            file_name=f"profitable_trades_export_{datetime.now().strftime('%Y%m%d')}.csv",
            mime="text/csv",
            use_container_width=True,
            type="secondary"
        )
        
        # Show summary statistics
        avg_return = trades_df['return_pct'].mean()
        avg_benchmark = trades_df['NSDAQ100etf_return_pct'].mean()
        total_trades = len(trades_df)
        
        st.markdown(f"""
        <div class="metric-container">
            <h4>📊 Trade Summary</h4>
            <p><strong>Total Profitable Trades:</strong> {total_trades}</p>
            <p><strong>Average Return:</strong> {avg_return:.2f}%</p>
            <p><strong>Average Benchmark Return:</strong> {avg_benchmark:.2f}%</p>
            <p><strong>Outperformance:</strong> {avg_return - avg_benchmark:.2f}%</p>
        </div>
        """, unsafe_allow_html=True)
        
    else:
        st.info("No profitable trades yet. Complete some buy-sell pairs to see results.")

# Footer
st.divider()
st.markdown("""
<div style="text-align: center; padding: 2rem; color: #6c757d;">
    <strong>📈 Financial Analysis Tool</strong> | Built with Streamlit | 
    <em>Migrated from Dash for free deployment</em> | Data updates daily via GitHub Actions
</div>
""", unsafe_allow_html=True)

# Clear messages at the end
if st.session_state.warning_message or st.session_state.success_message:
    # Clear messages after 3 seconds (simulation)
    import time
    if 'message_time' not in st.session_state:
        st.session_state.message_time = time.time()
    elif time.time() - st.session_state.message_time > 3:
        clear_messages()
