import dash
from dash import dcc, html, Input, Output, State, callback_context, dash_table
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
import os
from datetime import timedelta
from data_utils import (
    load_stock_data, 
    load_benchmark_data, 
    get_available_tickers, 
    get_ticker_data
)
# Initialize app first
app = dash.Dash(__name__, external_stylesheets=['/assets/styles.css'])
server = app.server  # For deployment

# Load data from local files
try:
    print("Loading data from local files...")
    stock_data = load_stock_data('stock_data')
    benchmark_data = load_benchmark_data('stock_data')
    available_tickers = get_available_tickers('stock_data')
    
    if stock_data.empty or not available_tickers:
        raise Exception("No data available in stock_data directory")
    
    print(f"Successfully loaded local data for {len(available_tickers)} tickers")
    
except Exception as e:
    print(f"Local data loading failed: {e}")
    # Create empty data to prevent app crash
    stock_data = pd.DataFrame()
    benchmark_data = pd.DataFrame()
    available_tickers = []

# --- UI Enhancements ---
# Color palette for tickers
TICKER_COLORS = [
    '#E6F3FF', '#F0FFF0', '#FFF5E6', '#F5F5F5', '#E6E6FA', 
    '#FFF0F5', '#F0F8FF', '#FAEBD7', '#F5FFFA', '#FFFACD'
]

# Handle empty data gracefully
if available_tickers:
    unique_tickers = available_tickers
    ticker_color_map = {ticker: TICKER_COLORS[i % len(TICKER_COLORS)] for i, ticker in enumerate(unique_tickers)}
    default_ticker = unique_tickers[0]
else:
    unique_tickers = []
    ticker_color_map = {}
    default_ticker = None

# App initialization
app.layout = html.Div([
    # Main Content Area
    html.Div([
        html.H1('Financial Analysis Tool'),
        
        # Show error message if no data available
        html.Div(id='data-status', children=[
            html.Div("⚠️ No data available. Please check database connection or ensure local CSV files are present.", 
                    style={'color': 'red', 'textAlign': 'center', 'padding': '20px', 'fontWeight': 'bold'})
        ] if not available_tickers else []),
        
        dcc.Dropdown(
            id='ticker-dropdown',
            options=[{'label': ticker, 'value': ticker} for ticker in available_tickers],
            value=default_ticker,
            disabled=not available_tickers
        ),
        
        html.Div([
            html.H4('Technical Indicators', style={'margin-top': '20px', 'margin-bottom': '10px'}),
            dcc.Checklist(
                id='indicator-checklist',
                options=[
                    {'label': 'Bollinger Bands', 'value': 'bb'},
                    {'label': 'RSI', 'value': 'rsi'},
                    {'label': 'MACD', 'value': 'macd'},
                    {'label': 'Ichimoku Cloud', 'value': 'ichimoku'},
                    {'label': 'ADX', 'value': 'adx'},
                    {'label': 'Parabolic SAR', 'value': 'psar'},
                    {'label': 'Donchian Channels', 'value': 'donchian'},
                    {'label': 'Rate of Change', 'value': 'roc'},
                    {'label': 'Elliott Wave Oscillator', 'value': 'ewo'},
                ],
                value=[],
                labelStyle={'display': 'inline-block', 'margin-right': '15px', 'cursor': 'pointer'},
                style={'padding-bottom': '15px', 'border-bottom': '1px solid #ddd'}
            ),
        ], style={'textAlign': 'center'}),

        dcc.Graph(id='stock-graph'),
        
        html.Div([
            html.Button('Buy Signal', id='buy-button', n_clicks=0, style={'margin-right': '10px'}),
            html.Button('Sell Signal', id='sell-button', n_clicks=0, style={'margin-right': '10px'}),
            html.Button('Remove Last Signal', id='remove-last-button', n_clicks=0, className='button2-primary', style={'margin-right': '10px'}),
            html.Button('Save Signals', id='save-button', n_clicks=0, className='button-primary'),
        ], style={'textAlign': 'center', 'padding': '20px', 'margin-top': '50px'}),
        
        html.Div(id='selected-point-info', style={'margin-top': '10px', 'textAlign': 'center'}),
        html.Div(id='save-status', style={'margin-top': '10px', 'textAlign': 'center'}),
        html.Div(id='trade-profitability-status', style={'margin-top': '10px', 'textAlign': 'center', 'fontWeight': 'bold'}),
        html.Div(id='edit-warning', style={'margin-top': '10px', 'textAlign': 'center', 'color': 'red', 'fontWeight': 'bold'}),

        html.Div([
            html.Div([
                html.H3('Selected Signals', style={'textAlign': 'center'}),
                dash_table.DataTable(
                    id='signals-table',
                    columns=[
                        {'name': 'Date', 'id': 'Date'},
                        {'name': 'Ticker', 'id': 'ticker'},
                        {'name': 'Close', 'id': 'Close', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                        {'name': 'Signal', 'id': 'signal'},
                    ],
                    data=[],
                    row_deletable=True,
                    style_table={
                        'height': '300px',
                        'overflowY': 'auto',
                        'width': '100%',
                        'borderRadius': '18px',
                        'boxShadow': '0 2px 12px rgba(0,0,0,0.07)'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'fontWeight': 'bold',
                        'border': 'none',
                        'fontSize': '16px',
                        'padding': '10px 0',
                    },
                    style_header={
                        'backgroundColor': '#fff',
                        'fontWeight': 'bold',
                        'fontSize': '20px',
                        'textAlign': 'center',
                        'border': 'none',
                    },
                    style_data_conditional=[],
                    fixed_rows={'headers': True}
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginRight': '2%'}),
            
            html.Div([
                html.H3('Profitable Trades', style={'textAlign': 'center'}),
                dash_table.DataTable(
                    id='profitable-trades-table',
                    columns=[
                        {'name': 'Ticker', 'id': 'Ticker'},
                        {'name': 'Buy Date', 'id': 'buy_date'},
                        {'name': 'Buy Price', 'id': 'price_at_buy', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                        {'name': 'Sell Date', 'id': 'sell_date'},
                        {'name': 'Sell Price', 'id': 'price_at_sell', 'type': 'numeric', 'format': {'specifier': '$,.2f'}},
                        {'name': 'Return %', 'id': 'return_pct', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                        {'name': 'Benchmark Return %', 'id': 'NSDAQ100etf_return_pct', 'type': 'numeric', 'format': {'specifier': '.2f'}},
                    ],
                    data=[],
                    style_table={
                        'height': '300px',
                        'overflowY': 'auto',
                        'width': '100%',
                        'borderRadius': '18px',
                        'boxShadow': '0 2px 12px rgba(0,0,0,0.07)'
                    },
                    style_cell={
                        'textAlign': 'center',
                        'fontWeight': 'bold',
                        'border': 'none',
                        'fontSize': '16px',
                        'padding': '10px 0',
                    },
                    style_header={
                        'backgroundColor': '#fff',
                        'fontWeight': 'bold',
                        'fontSize': '20px',
                        'textAlign': 'center',
                        'border': 'none',
                    },
                ),
            ], style={'width': '48%', 'display': 'inline-block', 'verticalAlign': 'top', 'marginLeft': '2%'})
        ], style={'display': 'flex', 'flexDirection': 'row', 'width': '100%'}),

        dcc.Store(id='signals-storage', data=[]),
        dcc.Store(id='profitable-trades-storage', data=[]),

        html.Div([
            html.Button("Download Signals CSV", id="btn_download_signals_csv", className="button-secondary", style={'marginRight': '10px'}),
            dcc.Download(id="download-signals-csv"),
            html.Button("Download Profitable Trades CSV", id="btn_download_profitable_csv", className="button-secondary"),
            dcc.Download(id="download-profitable-csv"),
        ], style={'textAlign': 'center', 'padding': '20px'}),

    ], className='main-content')
], className='app-container')


### FIX: Centralized function to calculate profitable trades. This is now the single source of truth.
def calculate_profitable_trades(signals_df, benchmark_df):
    """Calculate profitable trades from a DataFrame of signals."""
    if signals_df.empty:
        return pd.DataFrame()

    trades = []
    # Ensure signals are sorted correctly to process pairs
    signals_df = signals_df.sort_values(by=['ticker', 'Date'])

    for ticker, group in signals_df.groupby('ticker'):
        buy_signal = None
        for index, row in group.iterrows():
            if row['signal'] == 'buy':
                # If we encounter a new buy signal, the previous one is now orphaned.
                buy_signal = row
            elif row['signal'] == 'sell' and buy_signal is not None:
                # We have a complete buy-sell pair
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
                
                # Reset buy_signal to look for the next pair
                buy_signal = None

    return pd.DataFrame(trades)


### FIX: Centralized logic to check if a signal can be removed.
def can_remove_signal(ticker_signals, signal_to_remove):
    """
    Checks if a signal can be removed.
    A buy signal can only be removed if it's the last signal for that ticker, or if the next signal is also a buy.
    A sell signal can always be removed (it's the end of a pair).
    """
    signal_index = -1
    for i, s in enumerate(ticker_signals):
        # Find the exact signal to remove
        if (s['Date'] == signal_to_remove['Date'] and 
            s['ticker'] == signal_to_remove['ticker'] and 
            s['signal'] == signal_to_remove['signal']):
            signal_index = i
            break
            
    if signal_index == -1:
        return False, "Signal not found."

    if signal_to_remove['signal'] == 'buy':
        # If it's not the last signal in the sequence for this ticker...
        if signal_index < len(ticker_signals) - 1:
            next_signal = ticker_signals[signal_index + 1]
            # ...and the next signal is a sell, then this removal is invalid.
            if next_signal['signal'] == 'sell':
                return False, "Cannot remove a 'buy' that has a corresponding 'sell'. Remove the 'sell' first."
    
    # Sell signals or orphaned buys can always be removed
    return True, ""


### Enhanced Signal Validation Function
def validate_signal_placement(signal_type, signal_date, ticker, existing_signals_df):
    """
    Validates if a buy/sell signal can be placed at the given date.
    
    Rules:
    1. Sell must be after buy date (and at least 1 month after)
    2. Cannot place buy between existing buy-sell pair
    3. Cannot place sell between existing buy-sell pair
    4. Must follow buy-sell sequence
    
    Returns:
        tuple: (is_valid, error_message)
    """
    from datetime import timedelta
    
    # Convert signal_date to datetime if it's a string
    if isinstance(signal_date, str):
        signal_date = pd.to_datetime(signal_date)
    
    # Get existing signals for this ticker, sorted by date
    ticker_signals = existing_signals_df[existing_signals_df['ticker'] == ticker].copy()
    if not ticker_signals.empty:
        ticker_signals['Date'] = pd.to_datetime(ticker_signals['Date'])
        ticker_signals = ticker_signals.sort_values('Date')
    
    # Rule 1: Basic sequence validation (buy-sell alternating)
    if not ticker_signals.empty:
        last_signal = ticker_signals.iloc[-1]
        if signal_type == 'buy' and last_signal['signal'] == 'buy':
            return False, "Cannot add a 'buy' after another 'buy'. Must be a 'sell' first."
        elif signal_type == 'sell' and last_signal['signal'] != 'buy':
            return False, "Cannot add a 'sell' without a preceding 'buy'."
    else:
        # First signal must be a buy
        if signal_type == 'sell':
            return False, "First signal must be a 'buy', not a 'sell'."
    
    # Rule 2: Sell must be at least 1 month after buy
    if signal_type == 'sell' and not ticker_signals.empty:
        # Find the most recent buy signal
        buy_signals = ticker_signals[ticker_signals['signal'] == 'buy']
        if not buy_signals.empty:
            last_buy_date = buy_signals.iloc[-1]['Date']
            min_sell_date = last_buy_date + timedelta(days=30)  # 1 month = 30 days
            
            if signal_date < min_sell_date:
                return False, f"Sell signal must be at least 1 month after buy signal. Earliest allowed date: {min_sell_date.strftime('%Y-%m-%d')}"
    
    # Rule 3 & 4: Cannot place signals within existing complete trade periods
    if len(ticker_signals) >= 2:
        # Create pairs of buy-sell trades
        trades = []
        i = 0
        while i < len(ticker_signals) - 1:
            if (ticker_signals.iloc[i]['signal'] == 'buy' and 
                ticker_signals.iloc[i + 1]['signal'] == 'sell'):
                trades.append({
                    'buy_date': ticker_signals.iloc[i]['Date'],
                    'sell_date': ticker_signals.iloc[i + 1]['Date']
                })
                i += 2
            else:
                i += 1
        
        # Check if new signal falls within any existing trade period
        for trade in trades:
            if trade['buy_date'] < signal_date < trade['sell_date']:
                if signal_type == 'buy':
                    return False, f"Cannot place buy signal within existing trade period ({trade['buy_date'].strftime('%Y-%m-%d')} to {trade['sell_date'].strftime('%Y-%m-%d')})"
                elif signal_type == 'sell':
                    return False, f"Cannot place sell signal within existing trade period ({trade['buy_date'].strftime('%Y-%m-%d')} to {trade['sell_date'].strftime('%Y-%m-%d')})"
    
    # Rule 5: Signal date cannot be before any existing signal if it would break sequence
    for _, existing_signal in ticker_signals.iterrows():
        existing_date = existing_signal['Date']
        existing_type = existing_signal['signal']
        
        # If new signal is before an existing signal, check if it maintains proper sequence
        if signal_date < existing_date:
            # Count how many signals would be before the new signal
            signals_before = ticker_signals[ticker_signals['Date'] < signal_date]
            
            # The new signal position in sequence
            new_position = len(signals_before)
            
            # Check if this position should be buy (even position: 0,2,4...) or sell (odd position: 1,3,5...)
            should_be_buy = (new_position % 2 == 0)
            
            if (should_be_buy and signal_type != 'buy') or (not should_be_buy and signal_type != 'sell'):
                return False, f"Signal at this date would break the buy-sell sequence. Expected {'buy' if should_be_buy else 'sell'} signal."
    
    return True, ""


@app.callback(
    Output('stock-graph', 'figure'),
    Input('ticker-dropdown', 'value'),
    Input('signals-storage', 'data'),
    Input('indicator-checklist', 'value')
)
def update_graph(selected_ticker, signals, selected_indicators):
    if not selected_ticker or not available_tickers:
        # Return empty figure if no data
        return {
            'data': [],
            'layout': {
                'title': 'No data available',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Price'}
            }
        }
    
    # Get ticker data from local files
    try:
        df = get_ticker_data(selected_ticker, 'stock_data')
        if df.empty:
            # Fallback to pre-loaded data
            df = stock_data[stock_data['ticker'] == selected_ticker].copy()
    except Exception as e:
        print(f"Error loading ticker data: {e}")
        df = stock_data[stock_data['ticker'] == selected_ticker].copy() if not stock_data.empty else pd.DataFrame()
    
    if df.empty:
        return {
            'data': [],
            'layout': {
                'title': f'No data available for {selected_ticker}',
                'xaxis': {'title': 'Date'},
                'yaxis': {'title': 'Price'}
            }
        }
    
    df.sort_values('Date', inplace=True)

    # --- TradingView Style Implementation ---

    # 1. Define TradingView colors
    INCREASING_COLOR = '#26a69a'
    DECREASING_COLOR = '#ef5350'
    GRID_COLOR = '#EAEAEA'

    # 2. Determine number of rows needed for indicators
    indicator_rows = [ind for ind in ['rsi', 'macd', 'adx', 'roc', 'ewo'] if ind in selected_indicators]
    num_rows = 2 + len(indicator_rows)
    row_heights = [0.7] + [0.15] * (len(indicator_rows) + 1) # Main chart, volume, then indicators

    specs = [[{"secondary_y": False}], [{"secondary_y": False}]] # Price and Volume panes
    for _ in indicator_rows:
        specs.append([{"secondary_y": False}]) # Indicator panes

    fig = make_subplots(rows=num_rows, cols=1, shared_xaxes=True,
                          vertical_spacing=0.02,
                          row_heights=row_heights,
                          specs=specs)

    # 3. Add Candlestick Trace (Price Pane - Row 1)
    fig.add_trace(go.Candlestick(x=df['Date'],
                                   open=df['Open'],
                                   high=df['High'],
                                   low=df['Low'],
                                   close=df['Close'],
                                   name='Price',
                                   increasing_fillcolor=INCREASING_COLOR,
                                   increasing_line_color=INCREASING_COLOR,
                                   decreasing_fillcolor=DECREASING_COLOR,
                                   decreasing_line_color=DECREASING_COLOR),
                  row=1, col=1)

    # Add price-based indicators to Price Pane
    if 'bb' in selected_indicators and all(c in df.columns for c in ['bb_upper', 'bb_lower', 'bb_middle']):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['bb_upper'], mode='lines', name='BB Upper', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['bb_lower'], mode='lines', name='BB Lower', line=dict(color='gray', width=1, dash='dash')), row=1, col=1)
    if 'ichimoku' in selected_indicators and all(c in df.columns for c in ['ichimoku_senkou_a', 'ichimoku_senkou_b']):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['ichimoku_senkou_a'], mode='lines', name='Ichimoku A', line=dict(color='rgba(0, 255, 0, 0.2)')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['ichimoku_senkou_b'], mode='lines', name='Ichimoku B', line=dict(color='rgba(255, 0, 0, 0.2)'), fill='tonexty'), row=1, col=1)
    if 'psar' in selected_indicators and 'psar' in df.columns:
        fig.add_trace(go.Scatter(x=df['Date'], y=df['psar'], mode='markers', name='Parabolic SAR', marker=dict(color='purple', size=4)), row=1, col=1)
    if 'donchian' in selected_indicators and all(c in df.columns for c in ['donchian_upper', 'donchian_lower']):
        fig.add_trace(go.Scatter(x=df['Date'], y=df['donchian_upper'], mode='lines', name='Donchian Upper', line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)
        fig.add_trace(go.Scatter(x=df['Date'], y=df['donchian_lower'], mode='lines', name='Donchian Lower', line=dict(color='cyan', width=1, dash='dash')), row=1, col=1)

    # 4. Add Volume Bar Trace (Volume Pane - Row 2)
    volume_colors = [INCREASING_COLOR if row['Close'] >= row['Open'] else DECREASING_COLOR for index, row in df.iterrows()]
    fig.add_trace(go.Bar(x=df['Date'], y=df['Volume'],
                         marker_color=volume_colors,
                         name='Volume'),
                  row=2, col=1)

    # 5. Add Indicator Traces (Rows 3+)
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

    # 6. Update the overall layout to match TradingView
    fig.update_layout(
        height=800,
        xaxis_rangeslider_visible=False,
        showlegend=False,
        plot_bgcolor='white',
        paper_bgcolor='white',
        margin=dict(l=10, r=50, t=10, b=10)
    )

    # 7. Update Y-Axes styles
    fig.update_yaxes(
        side='right',
        tickfont=dict(size=12, color='#333'),
        gridcolor=GRID_COLOR,
        row=1, col=1
    )
    fig.update_yaxes(
        showticklabels=False,  # Hide volume axis labels
        gridcolor=GRID_COLOR,
        row=2, col=1
    )

    # 8. Update X-Axis style for all panes
    fig.update_xaxes(
        gridcolor=GRID_COLOR,
        tickfont=dict(size=12, color='#787878'),
        showticklabels=True  # Ensure x-axis labels are visible on the bottom pane
    )
    
    # Hide x-axis labels on all but the bottom chart
    for i in range(1, num_rows):
        fig.update_xaxes(showticklabels=False, row=i, col=1)

    # Add annotations for signals
    annotations = []
    for signal in signals:
        if signal['ticker'] == selected_ticker:
            signal_date = pd.to_datetime(signal['Date'])
            signal_df = df[df['Date'] == signal_date]
            if not signal_df.empty:
                if signal['signal'] == 'buy':
                    annotations.append(dict(x=signal_date, 
                                            y=signal_df.iloc[0]['Low'], 
                                            text="B", showarrow=True, arrowhead=2, 
                                            ax=0, ay=20, bgcolor="#26a69a"))
                elif signal['signal'] == 'sell':
                    annotations.append(dict(x=signal_date, 
                                            y=signal_df.iloc[0]['High'], 
                                            text="S", showarrow=True, arrowhead=2, 
                                            ax=0, ay=-20, bgcolor="#ef5350"))
    fig.update_layout(annotations=annotations)

    return fig


@app.callback(
    Output('selected-point-info', 'children'),
    Input('stock-graph', 'clickData')
)
def display_click_data(clickData):
    if clickData is None:
        return "Click on a point in the graph to select it."
    else:
        point = clickData['points'][0]
        date_str = point['x']
        return f"Selected Date: {date_str.split(' ')[0]}"


### FIX: This is the main refactored callback that handles all signal modifications.
@app.callback(
    Output('signals-storage', 'data'),
    Output('profitable-trades-storage', 'data'),
    Output('save-status', 'children'),
    Output('edit-warning', 'children'),
    Input('buy-button', 'n_clicks'),
    Input('sell-button', 'n_clicks'),
    Input('remove-last-button', 'n_clicks'),
    Input('save-button', 'n_clicks'),
    Input('signals-table', 'data_previous'), # This now correctly tracks deletions
    State('signals-table', 'data'),
    State('stock-graph', 'clickData'),
    State('ticker-dropdown', 'value'),
    State('signals-storage', 'data')
)
def update_signals_and_trades(buy_clicks, sell_clicks, remove_clicks, save_clicks,
                             table_data_previous, current_table_data, clickData, 
                             selected_ticker, existing_signals):
    
    changed_id = [p['prop_id'] for p in callback_context.triggered][0]
    
    # Always clear messages on a new action
    save_status = ""
    edit_warning = ""

    # Convert stored signals to DataFrame for easier manipulation
    signals_df = pd.DataFrame(existing_signals) if existing_signals else pd.DataFrame(columns=['Date', 'ticker', 'Close', 'signal'])
    
    # --- Handle Deletions ---
    if 'data_previous' in changed_id:
        if table_data_previous is not None and len(current_table_data) < len(table_data_previous):
            # Find the deleted row by comparing the table data
            # Convert current table data to a set of tuples for easier comparison
            current_rows = set()
            for row in current_table_data:
                current_rows.add((row['Date'], row['ticker'], row['signal']))
            
            # Find which row was deleted
            deleted_row = None
            for row in table_data_previous:
                row_tuple = (row['Date'], row['ticker'], row['signal'])
                if row_tuple not in current_rows:
                    deleted_row = row
                    break
            
            if deleted_row:
                # Find the corresponding signal in the storage data
                # Convert date to match storage format
                deleted_date = pd.to_datetime(deleted_row['Date']).strftime('%Y-%m-%d %H:%M:%S')
                
                # Check if this deletion is allowed
                ticker_signals = sorted([s for s in existing_signals if s['ticker'] == deleted_row['ticker']], key=lambda x: x['Date'])
                
                # Create a signal object that matches the storage format for validation
                storage_signal = None
                for s in existing_signals:
                    signal_date = pd.to_datetime(s['Date']).strftime('%Y-%m-%d')
                    table_date = deleted_row['Date']
                    if (signal_date == table_date and 
                        s['ticker'] == deleted_row['ticker'] and 
                        s['signal'] == deleted_row['signal']):
                        storage_signal = s
                        break
                
                if storage_signal:
                    can_be_removed, warning = can_remove_signal(ticker_signals, storage_signal)

                    if not can_be_removed:
                        edit_warning = warning
                        # Since we can't revert the table, we just show a warning and don't update storage
                        return existing_signals, dash.no_update, save_status, edit_warning
                    else:
                        # Remove the signal from our main DataFrame
                        signals_df = signals_df[~(
                            (pd.to_datetime(signals_df['Date']).dt.strftime('%Y-%m-%d') == deleted_row['Date']) &
                            (signals_df['ticker'] == deleted_row['ticker']) &
                            (signals_df['signal'] == deleted_row['signal'])
                        )]
        
    # --- Handle Remove Last Signal Button ---
    elif 'remove-last-button' in changed_id:
        ticker_signals = signals_df[signals_df['ticker'] == selected_ticker].sort_values('Date').to_dict('records')
        if ticker_signals:
            last_signal = ticker_signals[-1]
            signals_df = signals_df[~(
                (signals_df['Date'] == last_signal['Date']) &
                (signals_df['ticker'] == last_signal['ticker']) &
                (signals_df['signal'] == last_signal['signal'])
            )]
    
    # --- Handle Add Buy/Sell Signal ---
    elif 'buy-button' in changed_id or 'sell-button' in changed_id:
        if not clickData:
            return dash.no_update, dash.no_update, save_status, "Please click on the graph to select a date first."
        
        date = clickData['points'][0]['x']
        
        # Check if signal for this date already exists
        if not signals_df[(signals_df['Date'] == date) & (signals_df['ticker'] == selected_ticker)].empty:
            return dash.no_update, dash.no_update, save_status, "Signal already exists for this date."
        
        signal_type = 'buy' if 'buy-button' in changed_id else 'sell'

        # Use comprehensive validation function
        is_valid, validation_error = validate_signal_placement(signal_type, date, selected_ticker, signals_df)
        
        if not is_valid:
            edit_warning = validation_error
        else:
                # Add the new signal
                try:
                    # Get data from local files
                    signal_data_row = get_ticker_data(selected_ticker, 'stock_data')
                    if signal_data_row.empty:
                        # Fallback to pre-loaded data
                        signal_data_row = stock_data[(stock_data['ticker'] == selected_ticker) & (stock_data['Date'] == date)]
                    else:
                        signal_data_row = signal_data_row[signal_data_row['Date'] == date]
                    
                    if not signal_data_row.empty:
                        new_signal = {
                            **signal_data_row.to_dict('records')[0],
                            'signal': signal_type
                        }
                        signals_df = pd.concat([signals_df, pd.DataFrame([new_signal])], ignore_index=True)
                    else:
                        edit_warning = "No data found for the selected date."
                except Exception as e:
                    edit_warning = f"Error adding signal: {str(e)}"

    # --- Handle Save Button ---
    elif 'save-button' in changed_id:
        if not signals_df.empty:
            signals_df.to_csv('signals.csv', index=False)
            profitable_trades_df = calculate_profitable_trades(signals_df, benchmark_data)
            if not profitable_trades_df.empty:
                profitable_trades_df.to_csv('profitable_trades.csv', index=False)
            save_status = "Signals and profitable trades saved successfully!"
        else:
            save_status = "No signals to save."

    # --- Recalculate and Update Everything ---
    # This runs after any valid modification (add, delete)
    profitable_trades_df = calculate_profitable_trades(signals_df, benchmark_data)
    
    return (signals_df.to_dict('records'), 
            profitable_trades_df.to_dict('records'), 
            save_status, 
            edit_warning)


# --- Callbacks for Tables and Downloads ---

@app.callback(
    Output('signals-table', 'data'),
    Output('signals-table', 'style_data_conditional'),
    Input('signals-storage', 'data')
)
def update_signals_table(signals):
    if not signals:
        return [], []

    df = pd.DataFrame(signals)
    df['Date'] = pd.to_datetime(df['Date']).dt.strftime('%Y-%m-%d')
    table_data = df[['Date', 'ticker', 'Close', 'signal']].to_dict('records')

    styles = [
        {'if': {'filter_query': '{signal} = "buy"'}, 'color': '#28a745', 'fontWeight': 'bold'},
        {'if': {'filter_query': '{signal} = "sell"'}, 'color': '#dc3545', 'fontWeight': 'bold'}
    ]
    return table_data, styles


@app.callback(
    Output('profitable-trades-table', 'data'),
    Input('profitable-trades-storage', 'data')
)
def update_profitable_trades_table(trades):
    return trades


@app.callback(
    Output("download-signals-csv", "data"),
    Input("btn_download_signals_csv", "n_clicks"),
    State("signals-storage", "data"),
    prevent_initial_call=True,
)
def download_signals(n_clicks, signals):
    df = pd.DataFrame(signals)
    return dcc.send_data_frame(df.to_csv, "signals_export.csv", index=False)


@app.callback(
    Output("download-profitable-csv", "data"),
    Input("btn_download_profitable_csv", "n_clicks"),
    State("profitable-trades-storage", "data"),
    prevent_initial_call=True,
)
def download_profitable_trades(n_clicks, trades):
    df = pd.DataFrame(trades)
    return dcc.send_data_frame(df.to_csv, "profitable_trades_export.csv", index=False)


if __name__ == '__main__':
    app.run(debug=True)