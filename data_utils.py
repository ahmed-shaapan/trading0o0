import pandas as pd
import os
import warnings

warnings.filterwarnings('ignore')

def load_stock_data_from_files(data_dir='stock_data'):
    """
    Loads all stock data from local CSV files,
    filters it for the required date range, and excludes the 'QQQ' benchmark ticker.

    Args:
        data_dir (str): Directory containing the CSV files.

    Returns:
        pd.DataFrame: A DataFrame containing the combined and filtered stock data.
    """
    try:
        all_data = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv') and not filename.startswith('QQQ'):
                ticker = filename.split('_')[0]
                filepath = os.path.join(data_dir, filename)
                df = pd.read_csv(filepath)
                df['ticker'] = ticker
                
                # Handle datetime conversion more robustly
                try:
                    df['Date'] = pd.to_datetime(df['Date'], utc=True)
                    # Convert to timezone-naive datetime
                    df['Date'] = df['Date'].dt.tz_localize(None)
                except:
                    # Fallback to simple datetime conversion
                    df['Date'] = pd.to_datetime(df['Date'])
                
                # Filter for years 2020-2025
                df = df[(df['Date'].dt.year >= 2020) & (df['Date'].dt.year <= 2025)]
                all_data.append(df)
        
        if all_data:
            combined_df = pd.concat(all_data, ignore_index=True)
            return combined_df.sort_values(['ticker', 'Date'])
        else:
            return pd.DataFrame()
            
    except Exception as e:
        print(f"Error loading stock data from files: {e}")
        return pd.DataFrame()

def load_benchmark_data_from_files(data_dir='stock_data', benchmark_ticker='QQQ'):
    """
    Loads the benchmark data from local CSV files.

    Args:
        data_dir (str): Directory containing the CSV files.
        benchmark_ticker (str): The ticker symbol for the benchmark.

    Returns:
        pd.DataFrame: A DataFrame containing the benchmark data.
    """
    try:
        for filename in os.listdir(data_dir):
            if filename.startswith(benchmark_ticker) and filename.endswith('.csv'):
                filepath = os.path.join(data_dir, filename)
                df = pd.read_csv(filepath)
                
                # Handle datetime conversion more robustly
                try:
                    df['Date'] = pd.to_datetime(df['Date'], utc=True)
                    # Convert to timezone-naive datetime
                    df['Date'] = df['Date'].dt.tz_localize(None)
                except:
                    # Fallback to simple datetime conversion
                    df['Date'] = pd.to_datetime(df['Date'])
                
                # Filter for years 2020-2025
                df = df[(df['Date'].dt.year >= 2020) & (df['Date'].dt.year <= 2025)]
                return df.sort_values('Date')
        
        print(f"No file found for benchmark ticker {benchmark_ticker}")
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error loading benchmark data from files: {e}")
        return pd.DataFrame()

def get_available_tickers(data_dir='stock_data'):
    """Get list of available tickers from local CSV files."""
    try:
        tickers = []
        for filename in os.listdir(data_dir):
            if filename.endswith('.csv') and not filename.startswith('QQQ'):
                ticker = filename.split('_')[0]
                tickers.append(ticker)
        
        return sorted(list(set(tickers)))  # Remove duplicates and sort
        
    except Exception as e:
        print(f"Error getting available tickers: {e}")
        return []

def get_ticker_data(ticker, data_dir='stock_data'):
    """Get data for a specific ticker from local CSV files."""
    try:
        for filename in os.listdir(data_dir):
            if filename.startswith(ticker) and filename.endswith('.csv'):
                filepath = os.path.join(data_dir, filename)
                df = pd.read_csv(filepath)
                df['ticker'] = ticker
                
                # Handle datetime conversion more robustly
                try:
                    df['Date'] = pd.to_datetime(df['Date'], utc=True)
                    # Convert to timezone-naive datetime
                    df['Date'] = df['Date'].dt.tz_localize(None)
                except:
                    # Fallback to simple datetime conversion
                    df['Date'] = pd.to_datetime(df['Date'])
                
                # Filter for years 2020-2025
                df = df[(df['Date'].dt.year >= 2020) & (df['Date'].dt.year <= 2025)]
                return df.sort_values('Date')
        
        print(f"No file found for ticker {ticker}")
        return pd.DataFrame()
        
    except Exception as e:
        print(f"Error getting ticker data for {ticker}: {e}")
        return pd.DataFrame()

# Main wrapper functions for the app
def load_stock_data(data_dir='stock_data'):
    """Load stock data from local CSV files."""
    return load_stock_data_from_files(data_dir)

def load_benchmark_data(data_dir='stock_data', benchmark_ticker='QQQ'):
    """Load benchmark data from local CSV files."""
    return load_benchmark_data_from_files(data_dir, benchmark_ticker)
