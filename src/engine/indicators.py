import yfinance as yf
import pandas as pd
import ta
import numpy as np

class MarketDataFetcher:
    def __init__(self):
        pass

    def fetch_data_and_compute_indicators(self, ticker: str, target_time: pd.Timestamp, lookback_days: int = 5):
        """
        Fetches 1m data for the ticker and computes a wide array of technical indicators
        at the target_time to create a rich feature set for ML modeling.
        """
        # Increase lookback to ensure enough data for indicators like 200 EMA, though with 1m data we might only get a few days.
        start_date = target_time - pd.Timedelta(days=lookback_days)
        end_date = target_time + pd.Timedelta(days=1)

        try:
            stock = yf.Ticker(ticker)
            df = stock.history(start=start_date.strftime('%Y-%m-%d'), end=end_date.strftime('%Y-%m-%d'), interval="1m")
            if df.empty:
                return None

            # Localize index safely
            if df.index.tz is None:
                df.index = df.index.tz_localize('America/New_York')
            else:
                df.index = df.index.tz_convert('America/New_York')

            # --- Technical Indicators ---
            
            # Momentum: RSI, MACD, Stochastic
            df['RSI_14'] = ta.momentum.RSIIndicator(close=df['Close'], window=14).rsi()
            macd = ta.trend.MACD(close=df['Close'])
            df['MACD'] = macd.macd()
            df['MACD_Signal'] = macd.macd_signal()
            df['MACD_Hist'] = macd.macd_diff()
            df['Stoch_k'] = ta.momentum.StochasticOscillator(high=df['High'], low=df['Low'], close=df['Close']).stoch()

            # Trend: SMAs and EMAs
            df['SMA_9'] = ta.trend.SMAIndicator(close=df['Close'], window=9).sma_indicator()
            df['SMA_20'] = ta.trend.SMAIndicator(close=df['Close'], window=20).sma_indicator()
            df['SMA_50'] = ta.trend.SMAIndicator(close=df['Close'], window=50).sma_indicator()
            df['EMA_9'] = ta.trend.EMAIndicator(close=df['Close'], window=9).ema_indicator()
            df['EMA_21'] = ta.trend.EMAIndicator(close=df['Close'], window=21).ema_indicator()

            # Volatility: Bollinger Bands, ATR
            bb = ta.volatility.BollingerBands(close=df['Close'], window=20, window_dev=2)
            df['BB_High'] = bb.bollinger_hband()
            df['BB_Low'] = bb.bollinger_lband()
            df['BB_Mid'] = bb.bollinger_mavg()
            df['BB_Width'] = bb.bollinger_wband()
            df['ATR_14'] = ta.volatility.AverageTrueRange(high=df['High'], low=df['Low'], close=df['Close'], window=14).average_true_range()

            # Volume: VWAP & OBV
            df['OBV'] = ta.volume.OnBalanceVolumeIndicator(close=df['Close'], volume=df['Volume']).on_balance_volume()
            
            df['Date'] = df.index.date
            df['Typical_Price'] = (df['High'] + df['Low'] + df['Close']) / 3
            df['Volume_x_Typical_Price'] = df['Volume'] * df['Typical_Price']
            df['Cum_Vol'] = df.groupby('Date')['Volume'].cumsum()
            df['Cum_Vol_Price'] = df.groupby('Date')['Volume_x_Typical_Price'].cumsum()
            df['VWAP'] = df['Cum_Vol_Price'] / df['Cum_Vol']

            # Locate nearest point to target_time
            if target_time.tz is None:
                target_time = target_time.tz_localize('America/New_York')
            else:
                target_time = target_time.tz_convert('America/New_York')

            past_data = df[df.index <= target_time]
            if past_data.empty:
                return None

            latest_row = past_data.iloc[-1]
            current_date_data = past_data[past_data['Date'] == latest_row['Date']]
            
            # Intraday Price Extremes (Breakout detection)
            high_of_day = current_date_data['High'].max()
            low_of_day = current_date_data['Low'].min()
            
            close_price = float(latest_row['Close'])

            features = {
                'underlying_price': close_price,
                'RSI_14': float(latest_row['RSI_14']) if not pd.isna(latest_row['RSI_14']) else None,
                'MACD': float(latest_row['MACD']) if not pd.isna(latest_row['MACD']) else None,
                'MACD_Hist': float(latest_row['MACD_Hist']) if not pd.isna(latest_row['MACD_Hist']) else None,
                'Stoch_k': float(latest_row['Stoch_k']) if not pd.isna(latest_row['Stoch_k']) else None,
                'SMA_9': float(latest_row['SMA_9']) if not pd.isna(latest_row['SMA_9']) else None,
                'SMA_20': float(latest_row['SMA_20']) if not pd.isna(latest_row['SMA_20']) else None,
                'EMA_9': float(latest_row['EMA_9']) if not pd.isna(latest_row['EMA_9']) else None,
                'EMA_21': float(latest_row['EMA_21']) if not pd.isna(latest_row['EMA_21']) else None,
                'BB_High': float(latest_row['BB_High']) if not pd.isna(latest_row['BB_High']) else None,
                'BB_Low': float(latest_row['BB_Low']) if not pd.isna(latest_row['BB_Low']) else None,
                'ATR_14': float(latest_row['ATR_14']) if not pd.isna(latest_row['ATR_14']) else None,
                'OBV': float(latest_row['OBV']) if not pd.isna(latest_row['OBV']) else None,
                'VWAP': float(latest_row['VWAP']) if not pd.isna(latest_row['VWAP']) else None,
                'HOD': float(high_of_day),
                'LOD': float(low_of_day),
                'dist_to_HOD_pct': (high_of_day - close_price) / close_price * 100 if close_price else None,
                'dist_to_LOD_pct': (close_price - low_of_day) / close_price * 100 if close_price else None,
                'dist_to_VWAP_pct': (close_price - latest_row['VWAP']) / latest_row['VWAP'] * 100 if latest_row['VWAP'] else None
            }

            return features

        except Exception as e:
            print(f"Error fetching data for {ticker}: {e}")
            return None
