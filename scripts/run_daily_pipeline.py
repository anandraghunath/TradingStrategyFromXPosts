import sys
import os
import pytz
from datetime import datetime, timedelta
import pandas as pd

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.ingestion.parser import TweetParser
from src.engine.black_scholes import BlackScholesModel
from src.engine.indicators import MarketDataFetcher
from src.data.storage import TradeStorage

def main():
    # Mock examples from the user prompt
    mock_tweets = [
        {"text": "$AMZN 205C 2/18 .79 entry\n\nlotto for me because im on vacation", "created_at": "2024-02-14T10:05:00Z"},
        {"text": "$AAPL 262.5C 2/18 .80 entry\n\nstop loss 258\ntarget 260 and 261 then 264\n\ni think try it again at 253", "created_at": "2024-02-14T10:15:00Z"},
        {"text": "$SHOP 138C 2/13", "created_at": "2024-02-12T14:30:00Z"},
        {"text": "$MSFT 410C 2/9 1.55 entry\n\nI want a breakout at 408 to hold to 414", "created_at": "2024-02-08T11:00:00Z"}
    ]

    parser = TweetParser()
    bsm = BlackScholesModel()
    market_fetcher = MarketDataFetcher()

    # We assume a 2-minute delay between the trade and the tweet.
    TRADE_DELAY_MINUTES = 2

    # yfinance only gives 1m data for the last 7 days.
    # To test locally with real market data, we must mock the trade time to a recent business day.
    now = datetime.now(pytz.timezone('America/New_York'))
    days_back = 1
    # ensure it's a weekday
    while (now - timedelta(days=days_back)).weekday() > 4:
        days_back += 1
        
    recent_trading_day = now - timedelta(days=days_back)
    
    # We'll set the mocked tweet time to 11:00 AM on that recent trading day.
    target_market_time = recent_trading_day.replace(hour=11, minute=0, second=0, microsecond=0)

    print(f"Running retroactive pipeline simulation...")
    results = []

    for tweet in mock_tweets:
        print(f"\n{'-'*40}")
        print(f"Raw Tweet: {tweet['text']}")
        
        parsed = parser.parse(tweet["text"])
        if not parsed:
            print("Status: Could not parse options details. Skipping.")
            continue

        print(f"Parsed Signal: {parsed}")
        
        # Calculate real entry time based on assumed delay
        tweet_time = target_market_time
        entry_time = tweet_time - timedelta(minutes=TRADE_DELAY_MINUTES)
        
        print(f"Simulated Tweet Time: {tweet_time}")
        print(f"Estimated Entry Time: {entry_time}")

        ticker = parsed["ticker"]
        # Fetch wide set of indicators and underlying price at the exact entry time
        features = market_fetcher.fetch_data_and_compute_indicators(ticker, pd.Timestamp(entry_time))
        
        if not features:
            print(f"Error: Could not fetch underlying market data for {ticker}. Skipping.")
            continue
            
        underlying_price = features['underlying_price']
        print(f"Market Snapshot @ Entry - Underlying: ${underlying_price:.2f} | RSI(14): {features.get('RSI_14')} | VWAP Dist: {features.get('dist_to_VWAP_pct')}%")

        # Compute Options Greeks / IV
        # Time to Expiry (T) in years. We mock as 5 days for the sake of the demonstration.
        T = 5.0 / 365.0
        
        iv, delta, gamma = None, None, None

        if parsed["entry_price"] is not None:
            iv = bsm.implied_volatility(
                S=underlying_price, 
                K=parsed["strike"], 
                T=T, 
                market_price=parsed["entry_price"], 
                option_type=parsed["option_type"]
            )
            delta = bsm.delta(
                S=underlying_price, 
                K=parsed["strike"], 
                T=T, 
                sigma=iv, 
                option_type=parsed["option_type"]
            )
            gamma = bsm.gamma(
                S=underlying_price, 
                K=parsed["strike"], 
                T=T, 
                sigma=iv
            )
            print(f"Options Pricing @ Entry - IV: {iv*100:.2f}% | Delta: {delta:.4f} | Gamma: {gamma:.4f}")
        else:
            print("Status: No entry price available to calculate Greeks (needs pricing or preset IV).")

        # Combine parsed signals, options greeks, and technical features into one flat dictionary
        trade_record = {
            "timestamp": entry_time.isoformat(),
            "ticker": ticker,
            "strike": parsed["strike"],
            "option_type": parsed["option_type"],
            "entry_price": parsed["entry_price"],
            "iv": iv,
            "delta": delta,
            "gamma": gamma
        }
        # Merge technical features directly into the record
        trade_record.update(features)
        results.append(trade_record)

    print(f"\n{'-'*40}")
    print(f"Pipeline finished processing {len(results)} structured signals.")

    # Save to storage for ML processing later
    if results:
        storage = TradeStorage()
        storage.save_trades(results)

if __name__ == "__main__":
    main()
