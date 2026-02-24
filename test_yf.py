import yfinance as yf
print("Starting yfinance test...")
try:
    stock = yf.Ticker("AAPL")
    df = stock.history(period="1d", interval="1m")
    if not df.empty:
        print("Success! Data shape:", df.shape)
    else:
        print("Data is empty.")
except Exception as e:
    print("Error:", e)
print("Test finished.")
