import yfinance as yf
from pymongo import MongoClient
from pydantic import BaseModel
import numpy as np
from datetime import datetime, timedelta
from models.stock_price import StockPrice

client = MongoClient("mongodb+srv://dharmikparmarpd:dhp12345@cluster0.v5pxg.mongodb.net/stock_market?retryWrites=true&w=majority&appName=Cluster0")
db = client["stock_market"]
collection = db["stock_price"]


# Function to fetch stock data using yfinance
def fetch_stock_data(symbol: str, start_date: str, end_date: str):
    stock = yf.Ticker(symbol)
    bars = stock.history(start=start_date, end=end_date, interval="1d")
    return bars


# List of stock symbols to process
symbols = ["AAPL", "META", "MSFT", "AMZN", "TSLA"]

# Process each stock
for symbol in symbols:
    print(f"Processing {symbol}...")
    
    # Define date range
    end_date = datetime.today().strftime("%Y-%m-%d")
    start_date = (datetime.today() - timedelta(days=5*365)).strftime("%Y-%m-%d")

    # Fetch data
    stock_data = fetch_stock_data(symbol, start_date, end_date)

    # Extract and process stock data
    for index, row in stock_data.iterrows():
        stock_id = symbol  # Using the symbol as stock_id for simplicity
        date = row.name.date()
        open_price = row["Open"]
        high_price = row["High"]
        low_price = row["Low"]
        close_price = row["Close"]
        volume = row["Volume"]

        # Calculate technical indicators (SMA, RSI) using the Close prices
        # sma_20, sma_50, rsi_14 = calculate_indicators(np.array(stock_data["Close"], dtype=np.float64))

        # Prepare data for MongoDB, only including the indicators if they are not None
        stock_data_dict = {
            "stock_id": stock_id,
            "date": str(date),
            "open": open_price,
            "high": high_price,
            "low": low_price,
            "close": close_price,
            "volume": volume
        }

        # Only add indicators if they are calculated (i.e., not None)
        # if sma_20 is not None:
        #     stock_data_dict["sma_20"] = sma_20
        # if sma_50 is not None:
        #     stock_data_dict["sma_50"] = sma_50
        # if rsi_14 is not None:
        #     stock_data_dict["rsi_14"] = rsi_14

        # Insert into MongoDB
        collection.update_one(
            {"stock_id": stock_id, "date": str(date)},
            {"$set": stock_data_dict},
            upsert=True
        )

    print(f"✅ Successfully processed data for {symbol}")

print("✅ All stock data successfully processed and stored in MongoDB.")
