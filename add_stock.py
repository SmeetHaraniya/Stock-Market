# import yfinance as yf
# from pymongo import MongoClient
# from datetime import datetime, timedelta

# # MongoDB connection setup
# client = MongoClient("mongodb+srv://dharmikparmarpd:dhp12345@cluster0.v5pxg.mongodb.net/stock_market?retryWrites=true&w=majority&appName=Cluster0")
# db = client["stock_market"]
# collection = db["stock_price"]

# # List of stock ticker symbols
# stock_symbols = ["AAPL", "META", "MSFT", "AMZN"]

# # Calculate the date for 5 years ago from today
# end_date = datetime.today()
# start_date = end_date - timedelta(days=5*365)  # Roughly 5 years

# # Loop through each stock symbol to fetch data and store it in MongoDB
# for stock_symbol in stock_symbols:
#     print(f"Fetching data for {stock_symbol}...")
    
#     # Fetch historical stock data using yfinance
#     stock_data = yf.download(stock_symbol, start=start_date, end=end_date)
    
#     print(stock_data)
#     # # Check if data is empty
#     # if stock_data.empty:
#     #     print(f"No data found for {stock_symbol}.")
#     #     continue

#     # # Convert the data to a dictionary format suitable for MongoDB
#     # stock_data_dict = stock_data.reset_index().to_dict(orient="records")
    
#     # # Add a field for the stock symbol
#     # for record in stock_data_dict:
#     #     record["symbol"] = stock_symbol

#     # # Insert the data into MongoDB
#     # collection.insert_many(stock_data_dict)

#     # print(f"Data for {stock_symbol} has been successfully added to the database.")

# print("All data has been successfully added.")


import yfinance as yf
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List
from models.stock import Stock

# MongoDB connection setup
client = MongoClient("mongodb+srv://dharmikparmarpd:dhp12345@cluster0.v5pxg.mongodb.net/stock_market?retryWrites=true&w=majority&appName=Cluster0")
db = client["stock_market"]
collection = db["stock"]

# Function to fetch stock data using yfinance
def fetch_stock_data(symbol: str):
    stock = yf.Ticker(symbol)
    
    # Get stock details from Yahoo Finance
    stock_info = stock.info
    stock_name = stock_info.get('longName', 'Unknown')
    stock_exchange = stock_info.get('exchange', 'Unknown')
    shortable = stock_info.get('shortable', False)  # Some stocks may not have the shortable attribute

    # Return a dictionary with stock data
    return {
        "symbol": symbol,
        "name": stock_name,
        "exchange": stock_exchange,
        "shortable": shortable
    }

# Function to insert stock data into MongoDB
def insert_stock_data(stock_data: dict):
    # Convert the stock data to a Stock model for validation
    stock = Stock(**stock_data)

    # Insert into MongoDB
    collection.insert_one(stock.dict())  # Convert the Pydantic model to dictionary

    print(f"Stock {stock.symbol} data inserted into MongoDB.")

# List of stock symbols you want to fetch data for
symbols = ['AAPL', 'META', 'MSFT', 'AMZN', 'TSLA']

# Fetch and insert data for each stock symbol
for symbol in symbols:
    stock_data = fetch_stock_data(symbol)
    insert_stock_data(stock_data)

print("✅ Successfully added stock data to MongoDB.")
