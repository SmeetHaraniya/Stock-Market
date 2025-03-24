import yfinance as yf
from pymongo import MongoClient
from pydantic import BaseModel
from typing import List
from models.stock import Stock

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
