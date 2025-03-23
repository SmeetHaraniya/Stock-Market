from fastapi.templating import Jinja2Templates
import pandas as pd
from fastapi import APIRouter, Request
from models.stock import Stock
from pymongo import MongoClient
import os
import joblib
import yfinance as yf

from models.stock_data import StockData

stock_router = APIRouter()
client = MongoClient("mongodb+srv://dharmikparmarpd:dhp12345@cluster0.v5pxg.mongodb.net/stock_market?retryWrites=true&w=majority&appName=Cluster0")

db = client["stock_market"]

templates = Jinja2Templates(directory="templates")

@stock_router.post("/stocks/", response_model=Stock)
def create_stock(stock: Stock):
    result = db.stock.insert_one(stock.dict())
    return {"id": str(result.inserted_id), **stock.dict()}

@stock_router.get("/stocks/")
def get_stocks():
    stocks = list(db.stock.find())
    for stock in stocks:
        stock["id"] = str(stock["_id"])
        del stock["_id"]
    return stocks

@stock_router.get("/predict/{symbol}")
def predict(request: Request, symbol: str):
    data = get_last_date_data(symbol)  # Get formatted last day data

    print("last-date-data", data)  # Debugging output
    
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "data": data,
        "symbol": symbol  # Ensure the variable name matches the template
    })

def get_last_date_data(symbol):
    try:
        # Fetch the last 1 day's data
        ticker = yf.download(symbol, period="1d", interval="1d").iloc[-1]

        # # Extract the last available date
        last_date = ticker.name.strftime('%Y-%m-%d')

        # Convert to a standard dictionary
        last_day_data = {
            "Date": last_date,
            "Open": round(ticker["Open"][symbol], 2),
            "High": round(ticker["High"][symbol], 2),
            "Low": round(ticker["Low"][symbol], 2),
            "Close": round(ticker["Close"][symbol], 2),
            "Volume": int(ticker["Volume"][symbol])
        }
        return last_day_data
    
    except Exception as e:
        print(f"Error fetching data: {e}")
        return {"error": str(e)}



@stock_router.post("/predict")
def predict(data: StockData):
    model_path = os.path.join(os.path.dirname(__file__), "..", "ml_model", "stock_prediction_model.pkl")
    model = joblib.load(model_path)
    # Convert input into DataFrame format for model
    df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(df)[0]

    return {"prediction": "Up" if prediction == 0 else "Down"}

