from fastapi.templating import Jinja2Templates
import pandas as pd
from fastapi import APIRouter, Request
from models.stock import Stock
from pymongo import MongoClient
import os
import joblib

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

@stock_router.get("/predict")
def predict(request: Request):
    return templates.TemplateResponse("predict.html",{
        "request": request
    })


@stock_router.post("/predict")
def predict(data: StockData):
    model_path = os.path.join(os.path.dirname(__file__), "..", "ml_model", "stock_prediction_model.pkl")
    model = joblib.load(model_path)
    # Convert input into DataFrame format for model
    df = pd.DataFrame([data.dict()])
    
    # Make prediction
    prediction = model.predict(df)[0]

    return {"prediction": "Up" if prediction == 1 else "Down"}

