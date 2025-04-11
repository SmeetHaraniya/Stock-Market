from datetime import datetime
from bson import ObjectId
from fastapi.templating import Jinja2Templates
import pandas as pd
from fastapi import APIRouter, Request
from models.stock import Stock
from pymongo import MongoClient
import os
import joblib
import yfinance as yf
import requests
from pydantic import BaseModel
import re
from transformers import AutoModelForSequenceClassification, AutoTokenizer, pipeline


stock_router = APIRouter()
client = MongoClient("mongodb+srv://dharmikparmarpd:dhp12345@cluster0.v5pxg.mongodb.net/stock_market?retryWrites=true&w=majority&appName=Cluster0")

db = client["stock_market"]
testing_data_collection = db["testing_data"]  # Collection for test data
cnbc_news_collection = db["cnbc_news"]  # Collection for CNBC data


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
    data = fetch_last_test_data()  # Get formatted last day data

    if isinstance(data, pd.DataFrame):
        filtered_data = data[data["Stock Name"] == symbol].to_dict(orient="records")
    else:
        filtered_data = []
    
    return templates.TemplateResponse("predict.html", {
        "request": request,
        "data": filtered_data[0]  # Ensure the variable name matches the template
    })


def get_last_date_data(symbol):
    try:
        # Fetch the last 1 day's data
        ticker = yf.download(symbol, start="2025-03-21",end="2025-03-22", interval="1d").iloc[-1]
        # ticker = yf.download(symbol, period="1d", interval="1d").iloc[-1]

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


def store_test_data():
    symbols = ['TSLA','MSFT','META','AMZN',"AAPL"]

    for symbol in symbols:
        stock_data = get_last_date_data(symbol)

        news = fetch_cnbc_news(stock_data["Date"],symbol)
        avg_sen_score = process_dataframe(news)
        [ma5, ma20, pl1, pl2] = get_moving_avg(symbol)

        feature_names = [
            "Open", "High", "Low", "Close", "Volume", "Tweet_Count",
            "Sentiment_Score", "Weighted_Sentiment", "MA5", "MA20",
            "Sentiment_Lag_1", "Sentiment_Lag_2", "Price_Lag_1", "Price_Lag_2"
        ]

        df = pd.DataFrame([[
            stock_data["Open"], stock_data["High"], stock_data["Low"], stock_data["Close"], stock_data["Volume"],
            len(news), avg_sen_score, avg_sen_score * len(news),
            ma5, ma20, 0.955, 0.955, pl1, pl2
        ]], columns=feature_names)

        features = {
            "Open": stock_data["Open"],
            "High": stock_data["High"],
            "Low": stock_data["Low"],
            "Close": stock_data["Close"],
            "Volume": stock_data["Volume"],
            "Tweet_Count": len(news),
            "Sentiment_Score": float(avg_sen_score.iloc[0]) if not avg_sen_score.empty else 0.0,  # ✅ Convert to float
            "Weighted_Sentiment": float(avg_sen_score.iloc[0] * len(news)) if not avg_sen_score.empty else 0.0,  # ✅ Convert to float
            "MA5": ma5,
            "MA20": ma20,
            "Sentiment_Lag_1": 0.955,
            "Sentiment_Lag_2": 0.955,
            "Price_Lag_1": pl1,
            "Price_Lag_2": pl2,
        }


        testing_data_record = {
            "_id": str(ObjectId()),
            "symbol": symbol,
            "date": stock_data["Date"],
            "features": features,
            "prediction": testing_prediction(symbol, stock_data["Date"], features),
            "created_at": datetime.utcnow()
        }
        testing_data_collection.insert_one(testing_data_record)


def fetch_last_test_data():
    symbols = ['TSLA', 'MSFT', 'META', 'AMZN', "AAPL"]
    all_data = []

    for symbol in symbols:
        # Fetch last 5 records for the symbol
        records = list(testing_data_collection.find({"symbol": symbol}).sort("date", -1).limit(1))

        for record in records:
            features = record["features"].copy()
            features["Stock Name"] = symbol
            features["Date"] = record["date"]
            features["prediction"] = record.get("prediction", None)
            all_data.append(features)

    # Convert to DataFrame
    df = pd.DataFrame(all_data)
    return df


def testing_prediction(symbol,date, features):
    model_path = os.path.join(os.path.dirname(__file__), "..", "ml_model", "stock_prediction_model.pkl")
    model = joblib.load(model_path)
    df = pd.DataFrame([features])

    prediction = model.predict(df)[0]
    return "Up" if prediction == 0 else "Down"


# @stock_router.post("/user_predict/{symbol}/{date}")
# def user_predict(request: Request, symbol: str, date: str):

#     record = testing_data_collection.find_one({"symbol": symbol, "date": date})

#     required_keys = {
#         "Tweet_Count": "Number of Tweets",
#         "Sentiment_Score": "Sentiment Score",
#         "Weighted_Sentiment": "Weighted Sentiment",
#         "MA5": "5-Day Moving Avg",
#         "MA20": "20-Day Moving Avg",
#         "Sentiment_Lag_1": "Previous Day Sentiment",
#         "Sentiment_Lag_2": "Two Days Ago Sentiment",
#         "Price_Lag_1": "Previous Day Price",
#         "Price_Lag_2": "Two Days Ago Price"
#     }
#     filtered_features = {new_key: record["features"][old_key] for old_key, new_key in required_keys.items() if old_key in record["features"]}

#     return {"prediction": record["prediction"], "parameters": filtered_features}
    

def fetch_cnbc_news(target_date, symbol):
    # API URL
    url = "https://cnbc.p.rapidapi.com/news/v2/list-by-symbol"

    # API Headers
    headers = {
        "x-rapidapi-key": "0685334e9bmsh739b5cdad882531p1f7519jsn45135e3125fd",
        "x-rapidapi-host": "cnbc.p.rapidapi.com"
    }

    # Initialize list to store all results
    all_results = []
    page = 1  # Start from page 1
    page_size = 30  # Number of articles per page

    while True:
        # Set query parameters for pagination
        querystring = {"page": str(page), "pageSize": str(page_size), "symbol": symbol}

        # Fetch data from API
        response = requests.get(url, headers=headers, params=querystring)
        response_json = response.json()

        # Extract data safely
        data = response_json.get('data', {})
        symbolEntries = data.get('symbolEntries', {})
        results = symbolEntries.get('results', [])

        # If no more results, stop pagination
        if not results:
            print(f"No more data on page {page}. Stopping pagination.")
            break

        # Add results to list
        all_results.extend(results)

        # Move to the next page
        page += 1

    # Convert list to DataFrame
    df = pd.DataFrame(all_results)

    if df.empty:
        print("\n⚠️ No news articles retrieved from API.")
        return pd.DataFrame(columns=['dateFirstPublished', 'description'])  # Return empty DataFrame

    # Convert 'dateFirstPublished' to datetime format and extract only the date (YYYY-MM-DD)
    df['dateFirstPublished'] = pd.to_datetime(df['dateFirstPublished']).dt.strftime('%Y-%m-%d')

    # 🔍 Filter news for the specific date
    filtered_news = df[df['dateFirstPublished'] == target_date]

    # Select relevant columns
    # sentiment_data = filtered_news[['headline', 'description', 'title', 'dateFirstPublished']]
    sentiment_data = filtered_news[['description','dateFirstPublished']]
    # Save data to CSV
    # csv_filename = f"cnbc_{symbol}_{target_date}.csv"
    # csv_path = save_path + csv_filename
    # sentiment_data.to_csv(csv_path, index=False)

    # Display summary
    if sentiment_data.empty:
        print(f"\n⚠️ No news articles found for {target_date}.")
    else:
        # print(f"\n✅ Successfully fetched {len(sentiment_data)} articles for {target_date} and saved to {csv_path}")
        print(sentiment_data)


    cnbc_news_record = {
        "_id": str(ObjectId()),
        "symbol": symbol,
        "date": target_date,
        "articles": sentiment_data.to_dict(orient="records"),
        "total_articles": len(sentiment_data),
        "created_at": datetime.utcnow()
    }
    cnbc_news_collection.insert_one(cnbc_news_record)

    return sentiment_data  # Return the DataFrame


def process_dataframe(df):
    text_column="description"

    model_name = "yiyanghkust/finbert-tone"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModelForSequenceClassification.from_pretrained(model_name)

    # Create a sentiment analysis pipeline
    nlp = pipeline("sentiment-analysis", model=model, tokenizer=tokenizer, batch_size=16)



    if text_column not in df.columns:
        raise ValueError(f"Column '{text_column}' not found in the DataFrame.")

    # Clean text
    df[text_column] = df[text_column].astype(str).apply(lambda text: re.sub(r'[^a-zA-Z\s]', '', text).lower())

    # Perform batch sentiment analysis
    sentiment_results = nlp(df[text_column].tolist())

    # Extract sentiment labels
    df["Sentiment"] = [result["label"] for result in sentiment_results]

    sentiment_mapping = {"Positive": 1, "Neutral": 0, "Negative": -1}
    df["Sentiment Score"] = df["Sentiment"].map(sentiment_mapping)

    # Aggregate tweets per date
    tweets_agg = df.groupby("dateFirstPublished").agg(
        Tweet_Count=("Sentiment Score", "count"),
        Avg_Tweet_Sentiment=("Sentiment Score", "mean")
    ).reset_index()

    return tweets_agg['Avg_Tweet_Sentiment']


def get_moving_avg(symbol):

    # ✅ Fetch Stock Data for the last 21 days (to calculate MA5 & MA20)

    data = yf.download(symbol, period="21d", interval="1d")

    # ✅ Compute Moving Averages
    data["MA5"] = data["Close"].rolling(window=5).mean()
    data["MA20"] = data["Close"].rolling(window=20).mean()
    data["pl1"] = data["Close"].shift(1)
    data["pl2"] = data["Close"].shift(2)

    # ✅ Get Only the Latest MA5 & MA20
    latest_ma5, latest_ma20, pl1, pl2 = data["MA5"].iloc[-1], data["MA20"].iloc[-1], data["pl1"].iloc[-1], data["pl2"].iloc[-1]

    return [latest_ma5, latest_ma20, pl1, pl2]


# class ModelData(BaseModel):
#     Close: float
#     High: float
#     Low: float
#     Open: float
#     Volume: int
   