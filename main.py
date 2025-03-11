from datetime import datetime

from bson import ObjectId
from pydantic import BaseModel
import config
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import JSONResponse, RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from pymongo import MongoClient
from fastapi.responses import HTMLResponse
from fastapi.templating import Jinja2Templates
from datetime import datetime, timedelta
import random

from models.stock import Stock
from models.user import User
from models.stock_price import StockPrice
from models.stock_holding import StockHolding
from models.transaction import Transaction

from routes.stock_routes import stock_router
from routes.user_routes import user_router
from routes.login_routes import login_router
from routes.signup_routes import signup_router

from fastapi.templating import Jinja2Templates
import requests

app = FastAPI()

# Connect to MongoDB
client = MongoClient("mongodb+srv://dharmikparmarpd:dhp12345@cluster0.v5pxg.mongodb.net/stock_market?retryWrites=true&w=majority&appName=Cluster0")
db = client['stock_market'] 

users_collection = db["user"] 
stock_collection = db['stock']
stock_price_collection = db['stock_price']
stock_holding_collection = db['stock_holding']
transaction_collection = db['transaction']

app.add_middleware(SessionMiddleware, secret_key=config.SPECIAL_KEY)


# Include Routers
app.include_router(stock_router)
app.include_router(user_router)
app.include_router(login_router)
app.include_router(signup_router)

templates = Jinja2Templates(directory="templates")

@app.get("/")
def root():
    return RedirectResponse(url="/login")

@app.get("/index")
def index(request: Request):
    print('Index:', request.session.get('user_id'))
    stock_filter = request.query_params.get('filter', False)

    rows = stock_collection.find({}, {"symbol": 1, "name": 1})
    print(rows)

    # ✅ Fetch latest closing prices for each stock
    latest_prices = stock_price_collection.aggregate([
        {"$sort": {"date": -1}},
        {"$group": {
            "_id": "$stock_id",
            "close": {"$first": "$close"}
        }}
    ])

    # ✅ Map closing prices to stock symbols
    closing_values = {}
    for price in latest_prices:
        stock = stock_collection.find_one({"symbol": price["_id"]})
        # print(stock)
        if stock:
            closing_values[stock['symbol']] = price['close']



    # # Apply filter logic for sorting based on closing price
    if stock_filter == 'new_closing_highs':
        # Sort by highest closing price (descending)
        closing_values = sorted(closing_values, key=lambda x: x['close'], reverse=True)

    elif stock_filter == 'new_closing_lows':
        # Sort by lowest closing price (ascending)
        closing_values = sorted(closing_values, key=lambda x: x['close'])


    # ✅ Return the template response
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stocks": rows,
        "closing_values": closing_values
    })

@app.get("/search_stocks")
def search_stocks(query: str = Query("")):
    # Use MongoDB regex to match stocks starting with 'query'
    stocks = stock_collection.find(
        {"symbol": {"$regex": f"^{query}", "$options": "i"}},  # Case-insensitive search
        {"_id": 0, "symbol": 1, "name": 1}  # Return only symbol and name
    ).sort("symbol", 1)

    # Convert MongoDB cursor to a list of dictionaries
    result = list(stocks)
    
    return JSONResponse(result)

@app.get("/stock/{symbol}")
def stock_details(request: Request, symbol: str):
    # Fetch stock details using the symbol
    stock = stock_collection.find_one({"symbol": symbol})

    if stock is None:
        return {"error": "Stock not found"}

    # Fetch stock price history for the given stock_id, sorted by latest date
    prices = list(stock_price_collection.find({"stock_id": stock["symbol"]}).sort("date", -1))

    for price in prices:
        if "_id" in price:
            price["_id"] = str(price["_id"])

    print(request.session)
    return templates.TemplateResponse("stock_details.html", {
        "request": request,
        "stock": stock,
        "bars": prices,
    })


@app.get("/trade_stocks")
def trade_stocks(request:Request):
    return templates.TemplateResponse("trade_stocks.html",{"request":request})
    
@app.get("/get-stock-price/{symbol}")
def get_stock_price(request: Request, symbol:str):
    # Find the stock ID based on the symbol
    stock = stock_collection.find_one({"symbol": symbol.upper()})
    print(stock)
    if not stock:
        raise HTTPException(status_code=404, detail="Stock symbol not found")
    
    # Get the latest stock price based on the stock ID
    stock_price = stock_price_collection.find(
        {"stock_id": stock["symbol"]}, 
        sort=[("date", -1)]  # Sort by date DESCENDING to get the latest price
    )

    latest_price = next(stock_price, None)
    print(latest_price)
    if not stock_price:
        raise HTTPException(status_code=404, detail="Stock price not available")
    

    # Return the latest stock price
    return {"price": round(latest_price['close'], 2) }

@app.post("/buy")
def buy(request: Request, symbol: str = Form(...), quantity: int = Form(...), price: float = Form(...)):
    user_id = request.session.get("user_id")
    print("post:",  request.session)
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    # user = user.to_list()[0]
    print("user",user)
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # request.body.user_id = user_id
    total_cost = quantity * price
    
    if user["cash"] < total_cost:
        raise HTTPException(status_code=400, detail="Insufficient funds")

    # Deduct cash
    users_collection.update_one({"_id": ObjectId(user_id)}, {"$inc": {"cash": -total_cost}})

    # Check existing stock holding
    existing_holding = stock_holding_collection.find_one({"user_id": user_id, "company_symbol": symbol})
    print(existing_holding)
    if existing_holding:
        new_quantity = existing_holding["number_of_shares"] + quantity
        new_avg_price = ((existing_holding["avg_price"] * existing_holding["number_of_shares"]) + total_cost) / new_quantity

        stock_holding_collection.update_one(
            {"_id": existing_holding["_id"]},
            {"$set": {"number_of_shares": new_quantity, "avg_price": new_avg_price}}
        )
    else:
        stock_holding_collection.insert_one({
            "user_id": user_id,
            "company_symbol": symbol,
            "number_of_shares": quantity,
            "avg_price": price
        })

    # Log transaction
    transaction_collection.insert_one({
        "user_id": user_id,
        "action": "BUY",
        "symbol": symbol,
        "quantity": quantity,
        "price": price,
        "timestamp": datetime.now(),
        "profit_loss": "-"
    })
    print(-1)
    return {"message": "Stock purchased successfully"}

@app.post("/sell")
def sell(request: Request, symbol: str = Form(...), quantity: int = Form(...), price: float = Form(...)):
    user_id = request.session.get("user_id")
    print("post:", request.session)
    
    # Fetch the user from the database
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Check for existing stock holdings
    existing_holding = stock_holding_collection.find_one({"user_id": user_id, "company_symbol": symbol})
    if not existing_holding:
        raise HTTPException(status_code=400, detail="You do not own any shares of this stock")

    # Ensure the user has enough shares to sell
    if existing_holding["number_of_shares"] < quantity:
        raise HTTPException(status_code=400, detail="Insufficient shares to sell")

    # Calculate the total revenue from the sale
    total_revenue = quantity * price

     # Calculate the profit or loss for the transaction
    avg_price = existing_holding["avg_price"]
    profit_loss = (price - avg_price) * quantity  # Profit if positive, loss if negative

    # Add the revenue to the user's cash balance
    users_collection.update_one({"_id": ObjectId(user_id)}, {"$inc": {"cash": total_revenue}})

    # Update the stock holding after selling
    new_quantity = existing_holding["number_of_shares"] - quantity

    if new_quantity > 0:
        # If the user still has shares left, update the holding
        stock_holding_collection.update_one(
            {"_id": existing_holding["_id"]},
            {"$set": {"number_of_shares": new_quantity}}
        )
    else:
        # If no shares are left, remove the holding
        stock_holding_collection.delete_one({"_id": existing_holding["_id"]})

    # Log the transaction
    transaction_collection.insert_one({
        "user_id": user_id,
        "action": "SELL",
        "symbol": symbol,
        "quantity": quantity,
        "price": price,
        "timestamp": datetime.now(),
        "profit_loss": str(profit_loss)
    })

    return {"message": "Stock sold successfully"}

# @app.get("/portfolio")
# def portfolio(request: Request):
#     user_id = request.session.get("user_id")
#     if not user_id:
#         raise HTTPException(status_code=401, detail="Not authenticated")

#     # Fetch user details
#     user = users_collection.find_one({"_id": ObjectId(user_id)})
#     if not user:
#         raise HTTPException(status_code=404, detail="User not found")

#     # Fetch stock holdings for the user
#     stock_holdings = stock_holding_collection.find({"user_id": user_id})
#     portfolio_stocks = {}
#     for holding in stock_holdings:
#         symbol = holding["company_symbol"]
#         portfolio_stocks[symbol] = {
#             "quantity": holding["number_of_shares"],
#             "avg_price": holding["avg_price"]
#         }

#     # Fetch transaction history for the user
#     transaction_history = transaction_collection.find({"user_id": user_id}).sort("timestamp", -1)

#     portfolio_data = {
#         "funds": user.get("cash", 0),  # User's available cash
#         "stocks": portfolio_stocks,
#         "history": [
#             {
#                 "action": transaction["action"],
#                 "stock": transaction["symbol"],
#                 "quantity": transaction["quantity"],
#                 "price": transaction["price"]
#             }
#             for transaction in transaction_history
#         ]
#     }

#     return templates.TemplateResponse("portfolio.html", {"request": request, "portfolio": portfolio_data})
    
@app.route("/transaction-history", methods=["GET"])
def stock_transaction_history(request: Request):
    user_id = request.session.get("user_id")
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")

    # Fetch user details
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    if not user:
        raise HTTPException(status_code=404, detail="User not found")

    # Fetch transaction history for the user
    transaction_history = transaction_collection.find({"user_id": user_id}).sort("timestamp", -1)

    portfolio_data = {
        "history": [
            {
                "timestamp": transaction["timestamp"],
                "action": transaction["action"],
                "stock": transaction["symbol"],
                "quantity": transaction["quantity"],
                "price": transaction["price"],
                "profit_loss": float(transaction.get("profit_loss", '0').replace('-', '0'))
            }
            for transaction in transaction_history
        ]
    }

    return templates.TemplateResponse("transaction_history.html", {"request": request, "portfolio": portfolio_data})

@app.get("/portfolio")
def get_portfolio(request: Request):
    # Fetch the current user from the database 
    current_user = get_current_user(request)
    print("current_user", current_user)
    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user details
    user = current_user
    
    # Get user's stock holdings from MongoDB
    holdings = list(stock_holding_collection.find({"user_id": request.session.get("user_id")}))
    print("holdings:", holdings)
    
    # Get current prices from MongoDB or any API
    symbols = [holding['company_symbol'] for holding in holdings]
    current_prices = get_current_prices(symbols)
    print("current_price", current_prices)
    
    # Get stock names
    stock_names = get_stock_names(symbols)
    
    # Calculate portfolio metrics
    portfolio = {
        "funds": user.get('cash'),
        "stocks": {}
    }
    total_investment = 0
    portfolio_value = user.get('cash')
    
    for holding in holdings:
        symbol = holding["company_symbol"]
        quantity = holding["number_of_shares"]
        avg_price = holding["avg_price"]
        
        # Add stock data to portfolio
        portfolio["stocks"][symbol] = {
            "quantity": quantity,
            "avg_price": avg_price
        }
        
        # Calculate investment and current value
        investment = quantity * avg_price
        current_value = quantity * current_prices.get(symbol, 0)
        
        total_investment += investment
        portfolio_value += current_value
    
    # Calculate profit/loss percentage
    total_profit_loss = ((portfolio_value - user.get('cash') - total_investment) / total_investment * 100) if total_investment > 0 else 0
    print('total_profit_loss:------', total_profit_loss)
    
    # Get recent transactions from MongoDB
    recent_transactions = list(transaction_collection.find({"user_id": request.session.get("user_id")}).sort("timestamp", -1).limit(10))
    
    # Generate performance data for chart
    performance_dates, performance_values = generate_performance_data()


    ########################################################################
    # Calculate profitable and losing holdings
    profitable_stocks = 0
    losing_stocks = 0
    
    for holding in holdings:
        symbol = holding["company_symbol"]
        quantity = holding["number_of_shares"]
        avg_price = holding["avg_price"]
        
        # Calculate investment and current value
        investment = quantity * avg_price
        current_value = quantity * current_prices.get(symbol, 0)
        
        # Calculate profit/loss
        profit_loss = current_value - investment
        
        # Count profitable and losing holdings
        if profit_loss > 0:
            profitable_stocks += 1
        else:
            losing_stocks += 1

    print("profit_loss:", profit_loss)
    
    # Data for graph (Profitable vs. Losing Holdings)
    labels = ['Profitable Holdings', 'Losing Holdings']
    data = [profitable_stocks, losing_stocks]

    print(data);
    print("total_profit_loss;", total_profit_loss)
    
    # Render the portfolio template
    return templates.TemplateResponse("portfolio.html", {
        "request": request,
        "portfolio": portfolio,
        "current_prices": current_prices,
        "stock_names": stock_names,
        "total_investment": total_investment,
        "portfolio_value": portfolio_value,
        "profit_loss": total_profit_loss,
        "recent_transactions": recent_transactions,
        "performance_dates": performance_dates,
        "performance_values": performance_values,
        "labels": labels,
        "data": data
    })


# ✅ Function to fetch current prices from MongoDB or an API
# def get_current_prices(symbols: list):
#     price_dict = {}
#     for symbol in symbols:
#         # Fetching a single document for each symbol
#         price = db.stock_price.find_one({"stock_id": symbol})
#         if price:
#             price_dict[symbol] = price['close']
#     return price_dict

def get_current_prices(symbols: list):
    price_dict = {}
    for symbol in symbols:
        # Fetch the latest document sorted by timestamp in descending order
        price = db.stock_price.find_one(
            {"stock_id": symbol}, 
            sort=[("date", -1)]  # Sort by timestamp descending
        )
        print("price:", price)
        if price:
            price_dict[symbol] = price['close']
    return price_dict



# ✅ Function to fetch stock names from MongoDB
def get_stock_names(symbols: list):
    stock_names = db.stocks.find({})
    name_dict = {stock['symbol']: stock['name'] for stock in stock_names}
    return name_dict


# ✅ Function to generate performance data
def generate_performance_data():
    dates = []
    values = []
    base_value = 10000
    current_value = base_value
    
    for i in range(30, 0, -1):
        date = (datetime.now() - timedelta(days=i)).strftime('%b %d')
        dates.append(date)
        
        # Random daily change between -2% and +2%
        change = random.uniform(-0.02, 0.02)
        current_value = current_value * (1 + change)
        values.append(round(current_value, 2))
    
    return dates, values


def get_current_user(request: Request):
    # Get user_id from session
    user_id = request.session.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Fetch the user from MongoDB
    user = users_collection.find_one({"_id": ObjectId(user_id)})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Return user data
    return {
        "user_id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "cash": user["cash"]
    }


##################### Dharmik ################################################
class NewsInput(BaseModel):
    news: list[str]
    
import re
from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

# Initialize the VADER sentiment analyzer
analyzer = SentimentIntensityAnalyzer()

# Function to get the sentiment label based on compound score
def get_sentiment_score(tweet):
    compound_score = analyzer.polarity_scores(tweet)['compound']
    return compound_score

    
def clean_text(text):
    text = re.sub(r'[^a-zA-Z\s]', '', text).lower()
    return text

@app.post("/analyze_sentiment")
def analyze_sentiment(data: NewsInput):
    cleaned_texts = [clean_text(news) for news in data.news]
    sentiment_scores = [get_sentiment_score(text) for text in cleaned_texts]

    if len(sentiment_scores) == 0:
        return {"sentiment_score": 0}  # Default to neutral if no input

    avg_sentiment = sum(sentiment_scores) / len(sentiment_scores)
    return {"sentiment_score": avg_sentiment}