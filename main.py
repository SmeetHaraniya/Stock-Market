from datetime import datetime

from bson import ObjectId
import config
from fastapi import FastAPI, Form, HTTPException, Query, Request
from fastapi.responses import  RedirectResponse
from starlette.middleware.sessions import SessionMiddleware
from pymongo import MongoClient
from fastapi.templating import Jinja2Templates
from datetime import datetime
import yfinance as yf

from routes.stock_routes import stock_router
from routes.user_routes import user_router
from routes.login_routes import login_router
from routes.signup_routes import signup_router

from fastapi.templating import Jinja2Templates

app = FastAPI()

client = MongoClient("mongodb+srv://dharmikparmarpd:dhp12345@cluster0.v5pxg.mongodb.net/stock_market?retryWrites=true&w=majority&appName=Cluster0")
db = client['stock_market'] 

users_collection = db["user"] 
stock_collection = db['stock']
# stock_price_collection = db['stock_price']
stock_holding_collection = db['stock_holding']
transaction_collection = db['transaction']
history_collection = db['history_collection']

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


def get_real_time_data(symbols):
    try:
        tickers = yf.download(symbols, period="1d", interval="1m")['Close'].iloc[-1]
        return {symbol: round(tickers[symbol], 6) for symbol in symbols}
    except Exception as e:
        return {"error": str(e)}


@app.get("/index")
def index(request: Request):
    rows = stock_collection.find({}, {"symbol": 1, "name": 1})

    symbols = ["AAPL", "AMZN", "TSLA", "MSFT", "META"]

    print(get_real_time_data(symbols))

    # Return the template response
    return templates.TemplateResponse("index.html", {
        "request": request,
        "stocks": rows,
        "closing_values": get_real_time_data(symbols)
    })


# use for widget
@app.get("/stock/{symbol}")
def stock_details(request: Request, symbol: str):
    # Fetch stock details using the symbol
    stock = stock_collection.find_one({"symbol": symbol})

    if stock is None:
        return {"error": "Stock not found"}

    # Fetch stock price history for the given stock_id, sorted by latest date
    # prices = list(stock_price_collection.find({"stock_id": stock["symbol"]}).sort("date", -1))

    # for price in prices:
    #     if "_id" in price:
    #         price["_id"] = str(price["_id"])


    histories = list(history_collection.find({"Stock Name": symbol}).sort("Date", -1))
    
    for history in histories:
        if "_id" in history:
            history["_id"] = str(history["_id"])
            
    return templates.TemplateResponse("stock_details.html", {
        "request": request,
        "stock": stock,
        "histories": histories
        # "bars": prices,
    })


# use for redirect page...
@app.get("/trade_stocks")
def trade_stocks(request:Request):
    return templates.TemplateResponse("trade_stocks.html",{"request":request})


# use for fetching real time data for buying and selling...   
@app.get("/get-stock-price/{symbol}")
def get_stock_price(request: Request, symbol:str):

    try:
        data = yf.download(symbol, period="1d", interval="1m")  # Fetch stock data
        close_price = data['Close'].iloc[-1]  # Get the latest closing price

        return {"price": round(close_price[symbol], 2)}
    except Exception as e:
        return {"error": str(e)}


@app.post("/buy")
def buy(request: Request, symbol: str = Form(...), quantity: int = Form(...), price: float = Form(...)):
    user_id = request.session.get("user_id")

    user = users_collection.find_one({"_id": ObjectId(user_id)})
    # user = user.to_list()[0]
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

    
    return {"message": "Stock purchased successfully"}


@app.post("/sell")
def sell(request: Request, symbol: str = Form(...), quantity: int = Form(...), price: float = Form(...)):
    user_id = request.session.get("user_id")
    
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
    
    if not current_user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Get user details
    user = current_user
    
    # Get user's stock holdings from MongoDB
    holdings = list(stock_holding_collection.find({"user_id": request.session.get("user_id")}))
    
    # Get current prices from MongoDB or any API
    symbols = [holding['company_symbol'] for holding in holdings]


    # Fetch data from yfinance data...
    current_prices = get_real_time_data(symbols)
    
    # Get stock names
    # stock_names = get_stock_names(symbols)
    
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
    
    # Get recent transactions from MongoDB
    recent_transactions = list(transaction_collection.find({"user_id": request.session.get("user_id")}).sort("timestamp", -1).limit(10))
    
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

    
    # Data for graph (Profitable vs. Losing Holdings)
    labels = ['Profitable Holdings', 'Losing Holdings']
    data = [profitable_stocks, losing_stocks]

    # Render the portfolio template
    return templates.TemplateResponse("portfolio.html", {
        "request": request,
        "portfolio": portfolio,
        "current_prices": current_prices,
        # "stock_names": stock_names,
        "total_investment": total_investment,
        "portfolio_value": portfolio_value,
        "profit_loss": total_profit_loss,
        "recent_transactions": recent_transactions,
        "labels": labels,
        "data": data
    })


@app.route("/prediction-history", methods=["GET"])
def model_prediction_history(request: Request):
    histories = list(history_collection.find().sort("Date", -1))
    
    for history in histories:
        if "_id" in history:
            history["_id"] = str(history["_id"])
            
    return templates.TemplateResponse("prediction_history.html", {
        "request": request,
        "histories": histories
    })
    

# Function to fetch stock names from MongoDB
# def get_stock_names(symbols: list):
#     stock_names = db.stocks.find({})
#     name_dict = {stock['symbol']: stock['name'] for stock in stock_names}
#     print(-1,name_dict)
#     return name_dict


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

