from fastapi import APIRouter, Form, Request
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import bcrypt
from pymongo import MongoClient

templates = Jinja2Templates(directory="templates")

signup_router = APIRouter()
client = MongoClient("mongodb+srv://dharmikparmarpd:dhp12345@cluster0.v5pxg.mongodb.net/stock_market?retryWrites=true&w=majority&appName=Cluster0")
db = client["stock_market"]
users_collection = db["user"]  

@signup_router.get("/signup")
def signup(request: Request):
    return templates.TemplateResponse("signup.html", {"request": request})

@signup_router.post("/signup")
def signup(request: Request, username: str = Form(...), email: str = Form(...), password: str = Form(...)):
    # Check if email already exists in the MongoDB database
    existing_user = users_collection.find_one({"email": email})

    if existing_user:
        return templates.TemplateResponse("signup.html", {"request": request, "error": "Email is already registered, please login"})

    # Hash the password before storing
    hashed_password = bcrypt.hashpw(password.encode("utf-8"), bcrypt.gensalt())

    # Insert the new user into MongoDB
    users_collection.insert_one({
        "username": username,
        "email": email,
        "password_hash": hashed_password.decode("utf-8"),
        "cash": 10000,
    })

    return RedirectResponse(url="/login", status_code=303)
