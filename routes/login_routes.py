from fastapi import APIRouter, Request, Form
from fastapi.templating import Jinja2Templates
from fastapi.responses import RedirectResponse
import bcrypt
from pymongo import MongoClient

templates = Jinja2Templates(directory="templates")

login_router = APIRouter()
client = MongoClient("mongodb+srv://dharmikparmarpd:dhp12345@cluster0.v5pxg.mongodb.net/stock_market?retryWrites=true&w=majority&appName=Cluster0")
db = client["stock_market"]
users_collection = db["user"] 

@login_router.get("/login")
def login(request: Request):
    return templates.TemplateResponse("login.html", {"request": request})

@login_router.post("/login")
def login(request: Request, email: str = Form(...), password: str = Form(...)):
    # Check if email exists in the MongoDB database
    user = users_collection.find_one({"email": email})

    if not user:
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid Email"})

    stored_password_hash = user["password_hash"]

    # Verify the password using bcrypt
    if not bcrypt.checkpw(password.encode("utf-8"), stored_password_hash.encode("utf-8")):
        return templates.TemplateResponse("login.html", {"request": request, "error": "Invalid Password"})

    # You can store the user ID in a session or a cookie
    request.session["user_id"] = str(user["_id"])

    return RedirectResponse(url="/index", status_code=303)


@login_router.get("/logout")
def logout(request: Request):
    request.session.clear()  # Clear the session data
    return RedirectResponse(url="/login", status_code=303)
