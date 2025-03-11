from bson import ObjectId
from fastapi import APIRouter, HTTPException, Request
from models.user import User
from pymongo import MongoClient

user_router = APIRouter()
client = MongoClient("mongodb+srv://dharmikparmarpd:dhp12345@cluster0.v5pxg.mongodb.net/stock_market?retryWrites=true&w=majority&appName=Cluster0")
db = client["stock_market"]

user_collection = db['user']

@user_router.post("/users/", response_model=User)
def create_user(user: User):
    if db.user.find_one({"email": user.email}):
        raise HTTPException(status_code=400, detail="Email already registered")
    result = db.user.insert_one(user.dict())
    return {"id": str(result.inserted_id), **user.dict()}

@user_router.get("/users/")
def get_users():
    users = list(db.user.find())
    for user in users:
        user["id"] = str(user["_id"])
        del user["_id"]
    return {"message": "users fetched successfully"}

@user_router.get("/get_current_user")
def get_current_user(request: Request):
    # Get user_id from session
    user_id = request.session.get("user_id")
    
    if not user_id:
        raise HTTPException(status_code=401, detail="Not authenticated")
    
    # Fetch the user from MongoDB
    user = user_collection.find_one({"_id": ObjectId(user_id)})
    
    if not user:
        raise HTTPException(status_code=404, detail="User not found")
    
    # Return user data
    return {
        "user_id": str(user["_id"]),
        "username": user["username"],
        "email": user["email"],
        "cash": user["cash"]
    }