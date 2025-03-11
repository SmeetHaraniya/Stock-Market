from pydantic import BaseModel

class User(BaseModel):
    username: str
    email: str
    password_hash: str
    cash: float = 10000.00 