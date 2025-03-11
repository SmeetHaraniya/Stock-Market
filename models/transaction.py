from pydantic import BaseModel
from typing import Optional
from datetime import datetime

class Transaction(BaseModel):
    user_id: str
    action: str
    symbol: str
    quantity: int
    price: float
    timestamp: datetime = datetime.now()
    profit_loss: str