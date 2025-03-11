from pydantic import BaseModel
from typing import Optional

class StockPrice(BaseModel):
    stock_id: str
    date: str
    open: float
    high: float
    low: float
    close: float
    volume: int
    sma_20: Optional[float]
    sma_50: Optional[float]
    rsi_14: Optional[float]

