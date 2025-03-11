from pydantic import BaseModel

class StockData(BaseModel):
    Close: float
    High: float
    Low: float
    Open: float
    Volume: int
    Tweet_Count: int
    Sentiment_Score: float