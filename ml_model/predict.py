import joblib
import pandas as pd

# Load the trained model
model = joblib.load("./stock_prediction_model.pkl")

# Function to make predictions
def predict_stock(data):
    df = pd.DataFrame([data])  # Convert input to DataFrame
    prediction = model.predict(df)
    return int(prediction[0])  # Convert NumPy int to Python int

# Example usage
sample_data = {
    # "Date": "2022-09-30",
    "Close": 650,
    "High": 655,
    "Low": 640,
    "Open": 660,
    "Volume": 180000,
    "Tweet_Count": 120,
    "Sentiment_Score": -0.9,
}

result = predict_stock(sample_data)
print(f"Prediction: {result} (1: Up, 0: Down)")
