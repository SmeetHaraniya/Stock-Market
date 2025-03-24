from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from pymongo import MongoClient
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix,precision_score, recall_score, f1_score, accuracy_score, classification_report

from routes.stock_routes import fetch_last_test_data, get_last_date_data


# ✅ Load Processed Dataset (Ensure this is your final cleaned dataset)
file_path = "/content/drive/MyDrive/stock_market_2/processed_final_dataset.csv"
df = pd.read_csv(file_path)

test_data = fetch_last_test_data()
print(test_data)

symbols = ['TSLA', 'MSFT', 'META', 'AMZN', "AAPL"]
for symbol in symbols:
    actual_last_date_data = get_last_date_data(symbol)

    print(-2)
    test_symbol_data = test_data[test_data["Stock Name"] == symbol].sort_values(by="Date", ascending=False).iloc[0]
    print(actual_last_date_data)
    print(-1)
    print(test_symbol_data)
    # Extract necessary values
    actual_close = actual_last_date_data["Close"]
    test_close = test_symbol_data["Close"]
    predicted_movement = test_symbol_data["prediction"]  # Assuming `prediction` column exists

    # Determine Actual Movement (UP or DOWN)
    actual_movement = "UP" if actual_close > test_close else "DOWN"
    test_symbol_data["Target"] = 1 if actual_movement == "UP" else 0
    df = pd.concat([df, pd.DataFrame([test_symbol_data])], ignore_index=True)

df.drop(columns=["prediction"], inplace=True)
df.to_csv("/content/drive/MyDrive/stock_market_2/processed_final_dataset.csv", index=False)

features = ['Open', 'High', 'Low', 'Close', 'Volume', 'Tweet_Count',
            'Sentiment_Score', 'Weighted_Sentiment', 'MA5', 'MA20',
            'Sentiment_Lag_1', 'Sentiment_Lag_2', 'Price_Lag_1', 'Price_Lag_2']

target = "Target" # 1 (Up) / 0 (Down)

# ✅ Define X (features) and y (target)
X = df[features]
y = df[target]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)

print(f"Training Data: {X_train.shape}, Testing Data: {X_test.shape}")




from sklearn.ensemble import RandomForestClassifier

# Initialize and train the model
# rfc_model = RandomForestClassifier(random_state=42)
rfc_model = RandomForestClassifier(max_features=X_train.shape[1]-1, bootstrap=True, n_estimators=150)
rfc_model.fit(X_train, y_train)

# Make predictions
y_pred = rfc_model.predict(X_test)

# Evaluate the model
accuracy = accuracy_score(y_test, y_pred)
precision = precision_score(y_test, y_pred, average='weighted')
recall = recall_score(y_test, y_pred, average='weighted')
f1 = f1_score(y_test, y_pred, average='weighted')

print(f"Accuracy: {accuracy:.2f}")
print(f"Precision: {precision:.2f}")
print(f"Recall: {recall:.2f}")
print(f"F1-Score: {f1:.2f}")

report = classification_report(y_test, y_pred)

# Print Results
print()
print("Classification Report:\n", report)


# Plot confusion matrices for both models
# plot_confusion_matrix(y_test, y_pred, "Random Forest Classifier")

import joblib

# Save the trained model
import os

# Get the current directory# Save the trained model
model_filename = "stock_prediction_model.pkl"
joblib.dump(rfc_model, model_filename)
print(f"Model saved as {model_filename}")

# Plot confusion matrices for both models
# plot_confusion_matrix(y_test, y_pred, "Random Forest Classifier")

import joblib

# Save the trained model
import os

# Get the current directory
current_dir = os.getcwd()

# Define the full path for the model
model_filename = os.path.join(current_dir, "stock_prediction_model.pkl")
joblib.dump(rfc_model, model_filename)
print(f"Model saved as {model_filename}")





client = MongoClient("mongodb+srv://dharmikparmarpd:dhp12345@cluster0.v5pxg.mongodb.net/stock_market?retryWrites=true&w=majority&appName=Cluster0")

db = client["stock_market"]
history_collection = db["history_collection"]  # Collection for test data


def update_history():
    test_data = fetch_last_test_data()

    symbols = ['TSLA', 'MSFT', 'META', 'AMZN', "AAPL"]
    for symbol in symbols:
        actual_last_date_data = get_last_date_data(symbol)

        test_symbol_data = test_data[test_data["Stock Name"] == symbol].sort_values(by="Date", ascending=False).iloc[0]
        print(test_symbol_data)
        # Extract necessary values
        actual_close = actual_last_date_data["Close"]
        test_close = test_symbol_data["Close"]
        predicted_movement = test_symbol_data["prediction"]  # Assuming `prediction` column exists

        # Determine Actual Movement (UP or DOWN)
        actual_movement = "UP" if actual_close > test_close else "DOWN"



        history_record = {
            "Date": actual_last_date_data["Date"],
            "Stock Name": symbol,
            "Open Price": actual_last_date_data["Open"],
            "Close Price": actual_close,
            "Actual Movement": actual_movement,
            "Predicted Movement": predicted_movement,
            "created_at": datetime.utcnow()
        }

        # Insert into history collection
        history_collection.insert_one(history_record)

    print("History updated successfully!")

# update_history()