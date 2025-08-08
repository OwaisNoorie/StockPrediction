import numpy as np
import pandas as pd
from tensorflow.keras.models import load_model
import os

# Paths
DATA_DIR = r"D:/stock_prediction/data"
MODEL_DIR = r"D:/stock_prediction/models"
SEQUENCE_LENGTH = 60

def predict_price(company: str) -> float:
    company = company.upper()

    # Step 1: Load CSV data
    csv_path = os.path.join(DATA_DIR, f"{company}.csv")
    if not os.path.exists(csv_path):
        raise FileNotFoundError(f"CSV file not found for {company}")

    df = pd.read_csv(csv_path)[['Close']].dropna()
    if len(df) < SEQUENCE_LENGTH:
        raise ValueError("Not enough data for prediction")

    # Step 2: Load scaler max value from .npy
    max_path = os.path.join(MODEL_DIR, f"{company}_scaler_max.npy")
    if not os.path.exists(max_path):
        raise FileNotFoundError(f"Scaler max file not found for {company}")

    max_value = np.load(max_path)
    if max_value == 0:
        raise ValueError("Invalid scaler max value (0)")

    # Step 3: Scale the last SEQUENCE_LENGTH prices
    scaled_data = df[['Close']].values / max_value
    last_sequence = scaled_data[-SEQUENCE_LENGTH:]
    X_input = np.reshape(last_sequence, (1, SEQUENCE_LENGTH, 1))

    # Step 4: Load trained LSTM model
    model_path = os.path.join(MODEL_DIR, f"{company}_lstm_model.h5")
    if not os.path.exists(model_path):
        raise FileNotFoundError(f"Model file not found for {company}")

    model = load_model(model_path)

    # Step 5: Predict next price (scaled)
    prediction = model.predict(X_input)
    if prediction.ndim == 2:
        scaled_prediction = prediction[0][0]
    elif prediction.ndim == 1:
        scaled_prediction = prediction[0]
    else:
        raise ValueError("Unexpected prediction shape")

    # Step 6: Inverse scale the prediction
    predicted_price = scaled_prediction * max_value

    return float(predicted_price)
