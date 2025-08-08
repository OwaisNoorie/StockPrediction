import numpy as np
import pandas as pd
import os
import shutil

SOURCE_DATA_DIR = r"D:/stock_prediction/data"
MODEL_DIR = r"D:/stock_prediction/models"
DEST_STATIC_DIR = r"./static/data"

os.makedirs(DEST_STATIC_DIR, exist_ok=True)

for filename in os.listdir(SOURCE_DATA_DIR):
    if filename.endswith(".csv"):
        company = filename.replace(".csv", "")
        src = os.path.join(SOURCE_DATA_DIR, filename)
        dest = os.path.join(DEST_STATIC_DIR, filename)

        df = pd.read_csv(src)

        # Denormalize if needed
        max_path = os.path.join(MODEL_DIR, f"{company}_scaler_max.npy")
        if os.path.exists(max_path):
            max_value = np.load(max_path)
            if max_value > 1 and df['Close'].max() <= 1.0:
                df['Close'] = df['Close'] * max_value

        df.to_csv(dest, index=False)

print("✅ CSVs copied and denormalized to static/data/")
