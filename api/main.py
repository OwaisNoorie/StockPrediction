from fastapi import FastAPI, HTTPException
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse, JSONResponse
from fastapi.middleware.cors import CORSMiddleware  # 👈 Add CORS support
from api.predict import predict_price
import pandas as pd
import os

app = FastAPI(title="Stock Price Prediction API")

# ✅ Enable CORS to allow frontend on a different port (e.g. 5500)
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],  # In production, replace "*" with your frontend URL
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# ✅ Serve static files (e.g. CSVs, frontend assets)
app.mount("/static", StaticFiles(directory="static"), name="static")

# ✅ Route for frontend
@app.get("/")
def serve_spa():
    return FileResponse("frontend/index.html")

# ✅ API route to predict stock price
@app.get("/predict/{company}")
def get_prediction(company: str):
    try:
        company = company.upper()

        # Get predicted price
        predicted_price = predict_price(company)

        # Load latest actual price from CSV
        DATA_DIR = r"D:/stock_prediction/data"
        csv_path = os.path.join(DATA_DIR, f"{company}.csv")

        if not os.path.exists(csv_path):
            raise FileNotFoundError(f"CSV file not found for {company}")

        df = pd.read_csv(csv_path)[['Close']].dropna()
        actual_price = df['Close'].iloc[-1]

        return JSONResponse(content={
            "company": company,
            "predicted_price": predicted_price,
            "actual_price": actual_price
        })

    except FileNotFoundError:
        raise HTTPException(status_code=404, detail=f"Model or data not found for {company}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Internal server error: {str(e)}")
