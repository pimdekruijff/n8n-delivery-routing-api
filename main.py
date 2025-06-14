# main.py
import pandas as pd
from fastapi import FastAPI, Query
from datetime import datetime

app = FastAPI()

@app.get("/")
def health():
    return {"status": "ok"}

@app.get("/optimize")
def optimize(delivery_date: str = Query(...)):
    df = pd.read_csv("data/orders.csv")
    df["delivery_date"] = pd.to_datetime(df["delivery_date"]).dt.date
    target_date = datetime.strptime(delivery_date, "%Y-%m-%d").date()
    df_filtered = df[df["delivery_date"] == target_date]

    if df_filtered.empty:
        return {"message": f"No orders found for {delivery_date}"}

    return {
        "date": delivery_date,
        "orders": df_filtered.to_dict(orient="records")
    }
