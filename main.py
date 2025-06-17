from fastapi import FastAPI, Request
from typing import List, Dict
from datetime import datetime
import openrouteservice
from ortools.constraint_solver import pywrapcp, routing_enums_pb2

app = FastAPI()
ORS_API_KEY = "your_ors_key"  # <-- Zet dit via env var in production
ors = openrouteservice.Client(key=ORS_API_KEY)

@app.get("/")
def health():
    return {"status": "ok"}

@app.post("/optimize")
async def optimize_all(request: Request):
    data = await request.json()
    delivery_date = data.get("delivery_date")
    drivers = data.get("drivers", [])
    orders = data.get("orders", [])

    if not delivery_date or not drivers or not orders:
        return {"error": "Missing delivery_date, drivers or orders"}

    result = []

    for driver in drivers:
        relevant_orders = [
            o for o in orders if o["delivery_date"] == delivery_date
        ]

        if not relevant_orders:
            result.append({
                "driver_id": driver["driver_id"],
                "orders_planned": [],
                "orders_left": [],
                "total_time_min": 0
            })
            continue

        # Vereenvoudigde placeholder
        # Hier zou je geocoding doen + ORS matrix ophalen + OR-Tools oplossen
        # Voor nu selecteren we maximaal 3 orders als mock
        planned = relevant_orders[:3]
        leftover = relevant_orders[3:]

        result.append({
            "driver_id": driver["driver_id"],
            "orders_planned": [o["order_id"] for o in planned],
            "orders_left": [o["order_id"] for o in leftover],
            "total_time_min": 180  # <-- placeholder
        })

    return {
        "delivery_date": delivery_date,
        "results": result
    }
