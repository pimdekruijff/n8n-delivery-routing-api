# main.py
import os
import math
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime, time
import requests

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dateutil import parser as dateparser

from ortools.constraint_solver import pywrapcp, routing_enums_pb2


# =========================
# Config
# =========================
ORS_API_KEY = os.getenv("ORS_API_KEY")
ORS_BASE_URL = os.getenv("ORS_BASE_URL", "https://api.openrouteservice.org")

if not ORS_API_KEY:
    print("[WARN] ORS_API_KEY not set. Geocoding/matrix calls will fail.")


# =========================
# FastAPI app
# =========================
app = FastAPI(title="Delivery Optimizer", version="1.0.0")


# =========================
# Models (light typing)
# =========================
class Order(BaseModel):
    row_number: int
    order_id: str
    delivery_date: str
    zipcode: str
    house_number: int | str
    service_time: int  # minutes


class Driver(BaseModel):
    row_number: int
    driver_id: str
    start_zipcode: str
    end_zipcode: str
    delivery_date: str
    start_time: str  # "9:00"
    stop_time: str   # "17:00"


class OptimizeDay(BaseModel):
    delivery_date: str
    orders: List[Order]
    drivers: List[Driver]


# =========================
# Helpers
# =========================
def parse_hhmm_to_seconds(hhmm: str) -> int:
    """'9:00' -> seconds since midnight."""
    try:
        t = datetime.strptime(hhmm.strip(), "%H:%M").time()
    except ValueError:
        # Try H:M or H.M fallbacks
        t = dateparser.parse(hhmm).time()
    return t.hour * 3600 + t.minute * 3600 // 60 + t.second


def ors_geocode(text: str) -> Tuple[float, float] | None:
    """Return (lat, lon) using ORS geocoding. None on failure."""
    url = f"{ORS_BASE_URL}/geocode/search"
    headers = {"Authorization": ORS_API_KEY}
    params = {"text": text, "boundary.country": "NL", "size": 1}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        data = r.json()
        feats = data.get("features", [])
        if not feats:
            return None
        coords = feats[0]["geometry"]["coordinates"]  # [lon, lat]
        lon, lat = coords
        return (lat, lon)
    except Exception:
        return None


def ors_matrix(coords_lonlat: List[Tuple[float, float]]) -> List[List[int]]:
    """
    coords_lonlat: list[(lon,lat)]  (ORS expects lon,lat)
    Returns durations (seconds) full matrix NxN
    """
    url = f"{ORS_BASE_URL}/v2/matrix/driving-car"
    headers = {
        "Authorization": ORS_API_KEY,
        "Content-Type": "application/json",
    }
    payload = {
        "locations": coords_lonlat,  # [[lon,lat], ...]
        "metrics": ["duration"],
        "units": "m",
    }
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    durations = data.get("durations")
    if durations is None:
        raise RuntimeError("ORS matrix: no durations in response")
    # ORS returns float seconds (or minutes depending on units). For driving-car it's seconds.
    # If it ever returns None for a leg, set to a large number.
    N = len(coords_lonlat)
    matrix = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            val = durations[i][j]
            if val is None:
                matrix[i][j] = 10**8  # effectively impossible
            else:
                matrix[i][j] = int(round(val))
    return matrix


# =========================
# Core VRP
# =========================
def solve_vrp_for_day(day: OptimizeDay) -> Dict[str, Any]:
    failed_geocodes: List[Dict[str, Any]] = []

    # 1) Geocode all orders
    order_nodes: List[Dict[str, Any]] = []
    for o in day.orders:
        q = f"{o.zipcode} {o.house_number}, Netherlands"
        latlon = ors_geocode(q)
        if not latlon:
            failed_geocodes.append(
                {"type": "order", "id": o.order_id, "query": q}
            )
            continue
        order_nodes.append({
            "order": o,
            "lat": latlon[0],
            "lon": latlon[1],
            "service_s": int(o.service_time) * 60
        })

    # 2) Geocode drivers start/end
    drivers_geo: List[Dict[str, Any]] = []
    for d in day.drivers:
        q_start = f"{d.start_zipcode}, Netherlands"
        start_latlon = ors_geocode(q_start)
        if not start_latlon:
            failed_geocodes.append({"type": "driver_start", "id": d.driver_id, "query": q_start})
            continue
        q_end = f"{d.end_zipcode}, Netherlands"
        end_latlon = ors_geocode(q_end)
        if not end_latlon:
            failed_geocodes.append({"type": "driver_end", "id": d.driver_id, "query": q_end})
            continue
        drivers_geo.append({
            "driver": d,
            "start_lat": start_latlon[0],
            "start_lon": start_latlon[1],
            "end_lat": end_latlon[0],
            "end_lon": end_latlon[1],
            "start_sec": parse_hhmm_to_seconds(d.start_time),
            "stop_sec": parse_hhmm_to_seconds(d.stop_time),
        })

    if not drivers_geo:
        raise HTTPException(status_code=400, detail="No valid drivers after geocoding.")
    if not order_nodes:
        # Return empty but valid
        return {
            "delivery_date": day.delivery_date,
            "results": [{"driver_id": d["driver"].driver_id, "orders_planned": [], "orders_left": [], "total_time_min": 0} for d in drivers_geo],
            "failed_geocodes": failed_geocodes,
        }

    # 3) Build nodes list for matrix: [driver starts..., driver ends..., orders...]
    # We'll map indexes → entities
    starts_idx = []
    ends_idx = []
    coords_lonlat: List[Tuple[float, float]] = []  # ORS wants (lon,lat)

    for dg in drivers_geo:
        starts_idx.append(len(coords_lonlat))
        coords_lonlat.append((dg["start_lon"], dg["start_lat"]))
    for dg in drivers_geo:
        ends_idx.append(len(coords_lonlat))
        coords_lonlat.append((dg["end_lon"], dg["end_lat"]))

    order_idx_map = {}  # node_index -> order object
    for on in order_nodes:
        node_index = len(coords_lonlat)
        coords_lonlat.append((on["lon"], on["lat"]))
        order_idx_map[node_index] = on

    # 4) Get time matrix
    try:
        time_matrix = ors_matrix(coords_lonlat)
    except Exception as e:
        raise HTTPException(status_code=502, detail=f"ORS matrix failed: {e}")

    num_drivers = len(drivers_geo)
    num_nodes = len(coords_lonlat)
    num_orders = len(order_idx_map)

    # 5) OR-Tools Manager & Routing
    manager = pywrapcp.RoutingIndexManager(
        num_nodes,
        num_drivers,
        starts_idx,
        ends_idx
    )
    routing = pywrapcp.RoutingModel(manager)

    # Transit (travel + service time at arrival node if it's an order)
    def time_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        travel = time_matrix[i][j]
        service = 0
        if j in order_idx_map:
            service = order_idx_map[j]["service_s"]
        return travel + service

    transit_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    # Time dimension
    routing.AddDimension(
        transit_idx,
        0,                     # no slack
        24 * 3600,             # big upper bound
        True,                  # fix start cumul to zero per vehicle
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Vehicle work windows
    for v, dg in enumerate(drivers_geo):
        start_var = time_dim.CumulVar(routing.Start(v))
        end_var = time_dim.CumulVar(routing.End(v))
        # Start at driver's declared start time
        start_var.SetRange(dg["start_sec"], dg["start_sec"])
        # Must finish before stop_sec (allow equal)
        end_var.SetRange(0, dg["stop_sec"])

    # Capacity "max 8 orders per driver"
    def demand_cb(index):
        node = manager.IndexToNode(index)
        return 1 if node in order_idx_map else 0

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    capacities = [8] * num_drivers
    routing.AddDimensionWithVehicleCapacity(
        demand_idx, 0, capacities, True, "Cap"
    )

    # Allow dropping orders (with penalty)
    penalty = 3 * 3600  # seconds; tune as needed
    for node in order_idx_map.keys():
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Search parameters
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(20)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        # No solution → everything 'left'
        return {
            "delivery_date": day.delivery_date,
            "results": [{"driver_id": d["driver"].driver_id, "orders_planned": [], "orders_left": [o.order_id for o in day.orders], "total_time_min": 0} for d in drivers_geo],
            "failed_geocodes": failed_geocodes,
        }

    # 6) Extract per driver
    def is_order_node(node_id: int) -> bool:
        return node_id in order_idx_map

    # Track visited orders
    visited_order_ids = set()
    driver_results = []

    for v, dg in enumerate(drivers_geo):
        idx = routing.Start(v)
        route_order_ids: List[str] = []

        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if is_order_node(node):
                oid = order_idx_map[node]["order"].order_id
                route_order_ids.append(oid)
                visited_order_ids.add(oid)
            idx = solution.Value(routing.NextVar(idx))

        start_cumul = solution.Value(time_dim.CumulVar(routing.Start(v)))
        end_cumul = solution.Value(time_dim.CumulVar(routing.End(v)))
        total_time_min = max(0, (end_cumul - start_cumul) // 60)

        driver_results.append({
            "driver_id": dg["driver"].driver_id,
            "orders_planned": route_order_ids,
            "orders_left": [],  # filled after loop, globally
            "total_time_min": int(total_time_min),
        })

    # Orders left = not visited (including geocode failures)
    all_order_ids = [o.order_id for o in day.orders]
    not_geocoded_ids = [f["id"] for f in failed_geocodes if f["type"] == "order"]
    left_ids = [oid for oid in all_order_ids if (oid not in visited_order_ids) and (oid not in not_geocoded_ids)]

    # Put same 'orders_left' for each driver in the day's result (zoals je huidige structuur)
    for r in driver_results:
        r["orders_left"] = left_ids

    return {
        "delivery_date": day.delivery_date,
        "results": driver_results,
        "failed_geocodes": failed_geocodes,
    }


# =========================
# API
# =========================
@app.post("/optimize")
async def optimize(request: Request):
    """
    Body: List[OptimizeDay]  (zoals jouw n8n flow stuurt)
    Returns: dezelfde lijst met resultaten per dag.
    """
    try:
        payload = await request.json()
        if not isinstance(payload, list):
            raise HTTPException(status_code=400, detail="Body must be a JSON array.")

        out = []
        for entry in payload:
            day = OptimizeDay(**entry)
            out.append(solve_vrp_for_day(day))
        return out

    except HTTPException:
        raise
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/healthz")
def health():
    return {"status": "ok"}
