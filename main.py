# main.py
import os
import json
from typing import List, Dict, Any, Tuple
from datetime import datetime
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

# =========================
# FastAPI app
# =========================
app = FastAPI(title="Delivery Optimizer", version="1.1.0 (tolerant body)")

# =========================
# Models
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
        t = dateparser.parse(hhmm).time()
    return t.hour * 3600 + t.minute * 60 + t.second

def ors_geocode(text: str) -> Tuple[float, float] | None:
    """Return (lat, lon) using ORS geocoding. None on failure."""
    if not ORS_API_KEY:
        return None
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
        lon, lat = feats[0]["geometry"]["coordinates"]
        return (lat, lon)
    except Exception:
        return None

def ors_matrix(coords_lonlat: List[Tuple[float, float]]) -> List[List[int]]:
    """
    coords_lonlat: list[(lon,lat)] as ORS expects
    Returns durations matrix in seconds (ints)
    """
    if not ORS_API_KEY:
        raise RuntimeError("ORS_API_KEY missing; cannot call ORS matrix.")
    url = f"{ORS_BASE_URL}/v2/matrix/driving-car"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    payload = {"locations": coords_lonlat, "metrics": ["duration"], "units": "m"}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    durations = data.get("durations")
    if durations is None:
        raise RuntimeError("ORS matrix: no durations in response")
    N = len(coords_lonlat)
    out = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            val = durations[i][j]
            out[i][j] = 10**8 if val is None else int(round(val))
    return out

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
            failed_geocodes.append({"type": "order", "id": o.order_id, "query": q})
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
        q_end = f"{d.end_zipcode}, Netherlands"
        start_latlon = ors_geocode(q_start)
        end_latlon = ors_geocode(q_end)
        if not start_latlon:
            failed_geocodes.append({"type": "driver_start", "id": d.driver_id, "query": q_start})
            continue
        if not end_latlon:
            failed_geocodes.append({"type": "driver_end", "id": d.driver_id, "query": q_end})
            continue
        drivers_geo.append({
            "driver": d,
            "start_lat": start_latlon[0], "start_lon": start_latlon[1],
            "end_lat": end_latlon[0],   "end_lon": end_latlon[1],
            "start_sec": parse_hhmm_to_seconds(d.start_time),
            "stop_sec": parse_hhmm_to_seconds(d.stop_time),
        })

    if not drivers_geo:
        raise HTTPException(status_code=400, detail="No valid drivers after geocoding.")
    if not order_nodes:
        return {
            "delivery_date": day.delivery_date,
            "results": [{"driver_id": d["driver"].driver_id, "orders_planned": [], "orders_left": [], "total_time_min": 0} for d in drivers_geo],
            "failed_geocodes": failed_geocodes,
        }

    # 3) Build nodes list: starts, ends, orders
    starts_idx: List[int] = []
    ends_idx: List[int] = []
    coords_lonlat: List[Tuple[float, float]] = []  # ORS needs (lon,lat)

    for dg in drivers_geo:
        starts_idx.append(len(coords_lonlat))
        coords_lonlat.append((dg["start_lon"], dg["start_lat"]))
    for dg in drivers_geo:
        ends_idx.append(len(coords_lonlat))
        coords_lonlat.append((dg["end_lon"], dg["end_lat"]))

    order_idx_map: Dict[int, Dict[str, Any]] = {}
    for on in order_nodes:
        node_index = len(coords_lonlat)
        coords_lonlat.append((on["lon"], on["lat"]))
        order_idx_map[node_index] = on

    # 4) Time matrix
    time_matrix = ors_matrix(coords_lonlat)

    # 5) OR-Tools setup
    num_drivers = len(drivers_geo)
    num_nodes = len(coords_lonlat)

    manager = pywrapcp.RoutingIndexManager(num_nodes, num_drivers, starts_idx, ends_idx)
    routing = pywrapcp.RoutingModel(manager)

    def time_cb(from_index, to_index):
        i = manager.IndexToNode(from_index)
        j = manager.IndexToNode(to_index)
        travel = time_matrix[i][j]
        service = order_idx_map[j]["service_s"] if j in order_idx_map else 0
        return travel + service

    transit_idx = routing.RegisterTransitCallback(time_cb)
    routing.SetArcCostEvaluatorOfAllVehicles(transit_idx)

    routing.AddDimension(
        transit_idx,
        0,                 # no slack
        24 * 3600,         # upper bound
        True,              # start cumul = 0
        "Time",
    )
    time_dim = routing.GetDimensionOrDie("Time")

    # Works windows
    for v, dg in enumerate(drivers_geo):
        time_dim.CumulVar(routing.Start(v)).SetRange(dg["start_sec"], dg["start_sec"])
        time_dim.CumulVar(routing.End(v)).SetRange(0, dg["stop_sec"])

    # Capacity: max 8 orders per driver
    def demand_cb(index):
        node = manager.IndexToNode(index)
        return 1 if node in order_idx_map else 0

    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [8] * num_drivers, True, "Cap")

    # Allow dropping orders
    penalty = 3 * 3600  # seconds
    for node in order_idx_map.keys():
        routing.AddDisjunction([manager.NodeToIndex(node)], penalty)

    # Solve
    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(20)

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return {
            "delivery_date": day.delivery_date,
            "results": [{"driver_id": d["driver"].driver_id, "orders_planned": [], "orders_left": [o.order_id for o in day.orders], "total_time_min": 0} for d in drivers_geo],
            "failed_geocodes": failed_geocodes,
        }

    def is_order_node(node_id: int) -> bool:
        return node_id in order_idx_map

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
            "orders_left": [],  # vullen we zo
            "total_time_min": int(total_time_min),
        })

    all_order_ids = [o.order_id for o in day.orders]
    not_geocoded_ids = [f["id"] for f in failed_geocodes if f["type"] == "order"]
    left_ids = [oid for oid in all_order_ids if (oid not in visited_order_ids) and (oid not in not_geocoded_ids)]

    for r in driver_results:
        r["orders_left"] = left_ids

    return {
        "delivery_date": day.delivery_date,
        "results": driver_results,
        "failed_geocodes": failed_geocodes,
    }

# =========================
# Tolerant body parsing
# =========================
def _normalize_payload(obj: Any) -> List[Dict[str, Any]]:
    """
    Accept:
      - list of days
      - single day dict
      - {"body": {...}} or {"body": "[...json...]"}
    Return: list[dict]
    """
    if isinstance(obj, dict) and "body" in obj:
        body = obj["body"]
        # body could be dict, list, or JSON string
        if isinstance(body, (dict, list)):
            obj = body
        elif isinstance(body, str):
            obj = json.loads(body)

    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return obj
    raise HTTPException(status_code=400, detail="Body must be a JSON array or object.")

# =========================
# API
# =========================
@app.post("/optimize")
async def optimize(request: Request):
    """
    Accepts: list[OptimizeDay] | OptimizeDay | {"body": ...}
    Returns: list with results per day
    """
    # Try robust parsing even if Content-Type is wrong
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty body.")

    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        # fallback to FastAPI json parser
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON.")

    days = _normalize_payload(payload)

    out = []
    try:
        for entry in days:
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
