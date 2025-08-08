# main.py (robust)
import os, json
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
DEBUG_OR_TOOLS = os.getenv("DEBUG_OR_TOOLS", "0") == "1"

# Unreachable legs clamp (seconds). 1e6s ≈ 11.6 dagen; genoeg zonder solver te “breken”.
UNREACHABLE_S = int(os.getenv("UNREACHABLE_S", "1000000"))

# Max horizon (seconds) for Time dimension (laat solver ruimte)
TIME_HORIZON_S = int(os.getenv("TIME_HORIZON_S", str(48 * 3600)))

# Max stops per driver
MAX_STOPS = int(os.getenv("MAX_STOPS", "8"))

# Drop-penalty in seconden (zelfde units als cost). Groter = minder snel droppen.
DROP_PENALTY_S = int(os.getenv("DROP_PENALTY_S", str(3 * 3600)))

# =========================
# FastAPI
# =========================
app = FastAPI(title="Delivery Optimizer", version="1.2.0-robust")

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
    start_time: str
    stop_time: str

class OptimizeDay(BaseModel):
    delivery_date: str
    orders: List[Order]
    drivers: List[Driver]

# =========================
# Helpers
# =========================
def parse_hhmm_to_seconds(hhmm: str) -> int:
    try:
        t = datetime.strptime(hhmm.strip(), "%H:%M").time()
    except ValueError:
        t = dateparser.parse(hhmm).time()
    return t.hour * 3600 + t.minute * 60 + t.second

def ors_geocode(text: str) -> Tuple[float, float] | None:
    if not ORS_API_KEY:
        return None
    url = f"{ORS_BASE_URL}/geocode/search"
    headers = {"Authorization": ORS_API_KEY}
    params = {"text": text, "boundary.country": "NL", "size": 1}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        lon, lat = r.json()["features"][0]["geometry"]["coordinates"]
        return (lat, lon)
    except Exception:
        return None

def ors_matrix(coords_lonlat: List[Tuple[float, float]]) -> List[List[int]]:
    if not ORS_API_KEY:
        raise RuntimeError("ORS_API_KEY missing; cannot call ORS matrix.")
    url = f"{ORS_BASE_URL}/v2/matrix/driving-car"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    payload = {"locations": coords_lonlat, "metrics": ["duration"], "units": "m"}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    durations = r.json().get("durations")
    if durations is None:
        raise RuntimeError("ORS matrix: no durations in response")
    N = len(coords_lonlat)
    out = [[0] * N for _ in range(N)]
    for i in range(N):
        for j in range(N):
            val = durations[i][j]
            out[i][j] = UNREACHABLE_S if val is None else int(round(val))
            if out[i][j] < 0:  # guard
                out[i][j] = 0
    return out

# =========================
# VRP (single run helper)
# =========================
def _run_solver(day: OptimizeDay, clamp_windows: bool, widen_horizon: bool, ignore_end_windows: bool) -> Dict[str, Any]:
    failed_geocodes: List[Dict[str, Any]] = []

    # 1) Geocode orders
    order_nodes: List[Dict[str, Any]] = []
    for o in day.orders:
        q = f"{o.zipcode} {o.house_number}, Netherlands"
        latlon = ors_geocode(q)
        if not latlon:
            failed_geocodes.append({"type": "order", "id": o.order_id, "query": q})
            continue
        order_nodes.append({
            "order": o, "lat": latlon[0], "lon": latlon[1],
            "service_s": int(o.service_time) * 60
        })

    # 2) Geocode drivers
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
        return {
            "status": "no_drivers",
            "delivery_date": day.delivery_date,
            "results": [],
            "failed_geocodes": failed_geocodes,
            "debug": {"reason": "No valid drivers after geocoding."}
        }

    if not order_nodes:
        return {
            "status": "ok",
            "delivery_date": day.delivery_date,
            "results": [{"driver_id": d["driver"].driver_id, "orders_planned": [], "orders_left": [], "total_time_min": 0} for d in drivers_geo],
            "failed_geocodes": failed_geocodes,
            "debug": {"note": "No valid orders after geocoding."}
        }

    # 3) Build node list
    starts_idx: List[int] = []
    ends_idx: List[int] = []
    coords_lonlat: List[Tuple[float, float]] = []  # (lon,lat) for ORS

    for dg in drivers_geo:
        starts_idx.append(len(coords_lonlat))
        coords_lonlat.append((dg["start_lon"], dg["start_lat"]))
    for dg in drivers_geo:
        ends_idx.append(len(coords_lonlat))
        coords_lonlat.append((dg["end_lon"], dg["end_lat"]))

    order_idx_map: Dict[int, Dict[str, Any]] = {}
    for on in order_nodes:
        node = len(coords_lonlat)
        coords_lonlat.append((on["lon"], on["lat"]))
        order_idx_map[node] = on

    # 4) Matrix
    time_matrix = ors_matrix(coords_lonlat)

    # 5) OR-Tools
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

    horizon = TIME_HORIZON_S * (2 if widen_horizon else 1)
    routing.AddDimension(transit_idx, 0, horizon, True, "Time")
    time_dim = routing.GetDimensionOrDie("Time")

    # Windows per driver
    for v, dg in enumerate(drivers_geo):
        start_lb = dg["start_sec"]
        end_ub = dg["stop_sec"]

        # Optionally clamp: end must be >= start
        if clamp_windows and end_ub < start_lb:
            end_ub = start_lb

        # Fix start at start_sec to anchor the schedule
        time_dim.CumulVar(routing.Start(v)).SetRange(start_lb, start_lb)

        if ignore_end_windows:
            # Laat solver vrij, maar cap op horizon
            time_dim.CumulVar(routing.End(v)).SetRange(start_lb, start_lb + horizon)
        else:
            # Normaal: eindig vóór stop_sec (maar nooit vóór start)
            time_dim.CumulVar(routing.End(v)).SetRange(start_lb, max(end_ub, start_lb))

    # Capacity: max stops
    def demand_cb(index):
        return 1 if manager.IndexToNode(index) in order_idx_map else 0
    demand_idx = routing.RegisterUnaryTransitCallback(demand_cb)
    routing.AddDimensionWithVehicleCapacity(demand_idx, 0, [MAX_STOPS] * num_drivers, True, "Cap")

    # Allow dropping orders
    for node in order_idx_map.keys():
        routing.AddDisjunction([manager.NodeToIndex(node)], DROP_PENALTY_S)

    params = pywrapcp.DefaultRoutingSearchParameters()
    params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    params.local_search_metaheuristic = routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    params.time_limit.FromSeconds(20)
    if DEBUG_OR_TOOLS:
        params.log_search = True

    solution = routing.SolveWithParameters(params)
    if solution is None:
        return {
            "status": "no_solution",
            "delivery_date": day.delivery_date,
            "results": [{"driver_id": d["driver"].driver_id, "orders_planned": [], "orders_left": [o.order_id for o in day.orders], "total_time_min": 0} for d in drivers_geo],
            "failed_geocodes": failed_geocodes,
            "debug": {
                "note": "No solution under current constraints",
                "clamp_windows": clamp_windows,
                "widen_horizon": widen_horizon,
                "ignore_end_windows": ignore_end_windows,
                "horizon": horizon,
                "max_stops": MAX_STOPS,
                "drop_penalty_s": DROP_PENALTY_S,
                "unreachable_s": UNREACHABLE_S,
            }
        }

    # Extract
    def is_order_node(n: int) -> bool:
        return n in order_idx_map

    visited = set()
    driver_results = []
    for v, dg in enumerate(drivers_geo):
        idx = routing.Start(v)
        route_ids: List[str] = []
        while not routing.IsEnd(idx):
            node = manager.IndexToNode(idx)
            if is_order_node(node):
                oid = order_idx_map[node]["order"].order_id
                route_ids.append(oid)
                visited.add(oid)
            idx = solution.Value(routing.NextVar(idx))
        start_c = solution.Value(time_dim.CumulVar(routing.Start(v)))
        end_c = solution.Value(time_dim.CumulVar(routing.End(v)))
        driver_results.append({
            "driver_id": dg["driver"].driver_id,
            "orders_planned": route_ids,
            "orders_left": [],  # filled later
            "total_time_min": max(0, (end_c - start_c) // 60),
        })

    all_ids = [o.order_id for o in day.orders]
    left_ids = [oid for oid in all_ids if oid not in visited]
    for r in driver_results:
        r["orders_left"] = left_ids

    return {
        "status": "ok",
        "delivery_date": day.delivery_date,
        "results": driver_results,
        "failed_geocodes": failed_geocodes,
        "debug": {
            "clamp_windows": clamp_windows,
            "widen_horizon": widen_horizon,
            "ignore_end_windows": ignore_end_windows,
            "horizon": horizon,
            "max_stops": MAX_STOPS,
            "drop_penalty_s": DROP_PENALTY_S,
            "unreachable_s": UNREACHABLE_S,
        }
    }

# =========================
# Public API with fallback ladder
# =========================
def _normalize_payload(obj: Any) -> List[Dict[str, Any]]:
    if isinstance(obj, dict) and "body" in obj:
        body = obj["body"]
        if isinstance(body, (dict, list)):
            obj = body
        elif isinstance(body, str):
            obj = json.loads(body)
    if isinstance(obj, dict):
        return [obj]
    if isinstance(obj, list):
        return obj
    raise HTTPException(status_code=400, detail="Body must be a JSON array or object.")

@app.post("/optimize")
async def optimize(request: Request):
    raw = await request.body()
    if not raw:
        raise HTTPException(status_code=400, detail="Empty body.")
    try:
        payload = json.loads(raw.decode("utf-8"))
    except Exception:
        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON.")

    days = _normalize_payload(payload)

    results = []
    for entry in days:
        day = OptimizeDay(**entry)

        # Try strict
        res = _run_solver(day, clamp_windows=True, widen_horizon=False, ignore_end_windows=False)
        if res["status"] == "no_solution":
            # Fallback 1: widen horizon
            res = _run_solver(day, clamp_windows=True, widen_horizon=True, ignore_end_windows=False)

        if res["status"] == "no_solution":
            # Fallback 2: ignore end windows (laat solver eindtijd kiezen)
            res = _run_solver(day, clamp_windows=True, widen_horizon=True, ignore_end_windows=True)

        # Still none → return as-is; maar nooit 500
        results.append(res)

    return results

@app.get("/healthz")
def health():
    return {"status": "ok"}
