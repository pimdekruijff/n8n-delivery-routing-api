# main.py — ORS (geocode+matrix+repair) + VROOM (met tijden) + fallback greedy
import os, json, math, time
from typing import List, Dict, Any, Tuple, Optional, Union
from datetime import datetime
import requests

from fastapi import FastAPI, HTTPException, Request
from pydantic import BaseModel
from dateutil import parser as dateparser

# =========================
# Config
# =========================
ORS_API_KEY = os.getenv("ORS_API_KEY")
ORS_BASE_URL = os.getenv("ORS_BASE_URL", "https://api.openrouteservice.org")
VROOM_URL = os.getenv("VROOM_URL", "http://localhost:3000")
VROOM_PATHS = [p.strip() for p in os.getenv("VROOM_PATHS", "/,/api/solve").split(",")]

UNREACHABLE_S = int(os.getenv("UNREACHABLE_S", "1000000"))       # ~11.6 dagen
MAX_STOPS = int(os.getenv("MAX_STOPS", "8"))
DROP_PENALTY_S = int(os.getenv("DROP_PENALTY_S", str(3 * 3600))) # vroom 'penalty'
FALLBACK_SPEED_KMH = float(os.getenv("FALLBACK_SPEED_KMH", "45"))
MATRIX_MAX_TRIES = int(os.getenv("MATRIX_MAX_TRIES", "3"))

app = FastAPI(title="Delivery Optimizer (ORS+VROOM)", version="2.0.0")

# =========================
# Models
# =========================
class Order(BaseModel):
    row_number: int
    order_id: str
    delivery_date: str
    zipcode: str
    house_number: Union[int, str]
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

def _haversine_m(coords_a: Tuple[float, float], coords_b: Tuple[float, float]) -> float:
    R = 6371000.0
    lat1, lon1 = math.radians(coords_a[0]), math.radians(coords_a[1])
    lat2, lon2 = math.radians(coords_b[0]), math.radians(coords_b[1])
    dlat = lat2 - lat1; dlon = lon2 - lon1
    a = math.sin(dlat/2)**2 + math.cos(lat1)*math.cos(lat2)*math.sin(dlon/2)**2
    c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
    return R * c

def ors_geocode_one(q: str) -> Optional[Tuple[float, float]]:
    if not ORS_API_KEY:
        return None
    url = f"{ORS_BASE_URL}/geocode/search"
    headers = {"Authorization": ORS_API_KEY}
    params = {"text": q, "boundary.country": "NL", "size": 1}
    try:
        r = requests.get(url, headers=headers, params=params, timeout=15)
        r.raise_for_status()
        feats = r.json().get("features", [])
        if not feats:
            return None
        lon, lat = feats[0]["geometry"]["coordinates"]
        return (lat, lon)
    except Exception:
        return None

def ors_matrix(coords_lonlat: List[Tuple[float, float]]) -> Tuple[List[List[Optional[int]]], Dict[str, Any]]:
    """Build ORS matrix (seconds). Return None cells if any, we'll repair."""
    if not ORS_API_KEY:
        raise RuntimeError("Missing ORS_API_KEY")
    url = f"{ORS_BASE_URL}/v2/matrix/driving-car"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    payload = {"locations": coords_lonlat, "metrics": ["duration"], "units": "m"}
    r = requests.post(url, headers=headers, json=payload, timeout=60)
    r.raise_for_status()
    data = r.json()
    durations = data.get("durations")
    if not durations:
        raise RuntimeError("ORS matrix: no durations")
    meta = {"source": "ors", "repaired_pairs": 0}
    # keep None as None for now (we'll repair)
    return durations, meta

def ors_duration_between(a: Tuple[float, float], b: Tuple[float, float]) -> Optional[int]:
    """Repair helper: ask ORS directions for a→b; returns seconds or None."""
    if not ORS_API_KEY:
        return None
    url = f"{ORS_BASE_URL}/v2/directions/driving-car"
    headers = {"Authorization": ORS_API_KEY, "Content-Type": "application/json"}
    body = {"coordinates": [[a[0], a[1]], [b[0], b[1]]]}  # [lon,lat]
    try:
        r = requests.post(url, headers=headers, json=body, timeout=30)
        r.raise_for_status()
        data = r.json()
        sec = int(round(data["routes"][0]["summary"]["duration"]))
        return sec
    except Exception:
        return None

def build_repaired_matrix(coords_lonlat: List[Tuple[float, float]]) -> Tuple[List[List[int]], Dict[str, Any]]:
    """Matrix via ORS; repair nulls via /directions; fallback Haversine if needed."""
    durations, meta = ors_matrix(coords_lonlat)
    N = len(coords_lonlat)
    repaired = [[0]*N for _ in range(N)]
    # convert to ints; repair None
    for i in range(N):
        for j in range(N):
            val = durations[i][j]
            if val is None:
                # try repair a few times with backoff
                tries = MATRIX_MAX_TRIES
                sec = None
                while tries > 0 and sec is None:
                    sec = ors_duration_between(coords_lonlat[i], coords_lonlat[j])
                    if sec is None:
                        time.sleep(0.4 * (MATRIX_MAX_TRIES - tries + 1))
                    tries -= 1
                if sec is None:
                    # fallback Haversine
                    (lon_i, lat_i) = coords_lonlat[i]
                    (lon_j, lat_j) = coords_lonlat[j]
                    meters = _haversine_m((lat_i, lon_i), (lat_j, lon_j))
                    mps = (FALLBACK_SPEED_KMH * 1000.0) / 3600.0
                    sec = max(1, int(meters / mps))
                repaired[i][j] = min(sec, UNREACHABLE_S)
                meta["repaired_pairs"] += 1
            else:
                repaired[i][j] = max(0, int(round(val)))
    return repaired, meta

def tolerant_body_to_days(obj: Any) -> List[Dict[str, Any]]:
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

def fmt_hhmm(total_sec: int) -> str:
    total_sec = max(0, int(total_sec))
    h = (total_sec // 3600) % 24
    m = (total_sec % 3600) // 60
    return f"{h:02d}:{m:02d}"

def greedy_per_driver(matrix: List[List[int]], start_idx: int, end_idx: int, order_idxs: List[int], max_stops: int) -> List[int]:
    unvisited = set(order_idxs)
    route: List[int] = []
    cur = start_idx
    for _ in range(max_stops):
        if not unvisited:
            break
        nxt = min(unvisited, key=lambda j: matrix[cur][j])
        if matrix[cur][nxt] >= UNREACHABLE_S:
            break
        route.append(nxt)
        unvisited.remove(nxt)
        cur = nxt
    return route

# =========================
# ORS + VROOM pipeline
# =========================
def solve_day_with_vroom(day: OptimizeDay) -> Dict[str, Any]:
    failed_geocodes: List[Dict[str, Any]] = []

    # 1) Geocode orders
    order_nodes: List[Dict[str, Any]] = []
    for o in day.orders:
        q = f"{o.zipcode} {o.house_number}, Netherlands"
        latlon = ors_geocode_one(q)
        if not latlon:
            failed_geocodes.append({"type": "order", "id": o.order_id, "query": q})
            continue
        order_nodes.append({
            "order": o,
            "lat": latlon[0], "lon": latlon[1],
            "service_s": int(o.service_time) * 60
        })

    # 2) Geocode drivers
    drivers_geo: List[Dict[str, Any]] = []
    for d in day.drivers:
        s = ors_geocode_one(f"{d.start_zipcode}, Netherlands")
        e = ors_geocode_one(f"{d.end_zipcode}, Netherlands")
        if not s:
            failed_geocodes.append({"type": "driver_start", "id": d.driver_id, "query": d.start_zipcode})
            continue
        if not e:
            failed_geocodes.append({"type": "driver_end", "id": d.driver_id, "query": d.end_zipcode})
            continue
        drivers_geo.append({
            "driver": d,
            "start_lat": s[0], "start_lon": s[1],
            "end_lat": e[0],   "end_lon": e[1],
            "start_sec": parse_hhmm_to_seconds(d.start_time),
            "stop_sec": parse_hhmm_to_seconds(d.stop_time),
        })

    if not drivers_geo:
        return {"status":"no_drivers","delivery_date": day.delivery_date,"results":[],"failed_geocodes":failed_geocodes}

    # Als geen orders, korte happy path
    if not order_nodes:
        return {
            "status": "ok",
            "delivery_date": day.delivery_date,
            "results": [{"driver_id": d["driver"].driver_id, "orders_planned": [], "orders_left": [], "total_time_min": 0, "stops": []} for d in drivers_geo],
            "failed_geocodes": failed_geocodes,
            "debug": {"note": "No valid orders after geocoding."}
        }

    # 3) Node lijst in één matrix-indexruimte: starts, ends, orders
    starts_idx: List[int] = []
    ends_idx: List[int] = []
    coords_lonlat: List[Tuple[float, float]] = []  # ORS expects (lon,lat)

    for dg in drivers_geo:
        starts_idx.append(len(coords_lonlat)); coords_lonlat.append((dg["start_lon"], dg["start_lat"]))
    for dg in drivers_geo:
        ends_idx.append(len(coords_lonlat));   coords_lonlat.append((dg["end_lon"], dg["end_lat"]))

    order_node_index: Dict[int, Dict[str, Any]] = {}
    for on in order_nodes:
        idx = len(coords_lonlat)
        coords_lonlat.append((on["lon"], on["lat"]))
        order_node_index[idx] = on

    # 4) ORS matrix + repair
    matrix, meta = build_repaired_matrix(coords_lonlat)

    # 5) Bouw VROOM probleem met custom matrix
    # Vehicle capacity (max stops) en time windows in absolute seconden (0=midnight)
    vehicles = []
    for v, dg in enumerate(drivers_geo):
        vehicles.append({
            "id": v + 1,
            "profile": "car",
            "start_index": starts_idx[v],
            "end_index": ends_idx[v],
            "time_window": [dg["start_sec"], dg["stop_sec"]],
            "capacity": [MAX_STOPS],
        })

    # Jobs: elk order met service, amount=1, penalty → mag ‘unassigned’
    jobs = []
    for node_idx, data in order_node_index.items():
        jobs.append({
            "id": int(abs(hash(data["order"].order_id)) % 2_000_000_000),
            "location_index": node_idx,
            "service": data["service_s"],
            "amount": [1],
            "penalty": DROP_PENALTY_S
        })

    # Custom matrix voor profiel "car"
    vroom_request = {
        "vehicles": vehicles,
        "jobs": jobs,
        "matrix": {
            "car": {
                "durations": matrix
            }
        }
    }

    # 6) Call VROOM (probeer meerdere paths)
    vroom_resp = None
    vroom_ok = False
    last_err = None
    headers = {"Content-Type": "application/json"}
    for path in VROOM_PATHS:
        url = VROOM_URL.rstrip("/") + path
        try:
            r = requests.post(url, headers=headers, data=json.dumps(vroom_request), timeout=120)
            if r.status_code == 200:
                vroom_resp = r.json()
                vroom_ok = True
                break
            last_err = f"HTTP {r.status_code}: {r.text[:300]}"
        except Exception as e:
            last_err = str(e)

    # 7) Parse VROOM → ons formaat (met tijden)
    if vroom_ok and vroom_resp and "routes" in vroom_resp:
        # Index → order_id lookup
        nodeidx_to_orderid = {idx: data["order"].order_id for idx, data in order_node_index.items()}
        results = []
        visited_ids = set()

        for v, dg in enumerate(drivers_geo):
            veh_id = v + 1
            route = next((rt for rt in vroom_resp.get("routes", []) if rt.get("vehicle") == veh_id), None)
            stops_out = []
            orders_planned = []
            total_time_min = 0

            if route:
                steps = route.get("steps", [])
                # VROOM times are seconds since 0: arrival, waiting_time, service, duration?
                for st in steps:
                    if st.get("type") == "job":
                        loc_idx = st.get("location_index")
                        if loc_idx in nodeidx_to_orderid:
                            oid = nodeidx_to_orderid[loc_idx]
                            arrival = int(st.get("arrival", 0))
                            service = int(st.get("service", 0))
                            # vertrek = arrival + service
                            depart = arrival + service
                            stops_out.append({
                                "order_id": oid,
                                "arrival_sec": arrival,
                                "departure_sec": depart,
                                "arrival_time": fmt_hhmm(arrival),
                                "departure_time": fmt_hhmm(depart),
                            })
                            orders_planned.append(oid)
                            visited_ids.add(oid)
                # totale route-tijd uit VROOM summary (als aanwezig), anders grof via laatste step arrival
                total_time_min = int(round(route.get("duration", 0) / 60)) if "duration" in route else (
                    int(steps[-1].get("arrival", 0) / 60) if steps else 0
                )

            results.append({
                "driver_id": dg["driver"].driver_id,
                "orders_planned": orders_planned,
                "orders_left": [],  # vullen we zo
                "total_time_min": total_time_min,
                "stops": stops_out,
            })

        all_ids = [o.order_id for o in day.orders]
        left_ids = [oid for oid in all_ids if oid not in visited_ids]
        for r in results:
            r["orders_left"] = left_ids

        return {
            "status": "ok",
            "delivery_date": day.delivery_date,
            "results": results,
            "failed_geocodes": failed_geocodes,
            "debug": {"matrix_meta": meta, "vroom_unassigned": vroom_resp.get("unassigned", [])}
        }

    # 8) VROOM faalde → greedy fallback (nog steeds met ORS-matrix)
    # Bouw per driver een simpele route met tijden (schatting)
    nodeidx_to_orderid = {idx: data["order"].order_id for idx, data in order_node_index.items()}
    results = []
    visited_ids = set()
    all_order_nodes = list(order_node_index.keys())

    for v, dg in enumerate(drivers_geo):
        start_node = starts_idx[v]
        end_node = ends_idx[v]
        available = [n for n in all_order_nodes if nodeidx_to_orderid[n] not in visited_ids]
        seq = greedy_per_driver(matrix, start_node, end_node, available, MAX_STOPS)

        # timings: start op start_sec; add travel+service
        t = dg["start_sec"]
        stops_out = []
        orders_planned = []
        cur = start_node
        for n in seq:
            t += matrix[cur][n]            # aankomst
            arrival = t
            service = order_node_index[n]["service_s"]
            depart = arrival + service
            stops_out.append({
                "order_id": nodeidx_to_orderid[n],
                "arrival_sec": arrival,
                "departure_sec": depart,
                "arrival_time": fmt_hhmm(arrival),
                "departure_time": fmt_hhmm(depart),
            })
            orders_planned.append(nodeidx_to_orderid[n])
            visited_ids.add(nodeidx_to_orderid[n])
            t = depart
            cur = n
        # terug naar end
        t += matrix[cur][end_node]
        total_time_min = max(0, int((t - dg["start_sec"]) / 60))

        results.append({
            "driver_id": dg["driver"].driver_id,
            "orders_planned": orders_planned,
            "orders_left": [],  # later
            "total_time_min": total_time_min,
            "stops": stops_out,
        })

    all_ids = [o.order_id for o in day.orders]
    left_ids = [oid for oid in all_ids if oid not in visited_ids]
    for r in results:
        r["orders_left"] = left_ids

    return {
        "status": "fallback",
        "delivery_date": day.delivery_date,
        "results": results,
        "failed_geocodes": failed_geocodes,
        "debug": {"matrix_meta": meta, "vroom_error": last_err}
    }

# =========================
# API
# =========================
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

    days = tolerant_body_to_days(payload)
    out = []
    for entry in days:
        day = OptimizeDay(**entry)
        out.append(solve_day_with_vroom(day))
    return out

@app.get("/healthz")
def health():
    return {"status": "ok"}
