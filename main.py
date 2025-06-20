
import os
import openrouteservice
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from fastapi import FastAPI
from typing import List, Dict

app = FastAPI()
ors = openrouteservice.Client(key=os.getenv("ORS_API_KEY"))

zip_cache = {}

def geocode_zip(zipcode: str):
    if zipcode in zip_cache:
        return zip_cache[zipcode]

    try:
        result = ors.pelias_search(text=zipcode, size=1)
        coords = result['features'][0]['geometry']['coordinates']
        zip_cache[zipcode] = coords
        return coords
    except Exception as e:
        print(f"Geocode error for {zipcode}: {e}")
        return None

def tijd_in_seconden(start: str, stop: str):
    from datetime import datetime
    fmt = "%H:%M"
    t1 = datetime.strptime(start, fmt)
    t2 = datetime.strptime(stop, fmt)
    return int((t2 - t1).total_seconds())

@app.post("/optimize")
def optimize(payload: Dict):
    delivery_date = payload["delivery_date"]
    orders = payload["orders"]
    drivers = payload["drivers"]

    results = []
    failed_geocodes = []

    for driver in drivers:
        if driver["delivery_date"] != delivery_date:
            print(f"❌ Driver {driver['driver_id']} heeft een andere datum: {driver['delivery_date']}")
            continue

        relevant_orders = [o for o in orders if o["delivery_date"] == delivery_date]
        print(f"✅ Driver {driver['driver_id']} heeft {len(relevant_orders)} relevante orders")

        if not relevant_orders:
            continue

        all_coords_zip = [driver["start_zipcode"]] + [o["zipcode"] for o in relevant_orders] + [driver["end_zipcode"]]
        all_coords = []
        failed_zips = []

        for z in all_coords_zip:
            coord = geocode_zip(z)
            if coord is None:
                failed_zips.append(z)
            else:
                all_coords.append(coord)

        if len(all_coords) < 3:
            print(f"❌ Niet genoeg geocode resultaten om route te berekenen voor driver {driver['driver_id']}")
            failed_geocodes.extend(failed_zips)
            continue

        try:
            matrix = ors.distance_matrix(all_coords, profile="driving-car", metrics=["duration"], resolve_locations=True)
            duration_matrix = matrix["durations"]
        except Exception as e:
            print(f"❌ ORS distance_matrix error: {e}")
            continue

        service_times = [0] + [o["service_time"] * 60 for o in relevant_orders if o["zipcode"] not in failed_zips] + [0]

        manager = pywrapcp.RoutingIndexManager(len(duration_matrix), 1, 0, len(duration_matrix) - 1)
        routing = pywrapcp.RoutingModel(manager)

        def time_callback(from_index, to_index):
            from_node = manager.IndexToNode(from_index)
            to_node = manager.IndexToNode(to_index)
            return duration_matrix[from_node][to_node] + service_times[from_node]

        transit_callback_index = routing.RegisterTransitCallback(time_callback)
        routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)

        routing.AddDimension(
            transit_callback_index,
            0,
            tijd_in_seconden(driver["start_time"], driver["stop_time"]),
            True,
            "Time"
        )

        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        solution = routing.SolveWithParameters(search_params)

        if not solution:
            print(f"❌ Geen oplossing gevonden voor driver {driver['driver_id']}")
            continue

        index = routing.Start(0)
        route = []
        total_time = 0
        filtered_orders = [o for o in relevant_orders if o["zipcode"] not in failed_zips]
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0 and node_index != len(all_coords) - 1:
                route.append(filtered_orders[node_index - 1]["order_id"])
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            total_time += routing.GetArcCostForVehicle(prev_index, index, 0)

        all_ids = [o["order_id"] for o in filtered_orders]
        left = [oid for oid in all_ids if oid not in route]

        results.append({
            "driver_id": driver["driver_id"],
            "orders_planned": route,
            "orders_left": left,
            "total_time_min": total_time // 60
        })

        failed_geocodes.extend(failed_zips)

    return {
        "delivery_date": delivery_date,
        "results": results,
        "failed_geocodes": list(set(failed_geocodes))
    }
