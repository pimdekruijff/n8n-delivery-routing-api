import openrouteservice
from ortools.constraint_solver import pywrapcp, routing_enums_pb2
from fastapi import FastAPI
from typing import List, Dict

app = FastAPI()
ors = openrouteservice.Client(key="5b3ce3597851110001cf6248399c7733d0464627b9b6710c1a379a74")

@app.post("/optimize")
def optimize(payload: Dict):
    delivery_date = payload["delivery_date"]
    orders = payload["orders"]
    drivers = payload["drivers"]

    results = []

    for driver in drivers:
        relevant_orders = [o for o in orders if o["delivery_date"] == delivery_date]

        if not relevant_orders:
            continue

        # ⬇️ 1. Adressen → coördinaten (geocode zelf of meegegeven)
        all_coords = [driver["start_zipcode"]] + [o["zipcode"] for o in relevant_orders] + [driver["end_zipcode"]]
        locations = [geocode_zip(zipcode) for zipcode in all_coords]  # eigen geocode functie

        # ⬇️ 2. ORS: haal tijdmatrix op
        matrix = ors.distance_matrix(locations, profile="driving-car", metrics=["duration"], resolve_locations=True)
        duration_matrix = matrix["durations"]

        # ⬇️ 3. Service time (in seconden)
        service_times = [0] + [o["service_time"] * 60 for o in relevant_orders] + [0]

        # ⬇️ 4. OR-Tools setup
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

        # ⬇️ 5. Solve
        search_params = pywrapcp.DefaultRoutingSearchParameters()
        search_params.first_solution_strategy = routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
        solution = routing.SolveWithParameters(search_params)

        if not solution:
            continue

        # ⬇️ 6. Bouw route
        index = routing.Start(0)
        route = []
        total_time = 0
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            if node_index != 0 and node_index != len(locations) - 1:
                route.append(relevant_orders[node_index - 1]["order_id"])
            prev_index = index
            index = solution.Value(routing.NextVar(index))
            total_time += routing.GetArcCostForVehicle(prev_index, index, 0)

        all_ids = [o["order_id"] for o in relevant_orders]
        left = [oid for oid in all_ids if oid not in route]

        results.append({
            "driver_id": driver["driver_id"],
            "orders_planned": route,
            "orders_left": left,
            "total_time_min": total_time // 60
        })

    return {
        "delivery_date": delivery_date,
        "results": results
    }
