#!/usr/bin/env python3
"""
Capacitated Vehicle Routing Problem (CVRP) using Google OR-Tools

This example demonstrates how to model and solve a capacitated vehicle routing problem
where a fleet of vehicles with limited capacity must deliver goods to customers
while minimizing the total distance traveled.

Problem:
- A depot has a fleet of vehicles with limited capacity
- Multiple customers need deliveries of different quantities
- Each vehicle starts and ends at the depot
- Each customer must be visited exactly once
- The objective is to minimize the total distance traveled
"""

from ortools.constraint_solver import routing_enums_pb2
from ortools.constraint_solver import pywrapcp
import numpy as np
import matplotlib.pyplot as plt


def create_data_model():
    """Creates and returns the problem data model."""
    data = {}
    # Locations in (x, y) format: depot is at index 0, customers are at indices 1+
    data['locations'] = [
        (0, 0),    # Depot
        (10, 10),  # Customer 1
        (20, 20),  # Customer 2
        (30, 30),  # Customer 3
        (40, 40),  # Customer 4
        (50, 50),  # Customer 5
        (60, 60),  # Customer 6
        (70, 70),  # Customer 7
        (80, 80),  # Customer 8
        (90, 90),  # Customer 9
        (10, 90),  # Customer 10
        (90, 10),  # Customer 11
        (30, 70),  # Customer 12
        (70, 30),  # Customer 13
        (50, 10),  # Customer 14
        (10, 50),  # Customer 15
    ]
    
    # Calculate distances between locations
    num_locations = len(data['locations'])
    dist_matrix = np.zeros((num_locations, num_locations))
    
    for i in range(num_locations):
        for j in range(num_locations):
            # Euclidean distance
            x1, y1 = data['locations'][i]
            x2, y2 = data['locations'][j]
            dist_matrix[i][j] = int(np.sqrt((x2 - x1)**2 + (y2 - y1)**2))
    
    data['distance_matrix'] = dist_matrix.astype(int).tolist()
    
    # Demands for each location (depot has no demand)
    data['demands'] = [0]  # Depot
    # Random demands for customers between 1 and 9
    np.random.seed(42)  # For reproducibility
    data['demands'].extend(list(np.random.randint(1, 10, size=num_locations-1)))
    
    # Vehicle capacities
    data['vehicle_capacities'] = [30, 30, 30, 30]  # 4 vehicles with capacity 30 each
    
    # Number of vehicles
    data['num_vehicles'] = len(data['vehicle_capacities'])
    
    # Depot index
    data['depot'] = 0
    
    return data


def print_solution(data, manager, routing, solution):
    """Prints the solution."""
    print(f"Objective: {solution.ObjectiveValue()}")
    
    total_distance = 0
    total_load = 0
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        plan_output = f"Route for vehicle {vehicle_id}:\n"
        route_distance = 0
        route_load = 0
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            route_load += data['demands'][node_index]
            plan_output += f" {node_index} Load({route_load}) -> "
            previous_index = index
            index = solution.Value(routing.NextVar(index))
            route_distance += routing.GetArcCostForVehicle(previous_index, index, vehicle_id)
        
        plan_output += f" {manager.IndexToNode(index)} Load({route_load})\n"
        plan_output += f"Distance of the route: {route_distance}m\n"
        plan_output += f"Load of the route: {route_load}\n"
        print(plan_output)
        
        total_distance += route_distance
        total_load += route_load
    
    print(f"Total distance of all routes: {total_distance}m")
    print(f"Total load of all routes: {total_load}")


def plot_solution(data, manager, routing, solution):
    """Plots the solution."""
    plt.figure(figsize=(10, 10))
    
    # Plot depot
    depot_x, depot_y = data['locations'][0]
    plt.scatter(depot_x, depot_y, c='red', s=200, marker='*')
    plt.text(depot_x, depot_y, "Depot", fontsize=12)
    
    # Plot customers
    for i in range(1, len(data['locations'])):
        x, y = data['locations'][i]
        plt.scatter(x, y, c='blue', s=100)
        plt.text(x, y, f"{i}({data['demands'][i]})", fontsize=10)
    
    # Plot routes
    colors = ['green', 'orange', 'purple', 'brown', 'pink', 'gray', 'olive', 'cyan']
    
    for vehicle_id in range(data['num_vehicles']):
        index = routing.Start(vehicle_id)
        route_x = []
        route_y = []
        
        while not routing.IsEnd(index):
            node_index = manager.IndexToNode(index)
            x, y = data['locations'][node_index]
            route_x.append(x)
            route_y.append(y)
            index = solution.Value(routing.NextVar(index))
        
        # Add depot at the end
        node_index = manager.IndexToNode(index)
        x, y = data['locations'][node_index]
        route_x.append(x)
        route_y.append(y)
        
        plt.plot(route_x, route_y, c=colors[vehicle_id % len(colors)], linewidth=2, 
                 label=f"Vehicle {vehicle_id}")
    
    plt.legend()
    plt.title('Capacitated Vehicle Routing Problem Solution')
    plt.xlabel('X Coordinate')
    plt.ylabel('Y Coordinate')
    plt.grid(True)
    plt.savefig('vrp_solution.png')
    plt.close()


def main():
    """Entry point of the program."""
    # Create the data model
    data = create_data_model()
    
    # Create the routing index manager
    manager = pywrapcp.RoutingIndexManager(
        len(data['distance_matrix']), 
        data['num_vehicles'], 
        data['depot']
    )
    
    # Create the routing model
    routing = pywrapcp.RoutingModel(manager)
    
    # Create and register a transit callback
    def distance_callback(from_index, to_index):
        """Returns the distance between the two nodes."""
        # Convert from routing variable Index to distance matrix NodeIndex
        from_node = manager.IndexToNode(from_index)
        to_node = manager.IndexToNode(to_index)
        return data['distance_matrix'][from_node][to_node]
    
    transit_callback_index = routing.RegisterTransitCallback(distance_callback)
    
    # Define cost of each arc
    routing.SetArcCostEvaluatorOfAllVehicles(transit_callback_index)
    
    # Add capacity constraint
    def demand_callback(from_index):
        """Returns the demand of the node."""
        # Convert from routing variable Index to demands NodeIndex
        from_node = manager.IndexToNode(from_index)
        return data['demands'][from_node]
    
    demand_callback_index = routing.RegisterUnaryTransitCallback(demand_callback)
    
    routing.AddDimensionWithVehicleCapacity(
        demand_callback_index,
        0,  # null capacity slack
        data['vehicle_capacities'],  # vehicle maximum capacities
        True,  # start cumul to zero
        'Capacity'
    )
    
    # Setting first solution heuristic
    search_parameters = pywrapcp.DefaultRoutingSearchParameters()
    search_parameters.first_solution_strategy = (
        routing_enums_pb2.FirstSolutionStrategy.PATH_CHEAPEST_ARC
    )
    search_parameters.local_search_metaheuristic = (
        routing_enums_pb2.LocalSearchMetaheuristic.GUIDED_LOCAL_SEARCH
    )
    search_parameters.time_limit.seconds = 30
    
    # Solve the problem
    solution = routing.SolveWithParameters(search_parameters)
    
    # Print solution
    if solution:
        print("Solution found!")
        print_solution(data, manager, routing, solution)
        plot_solution(data, manager, routing, solution)
        print("Solution plot saved as 'vrp_solution.png'")
    else:
        print("No solution found!")


if __name__ == "__main__":
    main()
