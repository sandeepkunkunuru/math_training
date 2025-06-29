#!/usr/bin/env python3
"""
Knapsack Problem using Integer Programming with Google OR-Tools

The knapsack problem is a classic optimization problem:
Given a set of items, each with a weight and a value, determine which items to include
in a collection so that the total weight is less than or equal to a given limit (capacity)
and the total value is as large as possible.

This example demonstrates how to model and solve the knapsack problem using
integer programming with OR-Tools.
"""

from ortools.linear_solver import pywraplp
import numpy as np


def solve_knapsack(values, weights, capacity):
    """
    Solves the knapsack problem using integer programming.
    
    Args:
        values: List of values for each item
        weights: List of weights for each item
        capacity: Maximum weight capacity of the knapsack
    
    Returns:
        A tuple (total_value, selected_items) where selected_items is a list of
        indices of the selected items.
    """
    n = len(values)  # Number of items
    
    # Create the solver
    solver = pywraplp.Solver.CreateSolver('SCIP')
    if not solver:
        return None
    
    # Create binary decision variables
    x = {}
    for i in range(n):
        x[i] = solver.IntVar(0, 1, f'x_{i}')
    
    # Add capacity constraint
    solver.Add(sum(weights[i] * x[i] for i in range(n)) <= capacity)
    
    # Set objective function
    objective = solver.Objective()
    for i in range(n):
        objective.SetCoefficient(x[i], values[i])
    objective.SetMaximization()
    
    # Solve the problem
    status = solver.Solve()
    
    # Process the solution
    if status == pywraplp.Solver.OPTIMAL:
        total_value = solver.Objective().Value()
        selected_items = [i for i in range(n) if x[i].solution_value() > 0.5]
        return total_value, selected_items
    else:
        return None


def main():
    # Example data
    values = [60, 100, 120, 80, 30]  # Value of each item
    weights = [10, 20, 30, 15, 5]    # Weight of each item
    capacity = 50                    # Maximum weight capacity
    
    print("Knapsack Problem Example")
    print("=======================")
    print(f"Capacity: {capacity}")
    print("\nItems:")
    for i in range(len(values)):
        print(f"Item {i+1}: Value = {values[i]}, Weight = {weights[i]}")
    
    # Solve the problem
    result = solve_knapsack(values, weights, capacity)
    
    if result:
        total_value, selected_items = result
        
        print("\nSolution:")
        print(f"Total value: {total_value}")
        print("Selected items:")
        
        total_weight = 0
        for i in selected_items:
            print(f"Item {i+1}: Value = {values[i]}, Weight = {weights[i]}")
            total_weight += weights[i]
        
        print(f"\nTotal weight: {total_weight}/{capacity}")
        print(f"Unused capacity: {capacity - total_weight}")
        
        # Verify the solution
        verification_value = sum(values[i] for i in selected_items)
        verification_weight = sum(weights[i] for i in selected_items)
        
        print("\nVerification:")
        print(f"Calculated total value: {verification_value}")
        print(f"Calculated total weight: {verification_weight}")
        print(f"Solution is {'valid' if verification_weight <= capacity else 'invalid'}")
        
        # Compare with greedy approach
        print("\nComparison with Greedy Approach:")
        value_per_weight = [(values[i] / weights[i], i) for i in range(len(values))]
        value_per_weight.sort(reverse=True)  # Sort by value/weight ratio
        
        greedy_items = []
        greedy_weight = 0
        greedy_value = 0
        
        for ratio, i in value_per_weight:
            if greedy_weight + weights[i] <= capacity:
                greedy_items.append(i)
                greedy_weight += weights[i]
                greedy_value += values[i]
        
        print(f"Greedy approach value: {greedy_value}")
        print(f"Optimal approach value: {total_value}")
        print(f"Improvement: {((total_value - greedy_value) / greedy_value) * 100:.2f}%")
        
    else:
        print("\nNo solution found!")


if __name__ == "__main__":
    main()
