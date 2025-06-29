#!/usr/bin/env python3
"""
Simple Linear Programming Example using Google OR-Tools

Problem:
A furniture company makes tables and chairs. Each table requires 2 hours of
carpentry and 1 hour of finishing. Each chair requires 1 hour of carpentry and
2 hours of finishing. The company has 8 hours of carpentry time and 8 hours of
finishing time available each day. Each table yields a profit of $60 and each
chair a profit of $40. How many tables and chairs should the company make each
day to maximize profit?

Mathematical formulation:
Let x = number of tables
Let y = number of chairs

Maximize: 60x + 40y (profit)
Subject to:
  2x + y ≤ 8 (carpentry time)
  x + 2y ≤ 8 (finishing time)
  x, y ≥ 0 (non-negativity)
"""

from ortools.linear_solver import pywraplp


def main():
    # Create the linear solver with the GLOP backend
    solver = pywraplp.Solver.CreateSolver("GLOP")
    if not solver:
        print("Could not create solver")
        return

    # Create the variables x and y
    x = solver.NumVar(0, solver.infinity(), "x")  # Number of tables
    y = solver.NumVar(0, solver.infinity(), "y")  # Number of chairs

    print("Number of variables =", solver.NumVariables())

    # Create the constraints
    # Constraint 1: 2x + y ≤ 8 (carpentry time)
    solver.Add(2 * x + y <= 8)
    
    # Constraint 2: x + 2y ≤ 8 (finishing time)
    solver.Add(x + 2 * y <= 8)

    print("Number of constraints =", solver.NumConstraints())

    # Create the objective function: 60x + 40y
    solver.Maximize(60 * x + 40 * y)

    # Solve the system
    status = solver.Solve()

    # Print the solution
    if status == pywraplp.Solver.OPTIMAL:
        print("Solution:")
        print("Objective value =", solver.Objective().Value())
        print("x =", x.solution_value())
        print("y =", y.solution_value())
        
        # Additional analysis
        print("\nAdvanced analysis:")
        print("Problem solved in %f milliseconds" % solver.wall_time())
        print("Problem solved in %d iterations" % solver.iterations())
        
        # Shadow prices (dual values)
        print("\nDual values (shadow prices):")
        for i in range(solver.NumConstraints()):
            print(f"Constraint {i}: {solver.constraint(i).dual_value()}")
        
        # Reduced costs
        print("\nReduced costs:")
        print(f"x: {x.reduced_cost()}")
        print(f"y: {y.reduced_cost()}")
        
    else:
        print("The problem does not have an optimal solution.")


if __name__ == "__main__":
    main()
