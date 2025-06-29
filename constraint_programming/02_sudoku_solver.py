#!/usr/bin/env python3
"""
Sudoku Solver using Constraint Programming with Google OR-Tools

This example demonstrates how to model and solve a Sudoku puzzle using
constraint programming. Sudoku is a classic example that showcases the
power of constraint programming for problems with many constraints.
"""

from ortools.sat.python import cp_model
import numpy as np


def print_sudoku_solution(solution, grid_size=9):
    """Prints the Sudoku solution in a readable format."""
    # Determine the size of each sub-grid
    sub_grid_size = int(grid_size ** 0.5)
    
    # Print the top border
    print("+" + "-" * (grid_size * 2 + sub_grid_size - 1) + "+")
    
    for i in range(grid_size):
        line = "|"
        for j in range(grid_size):
            if j > 0 and j % sub_grid_size == 0:
                line += "|"
            line += f" {solution[i * grid_size + j]}"
        line += " |"
        print(line)
        
        if i < grid_size - 1 and (i + 1) % sub_grid_size == 0:
            print("|" + "-" * (grid_size * 2 + sub_grid_size - 1) + "|")
    
    # Print the bottom border
    print("+" + "-" * (grid_size * 2 + sub_grid_size - 1) + "+")


def solve_sudoku(initial_grid):
    """
    Solves a Sudoku puzzle using constraint programming.
    
    Args:
        initial_grid: A list of 81 integers where 0 represents empty cells
                     and 1-9 represent fixed values.
    
    Returns:
        A list of 81 integers with the solved puzzle, or None if no solution exists.
    """
    grid_size = 9
    sub_grid_size = 3
    
    # Create the model
    model = cp_model.CpModel()
    
    # Create the variables
    cells = {}
    for i in range(grid_size):
        for j in range(grid_size):
            # For each cell, create a variable with domain [1-9]
            cells[(i, j)] = model.NewIntVar(1, grid_size, f'cell_{i}_{j}')
    
    # Add the constraints
    
    # 1. Each row contains all digits from 1 to 9
    for i in range(grid_size):
        model.AddAllDifferent([cells[(i, j)] for j in range(grid_size)])
    
    # 2. Each column contains all digits from 1 to 9
    for j in range(grid_size):
        model.AddAllDifferent([cells[(i, j)] for i in range(grid_size)])
    
    # 3. Each 3x3 sub-grid contains all digits from 1 to 9
    for box_i in range(0, grid_size, sub_grid_size):
        for box_j in range(0, grid_size, sub_grid_size):
            box_vars = []
            for i in range(box_i, box_i + sub_grid_size):
                for j in range(box_j, box_j + sub_grid_size):
                    box_vars.append(cells[(i, j)])
            model.AddAllDifferent(box_vars)
    
    # 4. Fixed cells from the initial grid
    for i in range(grid_size):
        for j in range(grid_size):
            cell_value = initial_grid[i * grid_size + j]
            if cell_value != 0:  # If the cell has a fixed value
                model.Add(cells[(i, j)] == cell_value)
    
    # Create a solver and solve the model
    solver = cp_model.CpSolver()
    status = solver.Solve(model)
    
    # Check if a solution was found
    if status == cp_model.OPTIMAL or status == cp_model.FEASIBLE:
        solution = [0] * (grid_size * grid_size)
        for i in range(grid_size):
            for j in range(grid_size):
                solution[i * grid_size + j] = solver.Value(cells[(i, j)])
        return solution
    else:
        return None


def main():
    # Example Sudoku puzzle (0 represents empty cells)
    # This is a medium difficulty puzzle
    initial_grid = [
        5, 3, 0, 0, 7, 0, 0, 0, 0,
        6, 0, 0, 1, 9, 5, 0, 0, 0,
        0, 9, 8, 0, 0, 0, 0, 6, 0,
        8, 0, 0, 0, 6, 0, 0, 0, 3,
        4, 0, 0, 8, 0, 3, 0, 0, 1,
        7, 0, 0, 0, 2, 0, 0, 0, 6,
        0, 6, 0, 0, 0, 0, 2, 8, 0,
        0, 0, 0, 4, 1, 9, 0, 0, 5,
        0, 0, 0, 0, 8, 0, 0, 7, 9
    ]
    
    print("Initial Sudoku puzzle:")
    print_sudoku_solution(initial_grid)
    
    print("\nSolving...")
    solution = solve_sudoku(initial_grid)
    
    if solution:
        print("\nSolution found:")
        print_sudoku_solution(solution)
        
        # Verify the solution
        rows_ok = all(len(set(solution[i*9:(i+1)*9])) == 9 for i in range(9))
        cols_ok = all(len(set(solution[i::9])) == 9 for i in range(9))
        
        # Check 3x3 sub-grids
        sub_grids_ok = True
        for box_i in range(0, 9, 3):
            for box_j in range(0, 9, 3):
                sub_grid = []
                for i in range(box_i, box_i + 3):
                    for j in range(box_j, box_j + 3):
                        sub_grid.append(solution[i * 9 + j])
                if len(set(sub_grid)) != 9:
                    sub_grids_ok = False
                    break
        
        print(f"\nSolution verification:")
        print(f"Rows valid: {'Yes' if rows_ok else 'No'}")
        print(f"Columns valid: {'Yes' if cols_ok else 'No'}")
        print(f"Sub-grids valid: {'Yes' if sub_grids_ok else 'No'}")
        
    else:
        print("\nNo solution found!")


if __name__ == "__main__":
    main()
