# Integer Programming Implementation

This directory contains C++ implementations of integer programming algorithms and techniques.

## Files

### `branch_and_bound.cpp`
A complete implementation of the branch and bound algorithm for integer linear programming:

- **BranchNode Structure**: Represents nodes in the branch and bound tree
- **SimplexSolver**: Simplified LP solver for demonstration (in practice, use robust solvers)
- **BranchAndBoundSolver**: Main algorithm implementation with:
  - Priority queue for best-first search
  - Branching on fractional variables
  - Bound-based pruning
  - Integer feasibility checking

### Example Problems

1. **Integer Knapsack**: Binary variables with capacity constraint
2. **Mixed Integer Programming**: Combination of integer and continuous variables
3. **Facility Location**: Binary facility decisions with continuous flow variables

## Key Concepts Demonstrated

- **LP Relaxation**: Solving continuous relaxation for bounds
- **Branching Strategy**: Variable selection and bound creation
- **Pruning**: Eliminating suboptimal branches
- **Best-First Search**: Using priority queue for efficient exploration

## Algorithm Features

- **Node Selection**: Best-first using objective value
- **Variable Selection**: Most fractional variable heuristic
- **Termination**: Optimal solution or node limit reached
- **Solution Tracking**: Maintains best integer solution found

## Usage

```bash
g++ -std=c++17 -O2 branch_and_bound.cpp -o bb_solver
./bb_solver
```

## Learning Objectives

- Understand branch and bound methodology
- Learn integer programming formulations
- Practice tree-based search algorithms
- Explore mixed-integer optimization

This implementation demonstrates the core concepts behind commercial MIP solvers used in OR-Tools and other optimization packages.
