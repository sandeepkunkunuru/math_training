# Constraint Programming Implementation

This directory contains C++ implementations of constraint satisfaction problems (CSPs) and related algorithms.

## Files

### `constraint_satisfaction.cpp`
A comprehensive implementation of constraint satisfaction problems including:

- **CSP Framework**: Generic CSP solver with variables, domains, and constraints
- **Constraint Types**:
  - `AllDifferentConstraint`: Binary constraint ensuring two variables have different values
  - `SumConstraint`: Arithmetic constraint for sum relationships
- **Solving Algorithm**: Backtracking search with forward checking
- **Heuristics**: 
  - MRV (Minimum Remaining Values) for variable selection
  - Basic domain value ordering

### Example Problems

1. **Map Coloring**: Classic Australia map coloring with 3 colors
2. **N-Queens**: Simplified 4-Queens problem (row constraints only)
3. **4x4 Sudoku**: Subset of Sudoku with row and column constraints

## Key Concepts Demonstrated

- **Constraint Propagation**: Forward checking to reduce domains
- **Backtracking Search**: Systematic exploration with backtracking
- **Variable Ordering**: MRV heuristic for better performance
- **Constraint Consistency**: Checking feasibility during search

## Usage

```bash
g++ -std=c++17 -O2 constraint_satisfaction.cpp -o csp_solver
./csp_solver
```

## Learning Objectives

- Understand CSP formulation and solving
- Learn constraint propagation techniques
- Explore search strategies and heuristics
- Practice implementing generic constraint frameworks

This implementation provides a foundation for understanding more advanced constraint programming concepts used in OR-Tools' CP-SAT solver.
