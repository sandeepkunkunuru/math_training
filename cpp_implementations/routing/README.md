# Routing Algorithms Implementation

This directory contains C++ implementations of routing and vehicle routing algorithms.

## Files

### `traveling_salesman.cpp`
Comprehensive implementation of TSP algorithms:

- **Exact Algorithms**:
  - **Brute Force**: Complete enumeration for small instances (≤10 cities)
  - **Dynamic Programming**: Held-Karp algorithm with O(n²2ⁿ) complexity
- **Heuristic Algorithms**:
  - **Nearest Neighbor**: Greedy construction heuristic
  - **2-opt Local Search**: Local improvement method
- **Problem Instances**: Small examples, medium instances, and random generation

### Example Problems

1. **Small Instance**: 5-city problem comparing all methods
2. **Medium Instance**: 10-city problem with heuristics
3. **Random Instance**: Performance testing with timing

## Key Concepts Demonstrated

- **Exact vs Heuristic Methods**: Trade-offs between optimality and efficiency
- **Local Search**: 2-opt neighborhood and improvement strategies
- **Construction Heuristics**: Nearest neighbor with different starting points
- **Performance Analysis**: Runtime comparison and solution quality

## Algorithm Complexity

- **Brute Force**: O(n!) - only feasible for very small instances
- **Dynamic Programming**: O(n²2ⁿ) - feasible up to ~20 cities
- **Nearest Neighbor**: O(n²) - fast but approximate
- **2-opt**: O(n²) per iteration - good balance of speed and quality

## Usage

```bash
g++ -std=c++17 -O2 traveling_salesman.cpp -o tsp_solver
./tsp_solver
```

## Learning Objectives

- Understand TSP problem formulation
- Compare exact and approximate algorithms
- Learn local search principles
- Practice algorithm complexity analysis

This implementation provides foundation concepts for vehicle routing problems and demonstrates techniques used in OR-Tools routing module.
