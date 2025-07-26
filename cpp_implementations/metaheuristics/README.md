# Metaheuristics Implementation

This directory contains C++ implementations of metaheuristic optimization algorithms.

## Files

### `simulated_annealing.cpp`
Generic simulated annealing framework with multiple problem types:

- **Framework Components**:
  - **OptimizationProblem**: Abstract base class for problem definition
  - **SimulatedAnnealing**: Generic SA algorithm with configurable parameters
  - **Cooling Schedule**: Exponential temperature reduction
- **Problem Implementations**:
  - **TSP Problem**: Traveling salesman with 2-opt neighborhood
  - **Function Optimization**: Continuous optimization (Rosenbrock function)
  - **Knapsack Problem**: Binary optimization with constraint handling

### Example Applications

1. **TSP Instance**: 4-city traveling salesman problem
2. **Rosenbrock Function**: Classic continuous optimization benchmark
3. **Knapsack Problem**: Combinatorial optimization with capacity constraints

## Key Concepts Demonstrated

- **Generic Metaheuristic Design**: Template-based framework for different problem types
- **Acceptance Probability**: Metropolis criterion for worse solutions
- **Cooling Schedule**: Temperature reduction and convergence control
- **Neighborhood Generation**: Problem-specific move operators

## Algorithm Parameters

- **Initial Temperature**: Starting temperature for acceptance probability
- **Final Temperature**: Termination criterion
- **Cooling Rate**: Temperature reduction factor (typically 0.9-0.99)
- **Iterations per Temperature**: Number of moves at each temperature level

## Problem-Specific Features

- **TSP**: 2-opt moves for tour improvement
- **Function Optimization**: Gaussian perturbation with bounds checking
- **Knapsack**: Bit-flip moves with penalty for constraint violations

## Usage

```bash
g++ -std=c++17 -O2 simulated_annealing.cpp -o sa_solver
./sa_solver
```

## Learning Objectives

- Understand metaheuristic principles
- Learn simulated annealing algorithm
- Practice generic algorithm design
- Explore different problem formulations

This implementation demonstrates the flexibility of metaheuristics and provides foundation for understanding more advanced algorithms used in OR-Tools local search module.
