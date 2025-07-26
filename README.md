# Optimization Mathematics Learning Project

This repository contains learning materials, examples, and exercises to help learn optimization mathematics from scratch, with a focus on understanding and eventually contributing to Google's OR-Tools project.

## Directory Structure

### Core Mathematical Foundations
- **cpp_implementations/linear_algebra/**: Vector operations, matrices, eigenvalues, transformations
- **cpp_implementations/calculus/**: Derivatives, gradients, convexity, Lagrangian methods
- **cpp_implementations/probability/**: Distributions, random variables, Monte Carlo methods

### Optimization Algorithms
- **cpp_implementations/optimization/**: Unconstrained/constrained optimization, simplex method
- **cpp_implementations/linear_programming/**: Linear programming examples and theory
- **cpp_implementations/integer_programming/**: Branch and bound, cutting planes, MIP
- **cpp_implementations/constraint_programming/**: CSP solving, backtracking, constraint propagation
- **cpp_implementations/network_flows/**: Max flow, min cost flow, shortest paths

### Advanced Topics
- **routing/**: Vehicle routing problems and solutions
- **examples/**: General examples covering various optimization topics
- **resources/**: Additional learning resources, cheat sheets, and references

## Getting Started

1. Review the `optimization_learning_plan.md` file for a structured learning path
2. Install OR-Tools and dependencies (see below)
3. Work through the examples in order of increasing complexity

## Installing OR-Tools

For Python:
```bash
pip install ortools
```

For other languages and more detailed installation instructions, refer to the [official documentation](https://developers.google.com/optimization/install).

## Implementation Status

### ✅ Completed
- **Linear Algebra**: Complete with vector operations, matrix operations, eigenvalues, and transformations
- **Calculus**: Derivatives, gradients, convexity analysis, and Lagrangian methods
- **Probability**: Distributions, random variables, Monte Carlo, and stochastic processes
- **Linear Programming**: Full simplex method implementation with examples
- **Constraint Programming**: CSP framework with backtracking and forward checking
- **Integer Programming**: Branch and bound algorithm with MIP examples
- **Network Flows**: Max flow algorithms (Ford-Fulkerson, Edmonds-Karp, Dinic)

### 🚧 In Progress
- **Routing**: Vehicle routing and TSP implementations
- **Scheduling**: Job shop and resource allocation problems
- **Metaheuristics**: Local search and evolutionary algorithms

## Learning Path

The recommended learning path is:

1. **Mathematical Foundations** (linear algebra, calculus, probability)
2. **Linear Programming** basics and simplex method
3. **Constraint Programming** concepts and CSP solving
4. **Integer and Mixed-Integer Programming** with branch and bound
5. **Network Flows** and graph algorithms
6. **Specialized topics** (routing, scheduling, metaheuristics)

Each directory contains examples with increasing complexity, starting with basic concepts and progressing to more advanced topics.

## Resources

- [OR-Tools Documentation](https://developers.google.com/optimization)
- [OR-Tools GitHub Repository](https://github.com/google/or-tools)
- See `optimization_learning_plan.md` for recommended books and courses
