# Optimization Algorithms

This directory contains C++ implementations of key optimization algorithms that are fundamental to mathematical optimization and operations research. These implementations serve as educational examples to understand the mathematical foundations of optimization methods used in Google's OR-Tools.

## Contents

1. **Unconstrained Optimization** (`unconstrained_optimization.cpp`)
   - Gradient Descent with line search
   - Newton's Method
   - BFGS (Broyden–Fletcher–Goldfarb–Shanno) algorithm
   - Test functions: Rosenbrock, Himmelblau, Sphere, Booth

2. **Constrained Optimization** (`constrained_optimization.cpp`)
   - Penalty Method
   - Augmented Lagrangian Method
   - Projected Gradient Method for box constraints
   - Test problems with equality and inequality constraints

3. **Linear Programming** (`linear_programming.cpp`)
   - Simplex Method
   - Interior Point Method
   - Standard form and canonical form conversions
   - Duality theory examples

4. **Nonlinear Programming** (`nonlinear_programming.cpp`)
   - Sequential Quadratic Programming (SQP)
   - Interior Point Methods for nonlinear problems
   - KKT (Karush–Kuhn–Tucker) conditions verification
   - Trust Region Methods

5. **Metaheuristics** (`metaheuristics.cpp`)
   - Simulated Annealing
   - Genetic Algorithms
   - Particle Swarm Optimization
   - Tabu Search

## Relevance to Optimization

### Unconstrained Optimization
Unconstrained optimization techniques are essential for:
- Finding local and global minima of objective functions
- Training machine learning models (e.g., neural networks)
- Parameter estimation in statistical models
- Solving subproblems within constrained optimization algorithms

### Constrained Optimization
Constrained optimization methods are crucial for:
- Solving real-world problems with physical or logical constraints
- Resource allocation problems with limited resources
- Engineering design problems with safety constraints
- Economic models with budget constraints

### Linear Programming
Linear programming is fundamental for:
- Production planning and scheduling
- Transportation and assignment problems
- Network flow optimization
- Resource allocation with linear constraints

### Nonlinear Programming
Nonlinear programming techniques address:
- Engineering design optimization with nonlinear physics
- Portfolio optimization with risk constraints
- Chemical process optimization
- Machine learning with regularization

### Metaheuristics
Metaheuristic algorithms are valuable for:
- Combinatorial optimization problems
- Problems with complex, non-differentiable objective functions
- Multi-objective optimization
- Problems where gradient information is unavailable or unreliable

## Connection to OR-Tools

These optimization implementations provide the mathematical foundation for several features in Google's OR-Tools:

1. **Linear Solver Library**:
   - Our linear programming implementations demonstrate the core algorithms behind OR-Tools' linear solver
   - Understanding duality and sensitivity analysis as implemented in OR-Tools

2. **Constraint Solver**:
   - The constrained optimization techniques relate to how OR-Tools handles constraints
   - Penalty and augmented Lagrangian methods inform constraint propagation techniques

3. **Vehicle Routing Library**:
   - Metaheuristics implemented here are similar to those used in OR-Tools' routing solvers
   - Understanding local search and global optimization techniques

4. **Assignment Problems**:
   - Linear assignment algorithms connect to OR-Tools' assignment solvers
   - Network flow algorithms relate to min-cost flow solvers

## Building and Running

To build all the examples in this directory:

```bash
cd /path/to/math_training
mkdir build && cd build
cmake ..
make
```

To run a specific example:

```bash
./cpp_implementations/optimization/unconstrained_optimization
./cpp_implementations/optimization/constrained_optimization
./cpp_implementations/optimization/linear_programming
./cpp_implementations/optimization/nonlinear_programming
./cpp_implementations/optimization/metaheuristics
```

## Further Reading

- "Numerical Optimization" by Jorge Nocedal and Stephen J. Wright
- "Convex Optimization" by Stephen Boyd and Lieven Vandenberghe
- "Introduction to Linear Optimization" by Dimitris Bertsimas and John N. Tsitsiklis
- "Metaheuristics: From Design to Implementation" by El-Ghazali Talbi
- "Algorithms for Optimization" by Mykel J. Kochenderfer and Tim A. Wheeler
