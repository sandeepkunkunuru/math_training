# Learning Plan for Optimization Mathematics and OR-Tools

## Phase 1: Mathematical Foundations (4-6 weeks)

### Week 1-2: Linear Algebra Refresher
- **Topics**: Vectors, matrices, linear transformations, eigenvalues/eigenvectors
- **Resources**: 
  - Gilbert Strang's "Linear Algebra and Its Applications"
  - Khan Academy Linear Algebra course
  - 3Blue1Brown's "Essence of Linear Algebra" YouTube series

### Week 3-4: Calculus and Analysis
- **Topics**: Derivatives, gradients, Lagrangian methods, convexity
- **Resources**:
  - Stewart's "Calculus"
  - Boyd & Vandenberghe's "Convex Optimization" (Chapters 1-3)

### Week 5-6: Probability and Statistics
- **Topics**: Random variables, distributions, expectation, variance
- **Resources**:
  - Wasserman's "All of Statistics"
  - Khan Academy Probability and Statistics

## Phase 2: Optimization Fundamentals (6-8 weeks)

### Week 1-2: Introduction to Optimization
- **Topics**: Problem formulation, objective functions, constraints
- **Resources**:
  - Nocedal & Wright's "Numerical Optimization" (Chapters 1-2)
  - OR-Tools documentation: https://developers.google.com/optimization/introduction

### Week 3-4: Linear Programming
- **Topics**: Simplex method, duality, sensitivity analysis
- **Resources**:
  - Bertsimas & Tsitsiklis "Introduction to Linear Optimization"
  - OR-Tools linear solver examples (Glop)

### Week 5-6: Integer Programming and Mixed Integer Linear Programming
- **Topics**: Branch and bound, cutting planes, relaxations
- **Resources**:
  - Wolsey's "Integer Programming"
  - OR-Tools MIP examples

### Week 7-8: Constraint Programming
- **Topics**: Constraint satisfaction problems, propagation, search strategies
- **Resources**:
  - Rossi, Van Beek & Walsh "Handbook of Constraint Programming" (selected chapters)
  - OR-Tools CP-SAT solver documentation

## Phase 3: Advanced Topics and OR-Tools Specifics (8-10 weeks)

### Week 1-2: Network Flows and Graph Algorithms
- **Topics**: Shortest paths, max flow, min cost flow
- **Resources**:
  - Ahuja, Magnanti & Orlin "Network Flows"
  - OR-Tools graph module examples

### Week 3-4: Vehicle Routing Problems
- **Topics**: TSP, VRP, capacitated VRP
- **Resources**:
  - Toth & Vigo "Vehicle Routing: Problems, Methods, and Applications"
  - OR-Tools routing module tutorials

### Week 5-6: Scheduling and Resource Allocation
- **Topics**: Job shop scheduling, resource constraints
- **Resources**:
  - Pinedo's "Scheduling: Theory, Algorithms, and Systems"
  - OR-Tools scheduling examples

### Week 7-8: Metaheuristics and Local Search
- **Topics**: Simulated annealing, tabu search, genetic algorithms
- **Resources**:
  - Gendreau & Potvin "Handbook of Metaheuristics"
  - OR-Tools local search examples

### Week 9-10: Advanced Solvers and Integration
- **Topics**: Commercial solvers, solver integration, hybrid approaches
- **Resources**:
  - OR-Tools documentation on solver integration
  - OR-Tools examples using external solvers (Gurobi, SCIP, etc.)

## Phase 4: Practical Application and Contribution (Ongoing)

### Step 1: Set Up Development Environment
- Install OR-Tools and dependencies
- Configure build system (Bazel, CMake)
- Run examples and tests

### Step 2: Explore the Codebase
- Study the architecture and component interactions
- Understand the solver implementations
- Review existing issues and pull requests

### Step 3: Start Contributing
- Fix simple bugs or documentation issues
- Implement small enhancements
- Work on test cases

### Step 4: Advanced Contributions
- Implement new algorithms or heuristics
- Optimize existing implementations
- Contribute to core solver functionality

## Practical Learning Project Ideas

1. **Beginner**: Solve classic optimization problems (knapsack, assignment, etc.) using OR-Tools
2. **Intermediate**: Implement a real-world scheduling or routing application
3. **Advanced**: Develop a custom constraint or cutting plane for the CP-SAT or MIP solver
