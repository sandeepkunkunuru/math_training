# Introduction to Integer Programming

Integer Programming (IP) is a mathematical optimization technique where some or all of the variables are restricted to be integers. When only some variables are integers, we call it Mixed Integer Programming (MIP).

## Key Concepts in Integer Programming

1. **Integer Variables**: Variables that can only take integer values
2. **Binary Variables**: Special case of integer variables that can only be 0 or 1
3. **Combinatorial Optimization**: Finding an optimal object from a finite set of objects
4. **NP-Hard Problems**: Many integer programming problems are computationally difficult

## Standard Form of an Integer Program

An integer program in standard form looks like:

Maximize (or Minimize): $c_1x_1 + c_2x_2 + ... + c_nx_n$

Subject to:
- $a_{11}x_1 + a_{12}x_2 + ... + a_{1n}x_n \leq b_1$
- $a_{21}x_1 + a_{22}x_2 + ... + a_{2n}x_n \leq b_2$
- ...
- $a_{m1}x_1 + a_{m2}x_2 + ... + a_{mn}x_n \leq b_m$
- $x_1, x_2, ..., x_n \in \mathbb{Z}$ (integer variables)
- $x_1, x_2, ..., x_n \geq 0$ (non-negativity)

## Solving Integer Programs

Integer programs are typically solved using:

1. **Branch and Bound**: Divides the problem into subproblems and prunes branches that cannot lead to optimal solutions
2. **Cutting Planes**: Adds constraints to tighten the LP relaxation
3. **Branch and Cut**: Combines branch and bound with cutting planes
4. **Dynamic Programming**: For certain structured problems
5. **Heuristics**: For finding good (but not necessarily optimal) solutions quickly

## Applications of Integer Programming

- Facility location
- Production planning
- Scheduling
- Network design
- Vehicle routing
- Resource allocation with indivisible resources

## Modeling with Binary Variables

Binary variables (0-1 variables) are particularly useful for modeling:

1. **Logical conditions**: If-then constraints, disjunctions
2. **Fixed costs**: Costs that are incurred if and only if a certain activity takes place
3. **Selecting from alternatives**: Choosing exactly one option from many
4. **Sequencing and ordering**: Determining the order of tasks or events

## Challenges in Integer Programming

- **Integrality Gap**: Difference between the optimal values of the IP and its LP relaxation
- **Computational Complexity**: Many IP problems are NP-hard
- **Formulation Strength**: Different formulations can have significant impact on solution time
- **Symmetry**: Multiple equivalent solutions can slow down branch and bound
