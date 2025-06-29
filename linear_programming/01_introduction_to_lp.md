# Introduction to Linear Programming

Linear Programming (LP) is a mathematical optimization technique used to find the best outcome in a mathematical model whose requirements are represented by linear relationships.

## Key Components of Linear Programming

1. **Decision Variables**: The unknowns that we're trying to determine
2. **Objective Function**: The function we're trying to maximize or minimize
3. **Constraints**: Limitations or requirements expressed as linear inequalities or equations
4. **Non-negativity Constraints**: Often variables are required to be non-negative

## Standard Form of a Linear Program

A linear program in standard form looks like:

Maximize (or Minimize): $c_1x_1 + c_2x_2 + ... + c_nx_n$

Subject to:
- $a_{11}x_1 + a_{12}x_2 + ... + a_{1n}x_n \leq b_1$
- $a_{21}x_1 + a_{22}x_2 + ... + a_{2n}x_n \leq b_2$
- ...
- $a_{m1}x_1 + a_{m2}x_2 + ... + a_{mn}x_n \leq b_m$
- $x_1, x_2, ..., x_n \geq 0$

Where:
- $x_1, x_2, ..., x_n$ are the decision variables
- $c_1, c_2, ..., c_n$ are the coefficients of the objective function
- $a_{ij}$ are the coefficients of the constraints
- $b_1, b_2, ..., b_m$ are the right-hand side values of the constraints

## Solving Linear Programs

Linear programs can be solved using various algorithms:
1. **Simplex Method**: A popular algorithm that moves from vertex to vertex along the edges of the feasible region
2. **Interior Point Methods**: Algorithms that move through the interior of the feasible region
3. **Ellipsoid Method**: A polynomial-time algorithm for solving LPs

In practice, we use software libraries like OR-Tools to solve these problems efficiently.
