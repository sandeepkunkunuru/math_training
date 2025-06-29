# Linear Algebra for Optimization

This directory contains C++ implementations of key linear algebra concepts that are fundamental to optimization algorithms and techniques. These implementations serve as educational examples to understand the mathematical foundations of optimization methods.

## Contents

1. **Vector Operations** (`vector_operations.cpp`)
   - Basic vector operations (addition, subtraction, scalar multiplication)
   - Dot product and cross product
   - Vector normalization
   - Vector projection
   - Tests for orthogonality and parallelism

2. **Matrix Operations** (`matrix_operations.cpp`)
   - Basic matrix operations (addition, subtraction, scalar multiplication)
   - Matrix multiplication and matrix-vector multiplication
   - Transpose, determinant, and trace
   - Matrix inverse (for 2x2 matrices)
   - Tests for matrix symmetry

3. **Linear Transformations** (`linear_transformations.cpp`)
   - 2D and 3D transformation matrices (rotation, scaling, shear)
   - Composition of transformations
   - Area preservation tests
   - Applications of linear transformations

4. **Eigenvalues and Eigenvectors** (`eigenvalues.cpp`)
   - Power iteration method for finding dominant eigenvalues
   - Analytical solution for 2x2 matrices
   - Verification of eigenvectors
   - Handling of complex eigenvalues

## Relevance to Optimization

### Vector Operations
Vector operations are essential for:
- Representing variables and gradients in optimization problems
- Computing distances and directions in search algorithms
- Implementing vector calculus operations for gradient-based methods
- Projecting solutions onto constraint sets

### Matrix Operations
Matrix operations are fundamental to:
- Representing systems of linear equations and constraints
- Implementing linear and quadratic programming algorithms
- Computing Hessian matrices for second-order optimization methods
- Solving least squares problems

### Linear Transformations
Linear transformations are important for:
- Changing coordinate systems to simplify optimization problems
- Preconditioning to improve convergence of iterative methods
- Implementing dimensionality reduction techniques
- Understanding geometric interpretations of optimization algorithms

### Eigenvalues and Eigenvectors
Eigenvalues and eigenvectors are crucial for:
- Analyzing the convergence of iterative optimization methods
- Determining the convexity of objective functions
- Principal component analysis for dimensionality reduction
- Understanding the behavior of dynamical systems in optimization

## Connection to OR-Tools

These linear algebra implementations provide the mathematical foundation for several features in Google's OR-Tools:

1. **Linear Programming Solvers**:
   - Matrix representations of constraints and objective functions
   - Basis transformations in simplex methods
   - Preconditioning for interior point methods

2. **Quadratic Programming**:
   - Hessian matrix operations
   - Eigenvalue analysis for convexity determination

3. **Network Flow Algorithms**:
   - Incidence matrices and graph representations
   - Matrix operations for min-cost flow and max-flow algorithms

4. **Constraint Programming**:
   - Linear constraint representations
   - Transformation of variable domains

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
./cpp_implementations/linear_algebra/vector_operations
./cpp_implementations/linear_algebra/matrix_operations
./cpp_implementations/linear_algebra/linear_transformations
./cpp_implementations/linear_algebra/eigenvalues
```

## Further Reading

- "Linear Algebra and Its Applications" by Gilbert Strang
- "Numerical Linear Algebra" by Lloyd N. Trefethen and David Bau III
- "Matrix Computations" by Gene H. Golub and Charles F. Van Loan
- "Convex Optimization" by Stephen Boyd and Lieven Vandenberghe
