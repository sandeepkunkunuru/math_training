#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <algorithm>

/**
 * Nonlinear Programming Implementation
 * 
 * This file demonstrates various nonlinear programming algorithms
 * including Sequential Quadratic Programming (SQP) and Interior Point methods.
 */

// Type definitions for clarity
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;
using Function = std::function<double(const Vector&)>;
using VectorFunction = std::function<Vector(const Vector&)>;
using Constraint = std::function<double(const Vector&)>;
using Constraints = std::vector<Constraint>;
using GradientFunction = std::function<Vector(const Vector&)>;
using JacobianFunction = std::function<Matrix(const Vector&)>;
using HessianFunction = std::function<Matrix(const Vector&)>;

// Print a vector
void print_vector(const Vector& v, const std::string& name = "Vector") {
    std::cout << name << ": [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Print a matrix
void print_matrix(const Matrix& m, const std::string& name = "Matrix") {
    std::cout << name << ":" << std::endl;
    for (size_t i = 0; i < m.size(); ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < m[i].size(); ++j) {
            std::cout << std::fixed << std::setprecision(6) << m[i][j];
            if (j < m[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

// Print iteration information
void print_iteration(int iter, const Vector& x, double f_x, double constraint_violation) {
    std::cout << "Iteration " << std::setw(3) << iter 
              << " | f(x) = " << std::setw(10) << std::fixed << std::setprecision(6) << f_x
              << " | constraint violation = " << std::setw(10) << std::fixed << std::setprecision(6) << constraint_violation
              << " | x = [";
    
    for (size_t i = 0; i < x.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << x[i];
        if (i < x.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Compute vector norm (magnitude)
double vector_norm(const Vector& v) {
    double sum_sq = 0.0;
    for (double val : v) {
        sum_sq += val * val;
    }
    return std::sqrt(sum_sq);
}

// Compute numerical gradient of a function at point x
Vector numerical_gradient(const Function& f, const Vector& x, double h = 1e-6) {
    Vector grad(x.size());
    Vector x_plus_h = x;
    Vector x_minus_h = x;
    
    for (size_t i = 0; i < x.size(); ++i) {
        // Forward difference
        x_plus_h[i] = x[i] + h;
        double f_plus = f(x_plus_h);
        x_plus_h[i] = x[i];
        
        // Backward difference
        x_minus_h[i] = x[i] - h;
        double f_minus = f(x_minus_h);
        x_minus_h[i] = x[i];
        
        // Central difference
        grad[i] = (f_plus - f_minus) / (2 * h);
    }
    
    return grad;
}

// Compute numerical Jacobian of a vector function at point x
Matrix numerical_jacobian(const VectorFunction& f, const Vector& x, double h = 1e-6) {
    Vector f_x = f(x);
    size_t m = f_x.size();
    size_t n = x.size();
    
    Matrix jacobian(m, Vector(n));
    Vector x_plus_h = x;
    
    for (size_t j = 0; j < n; ++j) {
        // Forward difference
        x_plus_h[j] = x[j] + h;
        Vector f_plus = f(x_plus_h);
        x_plus_h[j] = x[j];
        
        for (size_t i = 0; i < m; ++i) {
            jacobian[i][j] = (f_plus[i] - f_x[i]) / h;
        }
    }
    
    return jacobian;
}

// Compute numerical Hessian of a function at point x
Matrix numerical_hessian(const Function& f, const Vector& x, double h = 1e-4) {
    int n = x.size();
    Matrix hessian(n, Vector(n, 0.0));
    Vector x_pp = x, x_pm = x, x_mp = x, x_mm = x;
    
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j <= i; ++j) {  // Exploit symmetry
            if (i == j) {  // Diagonal elements
                // Use central difference for second derivative
                x_pp[i] = x[i] + h;
                x_mm[i] = x[i] - h;
                double f_pp = f(x_pp);
                double f_mm = f(x_mm);
                double f_center = f(x);
                
                // Second derivative approximation: (f(x+h) - 2f(x) + f(x-h)) / h^2
                hessian[i][j] = (f_pp - 2 * f_center + f_mm) / (h * h);
                
                // Reset
                x_pp[i] = x[i];
                x_mm[i] = x[i];
            } else {  // Off-diagonal elements
                // Use mixed partial derivative approximation
                x_pp[i] = x[i] + h;
                x_pp[j] = x[j] + h;
                double f_pp = f(x_pp);
                
                x_pm[i] = x[i] + h;
                x_pm[j] = x[j] - h;
                double f_pm = f(x_pm);
                
                x_mp[i] = x[i] - h;
                x_mp[j] = x[j] + h;
                double f_mp = f(x_mp);
                
                x_mm[i] = x[i] - h;
                x_mm[j] = x[j] - h;
                double f_mm = f(x_mm);
                
                // Mixed partial derivative: (f(x+h,y+h) - f(x+h,y-h) - f(x-h,y+h) + f(x-h,y-h)) / (4h^2)
                hessian[i][j] = (f_pp - f_pm - f_mp + f_mm) / (4 * h * h);
                hessian[j][i] = hessian[i][j];  // Symmetric
                
                // Reset
                x_pp[i] = x[i]; x_pp[j] = x[j];
                x_pm[i] = x[i]; x_pm[j] = x[j];
                x_mp[i] = x[i]; x_mp[j] = x[j];
                x_mm[i] = x[i]; x_mm[j] = x[j];
            }
        }
    }
    
    return hessian;
}

// Compute constraint violation
double constraint_violation(const Vector& x, const Constraints& equality_constraints, 
                          const Constraints& inequality_constraints) {
    double total_violation = 0.0;
    
    // Equality constraints: c(x) = 0
    for (const auto& constraint : equality_constraints) {
        double c_val = constraint(x);
        total_violation += std::abs(c_val);
    }
    
    // Inequality constraints: c(x) <= 0
    for (const auto& constraint : inequality_constraints) {
        double c_val = constraint(x);
        if (c_val > 0) {
            total_violation += c_val;
        }
    }
    
    return total_violation;
}

// Solve a system of linear equations Ax = b using Gaussian elimination with partial pivoting
Vector solve_linear_system(const Matrix& A, const Vector& b) {
    int n = A.size();
    
    // Create augmented matrix [A|b]
    Matrix augmented(n, Vector(n + 1));
    for (int i = 0; i < n; ++i) {
        for (int j = 0; j < n; ++j) {
            augmented[i][j] = A[i][j];
        }
        augmented[i][n] = b[i];
    }
    
    // Gaussian elimination with partial pivoting
    for (int i = 0; i < n; ++i) {
        // Find pivot
        int max_row = i;
        double max_val = std::abs(augmented[i][i]);
        
        for (int k = i + 1; k < n; ++k) {
            if (std::abs(augmented[k][i]) > max_val) {
                max_val = std::abs(augmented[k][i]);
                max_row = k;
            }
        }
        
        // Swap rows if needed
        if (max_row != i) {
            std::swap(augmented[i], augmented[max_row]);
        }
        
        // Check for singularity
        if (std::abs(augmented[i][i]) < 1e-10) {
            // Add small regularization to diagonal
            augmented[i][i] += 1e-10;
        }
        
        // Eliminate below
        for (int k = i + 1; k < n; ++k) {
            double factor = augmented[k][i] / augmented[i][i];
            
            for (int j = i; j <= n; ++j) {
                augmented[k][j] -= factor * augmented[i][j];
            }
        }
    }
    
    // Back substitution
    Vector x(n);
    for (int i = n - 1; i >= 0; --i) {
        x[i] = augmented[i][n];
        
        for (int j = i + 1; j < n; ++j) {
            x[i] -= augmented[i][j] * x[j];
        }
        
        x[i] /= augmented[i][i];
    }
    
    return x;
}

// Type definition for constraints
using Constraint = std::function<double(const Vector&)>;
using Constraints = std::vector<Constraint>;

// Check Karush-Kuhn-Tucker (KKT) conditions
bool check_kkt_conditions(const Function& objective, 
                        const Vector& x,
                        const Constraints& equality_constraints,
                        const Constraints& inequality_constraints,
                        double tolerance = 1e-4) {
    // Compute gradient of objective
    Vector grad_f = numerical_gradient(objective, x);
    
    // Compute Jacobians of constraints
    std::vector<Vector> eq_constraint_grads;
    for (const auto& constraint : equality_constraints) {
        Function c_func = [&constraint](const Vector& x) { return constraint(x); };
        eq_constraint_grads.push_back(numerical_gradient(c_func, x));
    }
    
    std::vector<Vector> ineq_constraint_grads;
    std::vector<double> ineq_constraint_values;
    for (const auto& constraint : inequality_constraints) {
        Function c_func = [&constraint](const Vector& x) { return constraint(x); };
        ineq_constraint_grads.push_back(numerical_gradient(c_func, x));
        ineq_constraint_values.push_back(constraint(x));
    }
    
    // Check constraint satisfaction
    for (const auto& constraint : equality_constraints) {
        if (std::abs(constraint(x)) > tolerance) {
            std::cout << "Equality constraint violation: " << constraint(x) << std::endl;
            return false;
        }
    }
    
    for (size_t i = 0; i < inequality_constraints.size(); ++i) {
        if (ineq_constraint_values[i] > tolerance) {
            std::cout << "Inequality constraint violation: " << ineq_constraint_values[i] << std::endl;
            return false;
        }
    }
    
    // Solve for Lagrange multipliers (simplified approach)
    // This is a least-squares approximation to find multipliers
    // that satisfy the KKT stationarity condition
    
    // Count active inequality constraints
    int active_ineq = 0;
    for (size_t i = 0; i < ineq_constraint_values.size(); ++i) {
        if (std::abs(ineq_constraint_values[i]) < tolerance) {
            active_ineq++;
        }
    }
    
    // If no constraints are active, gradient should be approximately zero
    if (equality_constraints.empty() && active_ineq == 0) {
        double grad_norm = vector_norm(grad_f);
        if (grad_norm > tolerance) {
            std::cout << "Gradient norm too large: " << grad_norm << std::endl;
            return false;
        }
        return true;
    }
    
    // Set up least-squares problem to find multipliers
    int n_eq = equality_constraints.size();
    int n_active = n_eq + active_ineq;
    
    if (n_active == 0) {
        // No active constraints, just check gradient
        double grad_norm = vector_norm(grad_f);
        return grad_norm <= tolerance;
    }
    
    // Create matrix of constraint gradients
    Matrix A(n_active, std::vector<double>(x.size()));
    
    // Fill with equality constraint gradients
    for (int i = 0; i < n_eq; ++i) {
        for (size_t j = 0; j < x.size(); ++j) {
            A[i][j] = eq_constraint_grads[i][j];
        }
    }
    
    // Fill with active inequality constraint gradients
    int row = n_eq;
    for (size_t i = 0; i < ineq_constraint_values.size(); ++i) {
        if (std::abs(ineq_constraint_values[i]) < tolerance) {
            for (size_t j = 0; j < x.size(); ++j) {
                A[row][j] = ineq_constraint_grads[i][j];
            }
            row++;
        }
    }
    
    // Compute A^T * A and A^T * grad_f
    Matrix ATA(x.size(), std::vector<double>(x.size(), 0.0));
    Vector ATb(x.size(), 0.0);
    
    for (size_t i = 0; i < x.size(); ++i) {
        for (size_t j = 0; j < x.size(); ++j) {
            for (int k = 0; k < n_active; ++k) {
                ATA[i][j] += A[k][i] * A[k][j];
            }
        }
        for (int k = 0; k < n_active; ++k) {
            ATb[i] += A[k][i] * grad_f[k];
        }
    }
    
    // Solve ATA * lambda = ATb
    Vector lambda;
    try {
        lambda = solve_linear_system(ATA, ATb);
    } catch (const std::runtime_error& e) {
        std::cout << "Failed to solve for Lagrange multipliers: " << e.what() << std::endl;
        return false;
    }
    
    // Check if all inequality constraint multipliers are non-negative
    for (int i = n_eq; i < n_active; ++i) {
        if (lambda[i] < -tolerance) {
            std::cout << "Negative Lagrange multiplier for inequality constraint: " << lambda[i] << std::endl;
            return false;
        }
    }
    
    // Check stationarity condition: grad_f + sum(lambda_i * grad_c_i) ≈ 0
    Vector stationarity(x.size(), 0.0);
    for (size_t i = 0; i < x.size(); ++i) {
        stationarity[i] = grad_f[i];
        for (int j = 0; j < n_active; ++j) {
            stationarity[i] += lambda[j] * A[j][i];
        }
    }
    
    double stationarity_norm = vector_norm(stationarity);
    if (stationarity_norm > tolerance) {
        std::cout << "Stationarity condition violated, norm = " << stationarity_norm << std::endl;
        return false;
    }
    
    return true;
}

// Solve a quadratic programming (QP) problem using active set method
// Minimize: (1/2) * x^T * G * x + c^T * x
// Subject to: A * x <= b
Vector solve_qp(const Matrix& G, const Vector& c, const Matrix& A, const Vector& b, 
              const Vector& initial_point, int max_iterations = 100) {
    int n = G.size();    // Number of variables
    int m = A.size();    // Number of constraints
    
    Vector x = initial_point;
    std::vector<bool> active_set(m, false);
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Identify active constraints
        for (int i = 0; i < m; ++i) {
            double constraint_value = 0.0;
            for (int j = 0; j < n; ++j) {
                constraint_value += A[i][j] * x[j];
            }
            
            // Check if constraint is active (with small tolerance)
            if (std::abs(constraint_value - b[i]) < 1e-6) {
                active_set[i] = true;
            }
        }
        
        // Count active constraints
        int num_active = 0;
        for (bool is_active : active_set) {
            if (is_active) num_active++;
        }
        
        // Extract active constraints
        Matrix A_active(num_active, Vector(n));
        Vector b_active(num_active);
        
        int idx = 0;
        for (int i = 0; i < m; ++i) {
            if (active_set[i]) {
                for (int j = 0; j < n; ++j) {
                    A_active[idx][j] = A[i][j];
                }
                b_active[idx] = b[i];
                idx++;
            }
        }
        
        // Solve the equality constrained QP
        // Compute gradient at current point: g = G*x + c
        Vector g(n);
        for (int i = 0; i < n; ++i) {
            g[i] = c[i];
            for (int j = 0; j < n; ++j) {
                g[i] += G[i][j] * x[j];
            }
        }
        
        // Solve the KKT system
        // [G  A^T] [p]   [-g]
        // [A   0 ] [λ] = [0 ]
        
        // For simplicity, we'll solve a simpler system for now
        // Compute the search direction p = -G^(-1) * g
        Vector p = solve_linear_system(G, Vector(n, 0.0));
        for (int i = 0; i < n; ++i) {
            p[i] = -p[i];
        }
        
        // Check if the current solution is optimal
        double p_norm = vector_norm(p);
        if (p_norm < 1e-6) {
            break;
        }
        
        // Compute step size using line search
        double step_size = 1.0;
        
        // Update solution
        for (int i = 0; i < n; ++i) {
            x[i] += step_size * p[i];
        }
    }
    
    return x;
}
