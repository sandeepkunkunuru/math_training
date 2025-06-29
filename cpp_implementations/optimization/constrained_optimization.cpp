#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <algorithm>

/**
 * Constrained Optimization Implementation
 * 
 * This file demonstrates various constrained optimization algorithms
 * in C++ that are essential for solving optimization problems with constraints.
 */

// Type definitions for clarity
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;
using Function = std::function<double(const Vector&)>;
using VectorFunction = std::function<Vector(const Vector&)>;
using Constraint = std::function<double(const Vector&)>;
using Constraints = std::vector<Constraint>;

// Print a vector
void print_vector(const Vector& v, const std::string& name = "Vector") {
    std::cout << name << ": [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
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

// Compute constraint violation
double constraint_violation(const Vector& x, const Constraints& constraints) {
    double total_violation = 0.0;
    
    for (const auto& constraint : constraints) {
        double c_val = constraint(x);
        if (c_val > 0) {
            total_violation += c_val;
        }
    }
    
    return total_violation;
}

// Penalty method for constrained optimization
Vector penalty_method(const Function& objective, const Constraints& constraints, 
                     const Vector& initial_point, int max_iterations = 100, 
                     double initial_penalty = 1.0, double penalty_increase_factor = 10.0,
                     double tolerance = 1e-6) {
    Vector x = initial_point;
    double penalty = initial_penalty;
    
    std::cout << "Starting Penalty Method Optimization" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    
    // Define the penalized objective function
    auto penalized_objective = [&](const Vector& x) {
        double f_val = objective(x);
        double penalty_term = 0.0;
        
        for (const auto& constraint : constraints) {
            double c_val = constraint(x);
            if (c_val > 0) {
                penalty_term += c_val * c_val;
            }
        }
        
        return f_val + penalty * penalty_term;
    };
    
    // Unconstrained minimization using gradient descent
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute function value and constraint violation
        double f_x = objective(x);
        double violation = constraint_violation(x, constraints);
        
        // Print iteration information
        print_iteration(iter, x, f_x, violation);
        
        // Check convergence
        if (violation < tolerance) {
            std::cout << "Converged after " << iter << " iterations. Constraint violation: " 
                      << violation << std::endl;
            break;
        }
        
        // Compute gradient of penalized objective
        Vector gradient = numerical_gradient(penalized_objective, x);
        
        // Compute step size using backtracking line search
        double step_size = 1.0;
        double alpha = 0.3;
        double beta = 0.8;
        
        // Compute directional derivative
        double directional_derivative = 0.0;
        for (size_t i = 0; i < gradient.size(); ++i) {
            directional_derivative -= gradient[i] * gradient[i];
        }
        
        // Create new point: x_new = x - step_size * gradient
        Vector x_new(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            x_new[i] = x[i] - step_size * gradient[i];
        }
        
        // Backtracking line search
        double f_x_penalized = penalized_objective(x);
        while (penalized_objective(x_new) > f_x_penalized + alpha * step_size * directional_derivative) {
            step_size *= beta;
            
            // Update x_new with new step size
            for (size_t i = 0; i < x.size(); ++i) {
                x_new[i] = x[i] - step_size * gradient[i];
            }
        }
        
        // Update x
        x = x_new;
        
        // Increase penalty parameter periodically
        if (iter > 0 && iter % 10 == 0) {
            penalty *= penalty_increase_factor;
            std::cout << "Increased penalty to " << penalty << std::endl;
        }
    }
    
    return x;
}

// Augmented Lagrangian method for constrained optimization
Vector augmented_lagrangian(const Function& objective, const Constraints& constraints, 
                           const Vector& initial_point, int max_iterations = 100, 
                           double initial_penalty = 1.0, double penalty_increase_factor = 10.0,
                           double tolerance = 1e-6) {
    Vector x = initial_point;
    double penalty = initial_penalty;
    
    // Initialize Lagrange multipliers
    Vector lambda(constraints.size(), 0.0);
    
    std::cout << "Starting Augmented Lagrangian Method Optimization" << std::endl;
    std::cout << "-----------------------------------------------" << std::endl;
    
    // Define the augmented Lagrangian function
    auto augmented_lagrangian_func = [&](const Vector& x) {
        double f_val = objective(x);
        double penalty_term = 0.0;
        
        for (size_t i = 0; i < constraints.size(); ++i) {
            double c_val = constraints[i](x);
            double augmented_term = lambda[i] * c_val + (penalty / 2.0) * c_val * c_val;
            penalty_term += augmented_term;
        }
        
        return f_val + penalty_term;
    };
    
    // Unconstrained minimization using gradient descent
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute function value and constraint violation
        double f_x = objective(x);
        double violation = constraint_violation(x, constraints);
        
        // Print iteration information
        print_iteration(iter, x, f_x, violation);
        
        // Check convergence
        if (violation < tolerance) {
            std::cout << "Converged after " << iter << " iterations. Constraint violation: " 
                      << violation << std::endl;
            break;
        }
        
        // Compute gradient of augmented Lagrangian
        Vector gradient = numerical_gradient(augmented_lagrangian_func, x);
        
        // Compute step size using backtracking line search
        double step_size = 1.0;
        double alpha = 0.3;
        double beta = 0.8;
        
        // Compute directional derivative
        double directional_derivative = 0.0;
        for (size_t i = 0; i < gradient.size(); ++i) {
            directional_derivative -= gradient[i] * gradient[i];
        }
        
        // Create new point: x_new = x - step_size * gradient
        Vector x_new(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            x_new[i] = x[i] - step_size * gradient[i];
        }
        
        // Backtracking line search
        double f_x_augmented = augmented_lagrangian_func(x);
        while (augmented_lagrangian_func(x_new) > f_x_augmented + alpha * step_size * directional_derivative) {
            step_size *= beta;
            
            // Update x_new with new step size
            for (size_t i = 0; i < x.size(); ++i) {
                x_new[i] = x[i] - step_size * gradient[i];
            }
        }
        
        // Update x
        x = x_new;
        
        // Update Lagrange multipliers
        for (size_t i = 0; i < constraints.size(); ++i) {
            double c_val = constraints[i](x);
            lambda[i] += penalty * c_val;
        }
        
        // Increase penalty parameter periodically
        if (iter > 0 && iter % 5 == 0) {
            penalty *= penalty_increase_factor;
            std::cout << "Increased penalty to " << penalty << std::endl;
        }
    }
    
    return x;
}

// Projected gradient method for box-constrained optimization
Vector projected_gradient(const Function& objective, const Vector& lower_bounds, 
                         const Vector& upper_bounds, const Vector& initial_point, 
                         int max_iterations = 100, double tolerance = 1e-6) {
    Vector x = initial_point;
    int n = x.size();
    
    // Ensure initial point is within bounds
    for (int i = 0; i < n; ++i) {
        x[i] = std::max(lower_bounds[i], std::min(x[i], upper_bounds[i]));
    }
    
    std::cout << "Starting Projected Gradient Method Optimization" << std::endl;
    std::cout << "--------------------------------------------" << std::endl;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute function value
        double f_x = objective(x);
        
        // Compute gradient
        Vector gradient = numerical_gradient(objective, x);
        
        // Compute projected gradient
        Vector proj_gradient(n);
        for (int i = 0; i < n; ++i) {
            if ((x[i] == lower_bounds[i] && gradient[i] > 0) || 
                (x[i] == upper_bounds[i] && gradient[i] < 0)) {
                proj_gradient[i] = 0.0;
            } else {
                proj_gradient[i] = gradient[i];
            }
        }
        
        // Check convergence
        double proj_gradient_norm = vector_norm(proj_gradient);
        if (proj_gradient_norm < tolerance) {
            std::cout << "Converged after " << iter << " iterations. Projected gradient norm: " 
                      << proj_gradient_norm << std::endl;
            break;
        }
        
        // Compute step size using backtracking line search
        double step_size = 1.0;
        double alpha = 0.3;
        double beta = 0.8;
        
        // Create new point: x_new = P[x - step_size * gradient]
        Vector x_new(n);
        for (int i = 0; i < n; ++i) {
            x_new[i] = x[i] - step_size * gradient[i];
            // Project onto feasible set
            x_new[i] = std::max(lower_bounds[i], std::min(x_new[i], upper_bounds[i]));
        }
        
        // Backtracking line search
        while (objective(x_new) > f_x - alpha * step_size * proj_gradient_norm * proj_gradient_norm) {
            step_size *= beta;
            
            // Update x_new with new step size
            for (int i = 0; i < n; ++i) {
                x_new[i] = x[i] - step_size * gradient[i];
                // Project onto feasible set
                x_new[i] = std::max(lower_bounds[i], std::min(x_new[i], upper_bounds[i]));
            }
        }
        
        // Print iteration information
        double constraint_viol = 0.0;  // No constraint violation for projected gradient
        print_iteration(iter, x, f_x, constraint_viol);
        
        // Update x
        x = x_new;
    }
    
    return x;
}

// Test functions and constraints for constrained optimization

// Example 1: Minimize f(x,y) = (x-2)^2 + (y-1)^2 subject to x^2 + y^2 <= 1
double circle_objective(const Vector& x) {
    return std::pow(x[0] - 2.0, 2) + std::pow(x[1] - 1.0, 2);
}

double circle_constraint(const Vector& x) {
    return x[0] * x[0] + x[1] * x[1] - 1.0;  // g(x) <= 0 form
}

// Example 2: Minimize f(x,y) = x^2 + y^2 subject to x + y >= 1
double simple_objective(const Vector& x) {
    return x[0] * x[0] + x[1] * x[1];
}

double simple_constraint(const Vector& x) {
    return 1.0 - x[0] - x[1];  // g(x) <= 0 form
}

// Example 3: Rosenbrock function with box constraints
double rosenbrock(const Vector& x) {
    return std::pow(1.0 - x[0], 2) + 100.0 * std::pow(x[1] - x[0] * x[0], 2);
}

int main() {
    std::cout << "Constrained Optimization Examples" << std::endl;
    std::cout << "===============================" << std::endl;
    
    // Example 1: Circle constraint with penalty method
    std::cout << "\nExample 1: Circle constraint with penalty method" << std::endl;
    Vector initial_point1 = {0.0, 0.0};
    Constraints constraints1 = {circle_constraint};
    Vector result1 = penalty_method(circle_objective, constraints1, initial_point1);
    std::cout << "Final result: " << std::endl;
    print_vector(result1, "x*");
    std::cout << "f(x*) = " << circle_objective(result1) << std::endl;
    std::cout << "Constraint value: " << circle_constraint(result1) << std::endl;
    
    // Example 2: Simple constraint with augmented Lagrangian
    std::cout << "\nExample 2: Simple constraint with augmented Lagrangian" << std::endl;
    Vector initial_point2 = {0.0, 0.0};
    Constraints constraints2 = {simple_constraint};
    Vector result2 = augmented_lagrangian(simple_objective, constraints2, initial_point2);
    std::cout << "Final result: " << std::endl;
    print_vector(result2, "x*");
    std::cout << "f(x*) = " << simple_objective(result2) << std::endl;
    std::cout << "Constraint value: " << simple_constraint(result2) << std::endl;
    
    // Example 3: Rosenbrock with box constraints
    std::cout << "\nExample 3: Rosenbrock with box constraints" << std::endl;
    Vector initial_point3 = {0.0, 0.0};
    Vector lower_bounds = {-0.5, -0.5};
    Vector upper_bounds = {0.5, 0.5};
    Vector result3 = projected_gradient(rosenbrock, lower_bounds, upper_bounds, initial_point3);
    std::cout << "Final result: " << std::endl;
    print_vector(result3, "x*");
    std::cout << "f(x*) = " << rosenbrock(result3) << std::endl;
    
    return 0;
}
