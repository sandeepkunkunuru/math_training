#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <iomanip>
#include <algorithm>

/**
 * Lagrangian Methods Implementation
 * 
 * This file demonstrates Lagrangian methods for constrained optimization
 * in C++, which are essential for optimization mathematics.
 */

using Vector = std::vector<double>;
using Function = std::function<double(const Vector&)>;
using Constraint = std::function<double(const Vector&)>;

// Function to print a vector
void print_vector(const Vector& v, const std::string& name = "Vector") {
    std::cout << name << ": [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Compute the gradient of a function at a point
Vector compute_gradient(const Function& f, const Vector& x, double h = 1e-6) {
    Vector grad(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        Vector x_plus_h = x;
        Vector x_minus_h = x;
        
        x_plus_h[i] += h;
        x_minus_h[i] -= h;
        
        // Central difference formula for partial derivative
        grad[i] = (f(x_plus_h) - f(x_minus_h)) / (2 * h);
    }
    
    return grad;
}

// Compute the dot product of two vectors
double dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimensions must match for dot product");
    }
    
    double result = 0.0;
    for (size_t i = 0; i < v1.size(); ++i) {
        result += v1[i] * v2[i];
    }
    return result;
}

// Compute the norm (magnitude) of a vector
double vector_norm(const Vector& v) {
    return std::sqrt(dot_product(v, v));
}

// Lagrangian function: L(x, λ) = f(x) - λ * g(x)
// For equality constraint g(x) = 0
double lagrangian(const Function& f, const Constraint& g, const Vector& x, double lambda) {
    return f(x) - lambda * g(x);
}

// Augmented Lagrangian function: L_A(x, λ) = f(x) - λ * g(x) + (μ/2) * g(x)^2
// For equality constraint g(x) = 0
double augmented_lagrangian(const Function& f, const Constraint& g, 
                           const Vector& x, double lambda, double mu) {
    double g_x = g(x);
    return f(x) - lambda * g_x + (mu / 2.0) * g_x * g_x;
}

// Method of Lagrange multipliers for equality constrained optimization
// Solves min f(x) subject to g(x) = 0
std::pair<Vector, double> lagrange_multipliers(
    const Function& f, const Constraint& g,
    Vector x0, double lambda0 = 0.0,
    double learning_rate = 0.01, double tolerance = 1e-6, int max_iterations = 1000) {
    
    Vector x = x0;
    double lambda = lambda0;
    
    std::cout << "Method of Lagrange Multipliers:\n";
    std::cout << "Iteration 0: x = ";
    print_vector(x, "");
    std::cout << "λ = " << lambda << ", f(x) = " << f(x) << ", g(x) = " << g(x) << "\n";
    
    for (int iter = 1; iter <= max_iterations; ++iter) {
        // Compute gradients
        Vector grad_f = compute_gradient(f, x);
        Vector grad_g = compute_gradient(g, x);
        
        // Compute Lagrangian gradient: ∇L = ∇f - λ * ∇g
        Vector grad_L(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            grad_L[i] = grad_f[i] - lambda * grad_g[i];
        }
        
        // Update x: x = x - learning_rate * ∇L
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] -= learning_rate * grad_L[i];
        }
        
        // Update λ: λ = λ + learning_rate * g(x)
        // This moves λ to enforce the constraint g(x) = 0
        lambda += learning_rate * g(x);
        
        // Print progress every 10 iterations
        if (iter % 10 == 0 || iter == max_iterations) {
            double constraint_violation = std::abs(g(x));
            std::cout << "Iteration " << iter << ": x = ";
            print_vector(x, "");
            std::cout << "λ = " << lambda << ", f(x) = " << f(x) 
                      << ", |g(x)| = " << constraint_violation << "\n";
            
            // Check for convergence
            if (vector_norm(grad_L) < tolerance && constraint_violation < tolerance) {
                std::cout << "Converged after " << iter << " iterations.\n";
                break;
            }
        }
    }
    
    return {x, lambda};
}

// Augmented Lagrangian method for equality constrained optimization
// Solves min f(x) subject to g(x) = 0
std::pair<Vector, double> augmented_lagrangian_method(
    const Function& f, const Constraint& g,
    Vector x0, double lambda0 = 0.0, double mu0 = 10.0,
    double mu_factor = 10.0, double tolerance = 1e-6, int max_iterations = 100) {
    
    Vector x = x0;
    double lambda = lambda0;
    double mu = mu0;
    
    std::cout << "Augmented Lagrangian Method:\n";
    std::cout << "Iteration 0: x = ";
    print_vector(x, "");
    std::cout << "λ = " << lambda << ", μ = " << mu 
              << ", f(x) = " << f(x) << ", g(x) = " << g(x) << "\n";
    
    for (int outer_iter = 1; outer_iter <= max_iterations; ++outer_iter) {
        // Define the augmented Lagrangian function for the current λ and μ
        auto L_A = [&](const Vector& x_val) {
            return augmented_lagrangian(f, g, x_val, lambda, mu);
        };
        
        // Minimize the augmented Lagrangian with respect to x
        // (This is an unconstrained optimization problem)
        for (int inner_iter = 1; inner_iter <= 50; ++inner_iter) {
            Vector grad_L_A = compute_gradient(L_A, x);
            
            // Update x using gradient descent
            for (size_t i = 0; i < x.size(); ++i) {
                x[i] -= 0.01 / mu * grad_L_A[i];  // Smaller step size for larger μ
            }
            
            if (vector_norm(grad_L_A) < tolerance) {
                break;
            }
        }
        
        // Compute constraint violation
        double constraint_violation = std::abs(g(x));
        
        std::cout << "Iteration " << outer_iter << ": x = ";
        print_vector(x, "");
        std::cout << "λ = " << lambda << ", μ = " << mu 
                  << ", f(x) = " << f(x) << ", |g(x)| = " << constraint_violation << "\n";
        
        // Check for convergence
        if (constraint_violation < tolerance) {
            std::cout << "Converged after " << outer_iter << " iterations.\n";
            break;
        }
        
        // Update Lagrange multiplier: λ = λ - μ * g(x)
        lambda = lambda - mu * g(x);
        
        // Increase penalty parameter
        mu *= mu_factor;
    }
    
    return {x, lambda};
}

// KKT conditions check for equality constrained optimization
bool check_kkt_conditions(const Function& f, const Constraint& g, 
                         const Vector& x, double lambda, double tolerance = 1e-6) {
    Vector grad_f = compute_gradient(f, x);
    Vector grad_g = compute_gradient(g, x);
    
    // Compute Lagrangian gradient: ∇L = ∇f - λ * ∇g
    Vector grad_L(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        grad_L[i] = grad_f[i] - lambda * grad_g[i];
    }
    
    // Check stationarity condition: ∇f(x) - λ * ∇g(x) = 0
    bool stationarity = vector_norm(grad_L) < tolerance;
    
    // Check primal feasibility: g(x) = 0
    bool primal_feasibility = std::abs(g(x)) < tolerance;
    
    std::cout << "KKT Conditions Check:\n";
    std::cout << "Stationarity: " << (stationarity ? "Satisfied" : "Not satisfied") 
              << " (||∇L|| = " << vector_norm(grad_L) << ")\n";
    std::cout << "Primal feasibility: " << (primal_feasibility ? "Satisfied" : "Not satisfied") 
              << " (|g(x)| = " << std::abs(g(x)) << ")\n";
    
    return stationarity && primal_feasibility;
}

int main() {
    std::cout << "Lagrangian Methods for Constrained Optimization Examples\n";
    std::cout << "======================================================\n";
    
    // Example 1: Minimize f(x,y) = x^2 + y^2 subject to g(x,y) = x + y - 1 = 0
    std::cout << "\nExample 1: Minimize f(x,y) = x^2 + y^2 subject to x + y - 1 = 0\n";
    
    // Objective function: f(x,y) = x^2 + y^2
    auto f1 = [](const Vector& v) {
        double x = v[0];
        double y = v[1];
        return x*x + y*y;
    };
    
    // Constraint function: g(x,y) = x + y - 1 = 0
    auto g1 = [](const Vector& v) {
        double x = v[0];
        double y = v[1];
        return x + y - 1;
    };
    
    // Analytical solution: x = y = 0.5
    std::cout << "Analytical solution: x = y = 0.5, f(x,y) = 0.5\n\n";
    
    // Starting point
    Vector x0 = {0.0, 0.0};
    
    // Solve using method of Lagrange multipliers
    std::cout << "Solving using method of Lagrange multipliers:\n";
    auto [x_lagrange, lambda_lagrange] = lagrange_multipliers(f1, g1, x0);
    
    std::cout << "\nLagrange multipliers result:\n";
    print_vector(x_lagrange, "x");
    std::cout << "λ = " << lambda_lagrange << "\n";
    std::cout << "f(x) = " << f1(x_lagrange) << "\n";
    std::cout << "g(x) = " << g1(x_lagrange) << "\n";
    
    // Check KKT conditions
    std::cout << "\nChecking KKT conditions for Lagrange multipliers result:\n";
    check_kkt_conditions(f1, g1, x_lagrange, lambda_lagrange);
    
    // Solve using augmented Lagrangian method
    std::cout << "\nSolving using augmented Lagrangian method:\n";
    auto [x_augmented, lambda_augmented] = augmented_lagrangian_method(f1, g1, x0);
    
    std::cout << "\nAugmented Lagrangian result:\n";
    print_vector(x_augmented, "x");
    std::cout << "λ = " << lambda_augmented << "\n";
    std::cout << "f(x) = " << f1(x_augmented) << "\n";
    std::cout << "g(x) = " << g1(x_augmented) << "\n";
    
    // Check KKT conditions
    std::cout << "\nChecking KKT conditions for augmented Lagrangian result:\n";
    check_kkt_conditions(f1, g1, x_augmented, lambda_augmented);
    
    // Example 2: A more complex problem
    std::cout << "\nExample 2: Minimize f(x,y) = (x-2)^2 + (y-1)^2 subject to x^2 + y^2 - 1 = 0\n";
    
    // Objective function: f(x,y) = (x-2)^2 + (y-1)^2
    auto f2 = [](const Vector& v) {
        double x = v[0];
        double y = v[1];
        return std::pow(x-2, 2) + std::pow(y-1, 2);
    };
    
    // Constraint function: g(x,y) = x^2 + y^2 - 1 = 0
    auto g2 = [](const Vector& v) {
        double x = v[0];
        double y = v[1];
        return x*x + y*y - 1;
    };
    
    // Starting point
    Vector x0_2 = {0.7, 0.7};
    
    // Solve using augmented Lagrangian method
    std::cout << "\nSolving using augmented Lagrangian method:\n";
    auto [x_augmented_2, lambda_augmented_2] = augmented_lagrangian_method(f2, g2, x0_2);
    
    std::cout << "\nAugmented Lagrangian result:\n";
    print_vector(x_augmented_2, "x");
    std::cout << "λ = " << lambda_augmented_2 << "\n";
    std::cout << "f(x) = " << f2(x_augmented_2) << "\n";
    std::cout << "g(x) = " << g2(x_augmented_2) << "\n";
    
    // Check KKT conditions
    std::cout << "\nChecking KKT conditions for augmented Lagrangian result:\n";
    check_kkt_conditions(f2, g2, x_augmented_2, lambda_augmented_2);
    
    return 0;
}
