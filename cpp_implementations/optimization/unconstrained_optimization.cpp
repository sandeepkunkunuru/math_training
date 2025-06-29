#include <iostream>
#include <vector>
#include <cmath>
#include <functional>
#include <iomanip>
#include <limits>
#include <algorithm>

/**
 * Unconstrained Optimization Implementation
 * 
 * This file demonstrates various unconstrained optimization algorithms
 * in C++ that are essential for solving optimization problems.
 */

// Type definitions for clarity
using Vector = std::vector<double>;
using Function = std::function<double(const Vector&)>;
using GradientFunction = std::function<Vector(const Vector&)>;

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
void print_iteration(int iter, const Vector& x, double f_x, const Vector& gradient, double step_size) {
    std::cout << "Iteration " << std::setw(3) << iter 
              << " | f(x) = " << std::setw(10) << std::fixed << std::setprecision(6) << f_x
              << " | ||∇f|| = " << std::setw(10) << std::fixed << std::setprecision(6);
    
    // Compute gradient norm
    double grad_norm = 0.0;
    for (double g : gradient) {
        grad_norm += g * g;
    }
    grad_norm = std::sqrt(grad_norm);
    
    std::cout << grad_norm
              << " | step = " << std::setw(10) << std::fixed << std::setprecision(6) << step_size
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

// Line search to find step size using backtracking
double backtracking_line_search(const Function& f, const Vector& x, const Vector& gradient, 
                               double initial_step = 1.0, double alpha = 0.3, double beta = 0.8) {
    double step_size = initial_step;
    double f_x = f(x);
    
    // Compute directional derivative (dot product of gradient and negative gradient)
    double directional_derivative = 0.0;
    for (size_t i = 0; i < gradient.size(); ++i) {
        directional_derivative -= gradient[i] * gradient[i];  // -gradient · gradient
    }
    
    // Create new point: x_new = x - step_size * gradient
    Vector x_new(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        x_new[i] = x[i] - step_size * gradient[i];
    }
    
    // Backtracking line search
    while (f(x_new) > f_x + alpha * step_size * directional_derivative) {
        step_size *= beta;
        
        // Update x_new with new step size
        for (size_t i = 0; i < x.size(); ++i) {
            x_new[i] = x[i] - step_size * gradient[i];
        }
    }
    
    return step_size;
}

// Gradient Descent optimization
Vector gradient_descent(const Function& f, const Vector& initial_point, 
                       int max_iterations = 1000, double tolerance = 1e-6, 
                       bool use_line_search = true) {
    Vector x = initial_point;
    int n = x.size();
    
    std::cout << "Starting Gradient Descent Optimization" << std::endl;
    std::cout << "-------------------------------------" << std::endl;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute function value and gradient
        double f_x = f(x);
        Vector gradient = numerical_gradient(f, x);
        
        // Check convergence
        double gradient_norm = vector_norm(gradient);
        if (gradient_norm < tolerance) {
            std::cout << "Converged after " << iter << " iterations. Gradient norm: " 
                      << gradient_norm << std::endl;
            break;
        }
        
        // Determine step size
        double step_size;
        if (use_line_search) {
            step_size = backtracking_line_search(f, x, gradient);
        } else {
            step_size = 0.01;  // Fixed step size
        }
        
        // Print iteration information
        print_iteration(iter, x, f_x, gradient, step_size);
        
        // Update x: x = x - step_size * gradient
        for (int i = 0; i < n; ++i) {
            x[i] -= step_size * gradient[i];
        }
    }
    
    return x;
}

// Compute numerical Hessian of a function at point x
std::vector<std::vector<double>> numerical_hessian(const Function& f, const Vector& x, double h = 1e-4) {
    int n = x.size();
    std::vector<std::vector<double>> hessian(n, std::vector<double>(n, 0.0));
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

// Solve a system of linear equations Ax = b using Gaussian elimination
Vector solve_linear_system(const std::vector<std::vector<double>>& A, const Vector& b) {
    int n = A.size();
    
    // Create augmented matrix [A|b]
    std::vector<std::vector<double>> augmented(n, std::vector<double>(n + 1));
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

// Newton's Method optimization
Vector newtons_method(const Function& f, const Vector& initial_point, 
                     int max_iterations = 100, double tolerance = 1e-6) {
    Vector x = initial_point;
    int n = x.size();
    
    std::cout << "Starting Newton's Method Optimization" << std::endl;
    std::cout << "-----------------------------------" << std::endl;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Compute function value and gradient
        double f_x = f(x);
        Vector gradient = numerical_gradient(f, x);
        
        // Check convergence
        double gradient_norm = vector_norm(gradient);
        if (gradient_norm < tolerance) {
            std::cout << "Converged after " << iter << " iterations. Gradient norm: " 
                      << gradient_norm << std::endl;
            break;
        }
        
        // Compute Hessian
        std::vector<std::vector<double>> hessian = numerical_hessian(f, x);
        
        // Solve system: Hessian * direction = -gradient
        Vector neg_gradient(n);
        for (int i = 0; i < n; ++i) {
            neg_gradient[i] = -gradient[i];
        }
        
        Vector direction = solve_linear_system(hessian, neg_gradient);
        
        // Determine step size using line search
        double step_size = backtracking_line_search(f, x, direction, 1.0, 0.3, 0.5);
        
        // Print iteration information
        print_iteration(iter, x, f_x, gradient, step_size);
        
        // Update x: x = x + step_size * direction
        for (int i = 0; i < n; ++i) {
            x[i] += step_size * direction[i];
        }
    }
    
    return x;
}

// BFGS optimization method
Vector bfgs(const Function& f, const Vector& initial_point, 
           int max_iterations = 1000, double tolerance = 1e-6) {
    int n = initial_point.size();
    Vector x = initial_point;
    
    // Initialize approximate inverse Hessian to identity matrix
    std::vector<std::vector<double>> B_inv(n, std::vector<double>(n, 0.0));
    for (int i = 0; i < n; ++i) {
        B_inv[i][i] = 1.0;
    }
    
    // Initial gradient
    Vector gradient = numerical_gradient(f, x);
    
    std::cout << "Starting BFGS Optimization" << std::endl;
    std::cout << "------------------------" << std::endl;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        double f_x = f(x);
        
        // Check convergence
        double gradient_norm = vector_norm(gradient);
        if (gradient_norm < tolerance) {
            std::cout << "Converged after " << iter << " iterations. Gradient norm: " 
                      << gradient_norm << std::endl;
            break;
        }
        
        // Compute search direction: p = -B_inv * gradient
        Vector p(n, 0.0);
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                p[i] -= B_inv[i][j] * gradient[j];
            }
        }
        
        // Line search to find step size
        double step_size = backtracking_line_search(f, x, p, 1.0, 0.3, 0.8);
        
        // Print iteration information
        print_iteration(iter, x, f_x, gradient, step_size);
        
        // Store old x and gradient
        Vector x_old = x;
        Vector gradient_old = gradient;
        
        // Update x: x = x + step_size * p
        for (int i = 0; i < n; ++i) {
            x[i] += step_size * p[i];
        }
        
        // Compute new gradient
        gradient = numerical_gradient(f, x);
        
        // Compute s = x - x_old and y = gradient - gradient_old
        Vector s(n), y(n);
        for (int i = 0; i < n; ++i) {
            s[i] = x[i] - x_old[i];
            y[i] = gradient[i] - gradient_old[i];
        }
        
        // Compute rho = 1 / (y^T * s)
        double rho = 0.0;
        for (int i = 0; i < n; ++i) {
            rho += y[i] * s[i];
        }
        
        if (std::abs(rho) > 1e-10) {
            rho = 1.0 / rho;
            
            // BFGS update formula for inverse Hessian approximation
            // B_{k+1}^{-1} = (I - rho*s*y^T) * B_k^{-1} * (I - rho*y*s^T) + rho*s*s^T
            
            // Compute v = (I - rho*s*y^T) * B_k^{-1}
            std::vector<std::vector<double>> v(n, std::vector<double>(n));
            
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < n; ++k) {
                        sum += rho * s[i] * y[k] * B_inv[k][j];
                    }
                    v[i][j] = B_inv[i][j] - sum;
                }
            }
            
            // Compute B_{k+1}^{-1} = v * (I - rho*y*s^T) + rho*s*s^T
            std::vector<std::vector<double>> B_inv_new(n, std::vector<double>(n));
            
            for (int i = 0; i < n; ++i) {
                for (int j = 0; j < n; ++j) {
                    double sum = 0.0;
                    for (int k = 0; k < n; ++k) {
                        sum += v[i][k] * rho * y[k] * s[j];
                    }
                    B_inv_new[i][j] = v[i][j] - sum + rho * s[i] * s[j];
                }
            }
            
            B_inv = B_inv_new;
        }
    }
    
    return x;
}

// Test functions for optimization

// Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
double rosenbrock(const Vector& x) {
    return std::pow(1.0 - x[0], 2) + 100.0 * std::pow(x[1] - x[0] * x[0], 2);
}

// Himmelblau's function: f(x,y) = (x^2 + y - 11)^2 + (x + y^2 - 7)^2
double himmelblau(const Vector& x) {
    return std::pow(x[0] * x[0] + x[1] - 11.0, 2) + std::pow(x[0] + x[1] * x[1] - 7.0, 2);
}

// Sphere function: f(x) = sum(x_i^2)
double sphere(const Vector& x) {
    double sum = 0.0;
    for (double val : x) {
        sum += val * val;
    }
    return sum;
}

// Booth function: f(x,y) = (x + 2y - 7)^2 + (2x + y - 5)^2
double booth(const Vector& x) {
    return std::pow(x[0] + 2 * x[1] - 7, 2) + std::pow(2 * x[0] + x[1] - 5, 2);
}

int main() {
    std::cout << "Unconstrained Optimization Examples" << std::endl;
    std::cout << "================================" << std::endl;
    
    // Example 1: Minimize Rosenbrock function using Gradient Descent
    std::cout << "\nExample 1: Rosenbrock function with Gradient Descent" << std::endl;
    Vector initial_point1 = {-1.0, 1.0};
    Vector result1 = gradient_descent(rosenbrock, initial_point1);
    std::cout << "Final result: " << std::endl;
    print_vector(result1, "x*");
    std::cout << "f(x*) = " << rosenbrock(result1) << std::endl;
    
    // Example 2: Minimize Himmelblau function using Newton's Method
    std::cout << "\nExample 2: Himmelblau function with Newton's Method" << std::endl;
    Vector initial_point2 = {1.0, 1.0};
    Vector result2 = newtons_method(himmelblau, initial_point2);
    std::cout << "Final result: " << std::endl;
    print_vector(result2, "x*");
    std::cout << "f(x*) = " << himmelblau(result2) << std::endl;
    
    // Example 3: Minimize Sphere function using BFGS
    std::cout << "\nExample 3: Sphere function with BFGS" << std::endl;
    Vector initial_point3 = {2.0, 2.0, 2.0};
    Vector result3 = bfgs(sphere, initial_point3);
    std::cout << "Final result: " << std::endl;
    print_vector(result3, "x*");
    std::cout << "f(x*) = " << sphere(result3) << std::endl;
    
    // Example 4: Minimize Booth function using different methods
    std::cout << "\nExample 4: Booth function comparison" << std::endl;
    Vector initial_point4 = {0.0, 0.0};
    
    std::cout << "\nGradient Descent:" << std::endl;
    Vector result4a = gradient_descent(booth, initial_point4, 100, 1e-6);
    std::cout << "Final result: " << std::endl;
    print_vector(result4a, "x*");
    std::cout << "f(x*) = " << booth(result4a) << std::endl;
    
    std::cout << "\nNewton's Method:" << std::endl;
    Vector result4b = newtons_method(booth, initial_point4, 100, 1e-6);
    std::cout << "Final result: " << std::endl;
    print_vector(result4b, "x*");
    std::cout << "f(x*) = " << booth(result4b) << std::endl;
    
    std::cout << "\nBFGS:" << std::endl;
    Vector result4c = bfgs(booth, initial_point4, 100, 1e-6);
    std::cout << "Final result: " << std::endl;
    print_vector(result4c, "x*");
    std::cout << "f(x*) = " << booth(result4c) << std::endl;
    
    return 0;
}
