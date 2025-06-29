#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <iomanip>
#include <algorithm>

/**
 * Gradients Implementation
 * 
 * This file demonstrates gradient computation and gradient-based optimization
 * methods in C++, which are essential for optimization mathematics.
 */

using Vector = std::vector<double>;
using Function = std::function<double(const Vector&)>;

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

// Compute the Hessian matrix (matrix of second derivatives) of a function at a point
std::vector<Vector> compute_hessian(const Function& f, const Vector& x, double h = 1e-4) {
    size_t n = x.size();
    std::vector<Vector> hessian(n, Vector(n));
    
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = 0; j < n; ++j) {
            // For diagonal elements, use the central difference formula for second derivatives
            if (i == j) {
                Vector x_plus_h = x;
                Vector x_minus_h = x;
                x_plus_h[i] += h;
                x_minus_h[i] -= h;
                
                hessian[i][j] = (f(x_plus_h) - 2 * f(x) + f(x_minus_h)) / (h * h);
            }
            // For off-diagonal elements, use the mixed partial derivative formula
            else {
                Vector x_plus_plus = x;
                Vector x_plus_minus = x;
                Vector x_minus_plus = x;
                Vector x_minus_minus = x;
                
                x_plus_plus[i] += h;
                x_plus_plus[j] += h;
                
                x_plus_minus[i] += h;
                x_plus_minus[j] -= h;
                
                x_minus_plus[i] -= h;
                x_minus_plus[j] += h;
                
                x_minus_minus[i] -= h;
                x_minus_minus[j] -= h;
                
                hessian[i][j] = (f(x_plus_plus) - f(x_plus_minus) - f(x_minus_plus) + f(x_minus_minus)) / (4 * h * h);
            }
        }
    }
    
    return hessian;
}

// Print a matrix
void print_matrix(const std::vector<Vector>& A, const std::string& name = "Matrix") {
    std::cout << name << ":" << std::endl;
    for (const auto& row : A) {
        std::cout << "  [";
        for (size_t j = 0; j < row.size(); ++j) {
            std::cout << std::setw(10) << std::fixed << std::setprecision(6) << row[j];
            if (j < row.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
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

// Gradient Descent optimization method
Vector gradient_descent(const Function& f, Vector x0, double learning_rate = 0.01, 
                       double tolerance = 1e-6, int max_iterations = 1000) {
    Vector x = x0;
    Vector grad;
    double prev_value = f(x);
    
    std::cout << "Gradient Descent Optimization:\n";
    std::cout << "Iteration 0: f(x) = " << prev_value << ", x = ";
    print_vector(x, "");
    
    for (int iter = 1; iter <= max_iterations; ++iter) {
        // Compute gradient
        grad = compute_gradient(f, x);
        
        // Update x using gradient descent: x = x - learning_rate * grad
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] -= learning_rate * grad[i];
        }
        
        // Compute new function value
        double value = f(x);
        
        // Print progress every 10 iterations
        if (iter % 10 == 0 || iter == max_iterations) {
            std::cout << "Iteration " << iter << ": f(x) = " << value 
                      << ", ||grad|| = " << vector_norm(grad) << ", x = ";
            print_vector(x, "");
        }
        
        // Check for convergence
        if (vector_norm(grad) < tolerance) {
            std::cout << "Converged after " << iter << " iterations.\n";
            break;
        }
        
        // Check if we're making progress
        if (std::abs(value - prev_value) < tolerance) {
            std::cout << "Stopped: change in function value too small after " << iter << " iterations.\n";
            break;
        }
        
        prev_value = value;
    }
    
    return x;
}

// Gradient Descent with momentum
Vector gradient_descent_momentum(const Function& f, Vector x0, double learning_rate = 0.01,
                               double momentum = 0.9, double tolerance = 1e-6, int max_iterations = 1000) {
    Vector x = x0;
    Vector velocity(x.size(), 0.0);
    Vector grad;
    double prev_value = f(x);
    
    std::cout << "Gradient Descent with Momentum:\n";
    std::cout << "Iteration 0: f(x) = " << prev_value << ", x = ";
    print_vector(x, "");
    
    for (int iter = 1; iter <= max_iterations; ++iter) {
        // Compute gradient
        grad = compute_gradient(f, x);
        
        // Update velocity with momentum
        for (size_t i = 0; i < x.size(); ++i) {
            velocity[i] = momentum * velocity[i] - learning_rate * grad[i];
        }
        
        // Update x using velocity
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] += velocity[i];
        }
        
        // Compute new function value
        double value = f(x);
        
        // Print progress every 10 iterations
        if (iter % 10 == 0 || iter == max_iterations) {
            std::cout << "Iteration " << iter << ": f(x) = " << value 
                      << ", ||grad|| = " << vector_norm(grad) << ", x = ";
            print_vector(x, "");
        }
        
        // Check for convergence
        if (vector_norm(grad) < tolerance) {
            std::cout << "Converged after " << iter << " iterations.\n";
            break;
        }
        
        // Check if we're making progress
        if (std::abs(value - prev_value) < tolerance) {
            std::cout << "Stopped: change in function value too small after " << iter << " iterations.\n";
            break;
        }
        
        prev_value = value;
    }
    
    return x;
}

// Adaptive learning rate method (simplified AdaGrad)
Vector adagrad(const Function& f, Vector x0, double learning_rate = 0.01,
             double epsilon = 1e-8, double tolerance = 1e-6, int max_iterations = 1000) {
    Vector x = x0;
    Vector grad_squared_sum(x.size(), 0.0);
    Vector grad;
    double prev_value = f(x);
    
    std::cout << "AdaGrad Optimization:\n";
    std::cout << "Iteration 0: f(x) = " << prev_value << ", x = ";
    print_vector(x, "");
    
    for (int iter = 1; iter <= max_iterations; ++iter) {
        // Compute gradient
        grad = compute_gradient(f, x);
        
        // Update accumulated squared gradient
        for (size_t i = 0; i < x.size(); ++i) {
            grad_squared_sum[i] += grad[i] * grad[i];
        }
        
        // Update x using adaptive learning rate
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] -= learning_rate * grad[i] / (std::sqrt(grad_squared_sum[i]) + epsilon);
        }
        
        // Compute new function value
        double value = f(x);
        
        // Print progress every 10 iterations
        if (iter % 10 == 0 || iter == max_iterations) {
            std::cout << "Iteration " << iter << ": f(x) = " << value 
                      << ", ||grad|| = " << vector_norm(grad) << ", x = ";
            print_vector(x, "");
        }
        
        // Check for convergence
        if (vector_norm(grad) < tolerance) {
            std::cout << "Converged after " << iter << " iterations.\n";
            break;
        }
        
        // Check if we're making progress
        if (std::abs(value - prev_value) < tolerance) {
            std::cout << "Stopped: change in function value too small after " << iter << " iterations.\n";
            break;
        }
        
        prev_value = value;
    }
    
    return x;
}

int main() {
    std::cout << "Gradients and Gradient-Based Optimization Examples\n";
    std::cout << "================================================\n";
    
    // Example 1: Compute gradient of a simple function
    std::cout << "\nExample 1: Gradient Computation\n";
    
    // f(x,y) = x^2 + 2*y^2 + x*y
    auto f1 = [](const Vector& v) {
        double x = v[0];
        double y = v[1];
        return x*x + 2*y*y + x*y;
    };
    
    Vector point = {2.0, -1.0};
    Vector grad = compute_gradient(f1, point);
    
    std::cout << "Function: f(x,y) = x^2 + 2*y^2 + x*y\n";
    std::cout << "Point: (x,y) = (" << point[0] << ", " << point[1] << ")\n";
    print_vector(grad, "Gradient ∇f");
    std::cout << "Analytical gradient: [2x + y, 4y + x] = [" 
              << 2*point[0] + point[1] << ", " << 4*point[1] + point[0] << "]\n";
    
    // Example 2: Compute Hessian matrix
    std::cout << "\nExample 2: Hessian Computation\n";
    
    auto hessian = compute_hessian(f1, point);
    print_matrix(hessian, "Hessian matrix");
    std::cout << "Analytical Hessian:\n";
    std::cout << "  [    2.0,     1.0]\n";
    std::cout << "  [    1.0,     4.0]\n";
    
    // Example 3: Gradient Descent Optimization
    std::cout << "\nExample 3: Gradient Descent Optimization\n";
    
    // Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    auto rosenbrock = [](const Vector& v) {
        double x = v[0];
        double y = v[1];
        return std::pow(1 - x, 2) + 100 * std::pow(y - x*x, 2);
    };
    
    Vector start_point = {-1.0, 1.0};
    std::cout << "Optimizing Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2\n";
    std::cout << "Starting point: (" << start_point[0] << ", " << start_point[1] << ")\n";
    std::cout << "Global minimum at (1, 1)\n\n";
    
    // Run gradient descent
    Vector result_gd = gradient_descent(rosenbrock, start_point, 0.0001, 1e-6, 1000);
    std::cout << "\nGradient Descent result: ";
    print_vector(result_gd, "");
    std::cout << "Function value at result: " << rosenbrock(result_gd) << "\n";
    
    // Example 4: Gradient Descent with Momentum
    std::cout << "\nExample 4: Gradient Descent with Momentum\n";
    
    Vector result_gdm = gradient_descent_momentum(rosenbrock, start_point, 0.0001, 0.9, 1e-6, 1000);
    std::cout << "\nGradient Descent with Momentum result: ";
    print_vector(result_gdm, "");
    std::cout << "Function value at result: " << rosenbrock(result_gdm) << "\n";
    
    // Example 5: AdaGrad Optimization
    std::cout << "\nExample 5: AdaGrad Optimization\n";
    
    Vector result_adagrad = adagrad(rosenbrock, start_point, 0.1, 1e-8, 1e-6, 1000);
    std::cout << "\nAdaGrad result: ";
    print_vector(result_adagrad, "");
    std::cout << "Function value at result: " << rosenbrock(result_adagrad) << "\n";
    
    return 0;
}
