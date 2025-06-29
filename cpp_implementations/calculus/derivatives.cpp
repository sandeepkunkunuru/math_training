#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <iomanip>

/**
 * Derivatives Implementation
 * 
 * This file demonstrates numerical methods for computing derivatives
 * in C++, which are essential for optimization mathematics.
 */

// Function to compute the derivative of a single-variable function using central difference
double derivative(const std::function<double(double)>& f, double x, double h = 1e-5) {
    // Central difference formula: f'(x) ≈ [f(x+h) - f(x-h)] / (2h)
    return (f(x + h) - f(x - h)) / (2 * h);
}

// Function to compute the second derivative using central difference
double second_derivative(const std::function<double(double)>& f, double x, double h = 1e-5) {
    // Central difference formula: f''(x) ≈ [f(x+h) - 2f(x) + f(x-h)] / h²
    return (f(x + h) - 2 * f(x) + f(x - h)) / (h * h);
}

// Function to compute the partial derivative with respect to the i-th variable
double partial_derivative(const std::function<double(const std::vector<double>&)>& f, 
                         const std::vector<double>& x, int i, double h = 1e-5) {
    std::vector<double> x_plus_h = x;
    std::vector<double> x_minus_h = x;
    
    x_plus_h[i] += h;
    x_minus_h[i] -= h;
    
    // Central difference formula for partial derivative
    return (f(x_plus_h) - f(x_minus_h)) / (2 * h);
}

// Function to compute the directional derivative
double directional_derivative(const std::function<double(const std::vector<double>&)>& f,
                             const std::vector<double>& x, const std::vector<double>& direction,
                             double h = 1e-5) {
    // Normalize the direction vector
    double norm = 0.0;
    for (double d : direction) {
        norm += d * d;
    }
    norm = std::sqrt(norm);
    
    if (norm < 1e-10) {
        throw std::invalid_argument("Direction vector cannot be zero");
    }
    
    std::vector<double> unit_direction(direction.size());
    for (size_t i = 0; i < direction.size(); ++i) {
        unit_direction[i] = direction[i] / norm;
    }
    
    // Compute points along the direction
    std::vector<double> x_plus_h(x.size());
    std::vector<double> x_minus_h(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        x_plus_h[i] = x[i] + h * unit_direction[i];
        x_minus_h[i] = x[i] - h * unit_direction[i];
    }
    
    // Central difference formula for directional derivative
    return (f(x_plus_h) - f(x_minus_h)) / (2 * h);
}

// Function to compute all partial derivatives (gradient) of a function
std::vector<double> gradient(const std::function<double(const std::vector<double>&)>& f,
                           const std::vector<double>& x, double h = 1e-5) {
    std::vector<double> grad(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        grad[i] = partial_derivative(f, x, i, h);
    }
    
    return grad;
}

// Function to print a vector
void print_vector(const std::vector<double>& v, const std::string& name = "Vector") {
    std::cout << name << ": [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

int main() {
    std::cout << "Derivatives Examples\n";
    std::cout << "===================\n";
    
    // Example 1: Single-variable derivatives
    std::cout << "Example 1: Single-variable derivatives\n";
    
    // f(x) = x^2
    auto f1 = [](double x) { return x * x; };
    // Analytical derivative: f'(x) = 2x
    auto df1 = [](double x) { return 2 * x; };
    
    double x = 3.0;
    double df_numeric = derivative(f1, x);
    double df_analytic = df1(x);
    
    std::cout << "Function: f(x) = x^2\n";
    std::cout << "x = " << x << "\n";
    std::cout << "Numerical derivative: " << df_numeric << "\n";
    std::cout << "Analytical derivative: " << df_analytic << "\n";
    std::cout << "Difference: " << std::abs(df_numeric - df_analytic) << "\n";
    
    // Second derivative
    double d2f_numeric = second_derivative(f1, x);
    std::cout << "Numerical second derivative: " << d2f_numeric << "\n";
    std::cout << "Analytical second derivative: 2\n";
    std::cout << "Difference: " << std::abs(d2f_numeric - 2.0) << "\n";
    
    // Example 2: Multi-variable derivatives (partial derivatives)
    std::cout << "\nExample 2: Multi-variable derivatives\n";
    
    // f(x,y) = x^2 + xy + y^2
    auto f2 = [](const std::vector<double>& v) {
        double x = v[0];
        double y = v[1];
        return x*x + x*y + y*y;
    };
    
    std::vector<double> point = {2.0, 3.0};
    
    // Partial derivatives
    double df_dx = partial_derivative(f2, point, 0);
    double df_dy = partial_derivative(f2, point, 1);
    
    std::cout << "Function: f(x,y) = x^2 + xy + y^2\n";
    std::cout << "Point: (x,y) = (" << point[0] << ", " << point[1] << ")\n";
    std::cout << "∂f/∂x = " << df_dx << " (Analytical: " << 2*point[0] + point[1] << ")\n";
    std::cout << "∂f/∂y = " << df_dy << " (Analytical: " << point[0] + 2*point[1] << ")\n";
    
    // Gradient
    std::vector<double> grad = gradient(f2, point);
    print_vector(grad, "Gradient ∇f");
    
    // Example 3: Directional derivative
    std::cout << "\nExample 3: Directional derivative\n";
    
    std::vector<double> direction = {1.0, 1.0};
    double dir_deriv = directional_derivative(f2, point, direction);
    
    std::cout << "Direction: (";
    for (size_t i = 0; i < direction.size(); ++i) {
        std::cout << direction[i];
        if (i < direction.size() - 1) std::cout << ", ";
    }
    std::cout << ")\n";
    std::cout << "Directional derivative: " << dir_deriv << "\n";
    
    // Compute directional derivative using the gradient
    // D_v f = ∇f · v/|v|
    double dot_product = 0.0;
    double norm = 0.0;
    
    for (size_t i = 0; i < direction.size(); ++i) {
        dot_product += grad[i] * direction[i];
        norm += direction[i] * direction[i];
    }
    norm = std::sqrt(norm);
    
    double dir_deriv_from_grad = dot_product / norm;
    std::cout << "Directional derivative from gradient: " << dir_deriv_from_grad << "\n";
    std::cout << "Difference: " << std::abs(dir_deriv - dir_deriv_from_grad) << "\n";
    
    return 0;
}
