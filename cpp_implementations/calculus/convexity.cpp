#include <iostream>
#include <vector>
#include <functional>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <random>

/**
 * Convexity Implementation
 * 
 * This file demonstrates concepts of convexity and their importance
 * in optimization mathematics, with implementations in C++.
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

// Linear interpolation between two points
Vector lerp(const Vector& x, const Vector& y, double t) {
    if (x.size() != y.size()) {
        throw std::invalid_argument("Vector dimensions must match for interpolation");
    }
    
    Vector result(x.size());
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = (1 - t) * x[i] + t * y[i];
    }
    return result;
}

// Check if a function is convex using the definition:
// f((1-t)*x + t*y) <= (1-t)*f(x) + t*f(y) for all x, y and t in [0,1]
bool is_convex_by_definition(const Function& f, const Vector& domain_min, const Vector& domain_max, 
                           int num_tests = 1000, double tolerance = 1e-6) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    for (int test = 0; test < num_tests; ++test) {
        // Generate two random points in the domain
        Vector x(domain_min.size());
        Vector y(domain_min.size());
        
        for (size_t i = 0; i < domain_min.size(); ++i) {
            x[i] = domain_min[i] + dist(gen) * (domain_max[i] - domain_min[i]);
            y[i] = domain_min[i] + dist(gen) * (domain_max[i] - domain_min[i]);
        }
        
        // Generate a random t in [0,1]
        double t = dist(gen);
        
        // Compute the convex combination point
        Vector z = lerp(x, y, t);
        
        // Check the convexity condition
        double f_z = f(z);
        double convex_combo = (1 - t) * f(x) + t * f(y);
        
        if (f_z > convex_combo + tolerance) {
            std::cout << "Convexity violated at test " << test << ":\n";
            print_vector(x, "x");
            print_vector(y, "y");
            print_vector(z, "z = (1-t)*x + t*y with t = " + std::to_string(t));
            std::cout << "f(z) = " << f_z << "\n";
            std::cout << "(1-t)*f(x) + t*f(y) = " << convex_combo << "\n";
            std::cout << "Difference: " << f_z - convex_combo << "\n";
            return false;
        }
    }
    
    return true;
}

// Check if a function is convex using the Hessian criterion:
// A twice-differentiable function is convex if and only if its Hessian is positive semi-definite
bool is_convex_by_hessian(const Function& f, const Vector& domain_min, const Vector& domain_max, 
                         int num_tests = 100, double tolerance = 1e-6) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    for (int test = 0; test < num_tests; ++test) {
        // Generate a random point in the domain
        Vector x(domain_min.size());
        
        for (size_t i = 0; i < domain_min.size(); ++i) {
            x[i] = domain_min[i] + dist(gen) * (domain_max[i] - domain_min[i]);
        }
        
        // Compute the Hessian at this point
        auto hessian = compute_hessian(f, x);
        
        // Check if the Hessian is positive semi-definite
        // For a 2x2 matrix, we check if both eigenvalues are non-negative
        // For simplicity, we'll just check for 2x2 matrices here
        if (x.size() == 2) {
            double a = hessian[0][0];
            double b = hessian[0][1];
            double c = hessian[1][0];
            double d = hessian[1][1];
            
            // For a symmetric matrix, the eigenvalues are non-negative if:
            // 1. The trace (a+d) is non-negative
            // 2. The determinant (ad-bc) is non-negative
            double trace = a + d;
            double det = a * d - b * c;
            
            if (trace < -tolerance || det < -tolerance) {
                std::cout << "Hessian is not positive semi-definite at test " << test << ":\n";
                print_vector(x, "x");
                print_matrix(hessian, "Hessian");
                std::cout << "Trace: " << trace << "\n";
                std::cout << "Determinant: " << det << "\n";
                return false;
            }
        }
        else {
            // For higher dimensions, we would need to check all principal minors
            // or compute all eigenvalues, which is beyond the scope of this example
            std::cout << "Hessian check for dimensions > 2 is not implemented.\n";
            return false;
        }
    }
    
    return true;
}

// Check if a function is strictly convex using the Hessian criterion:
// A twice-differentiable function is strictly convex if its Hessian is positive definite
bool is_strictly_convex_by_hessian(const Function& f, const Vector& domain_min, const Vector& domain_max, 
                                 int num_tests = 100, double tolerance = 1e-6) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    for (int test = 0; test < num_tests; ++test) {
        // Generate a random point in the domain
        Vector x(domain_min.size());
        
        for (size_t i = 0; i < domain_min.size(); ++i) {
            x[i] = domain_min[i] + dist(gen) * (domain_max[i] - domain_min[i]);
        }
        
        // Compute the Hessian at this point
        auto hessian = compute_hessian(f, x);
        
        // Check if the Hessian is positive definite
        // For a 2x2 matrix, we check if both eigenvalues are positive
        if (x.size() == 2) {
            double a = hessian[0][0];
            double b = hessian[0][1];
            double c = hessian[1][0];
            double d = hessian[1][1];
            
            // For a symmetric matrix, the eigenvalues are positive if:
            // 1. The trace (a+d) is positive
            // 2. The determinant (ad-bc) is positive
            double trace = a + d;
            double det = a * d - b * c;
            
            if (trace <= tolerance || det <= tolerance || a <= tolerance) {
                std::cout << "Hessian is not positive definite at test " << test << ":\n";
                print_vector(x, "x");
                print_matrix(hessian, "Hessian");
                std::cout << "Trace: " << trace << "\n";
                std::cout << "Determinant: " << det << "\n";
                std::cout << "H[0][0]: " << a << "\n";
                return false;
            }
        }
        else {
            // For higher dimensions, we would need to check all principal minors
            // or compute all eigenvalues, which is beyond the scope of this example
            std::cout << "Hessian check for dimensions > 2 is not implemented.\n";
            return false;
        }
    }
    
    return true;
}

// Check if a function is convex using the first-order condition:
// f(y) >= f(x) + ∇f(x)·(y-x) for all x, y
bool is_convex_by_first_order(const Function& f, const Vector& domain_min, const Vector& domain_max, 
                             int num_tests = 1000, double tolerance = 1e-6) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    for (int test = 0; test < num_tests; ++test) {
        // Generate two random points in the domain
        Vector x(domain_min.size());
        Vector y(domain_min.size());
        
        for (size_t i = 0; i < domain_min.size(); ++i) {
            x[i] = domain_min[i] + dist(gen) * (domain_max[i] - domain_min[i]);
            y[i] = domain_min[i] + dist(gen) * (domain_max[i] - domain_min[i]);
        }
        
        // Compute gradient at x
        Vector grad_f_x = compute_gradient(f, x);
        
        // Compute y - x
        Vector y_minus_x(x.size());
        for (size_t i = 0; i < x.size(); ++i) {
            y_minus_x[i] = y[i] - x[i];
        }
        
        // Check the first-order condition
        double f_y = f(y);
        double f_x = f(x);
        double dot_grad_diff = dot_product(grad_f_x, y_minus_x);
        double first_order_bound = f_x + dot_grad_diff;
        
        if (f_y < first_order_bound - tolerance) {
            std::cout << "First-order convexity condition violated at test " << test << ":\n";
            print_vector(x, "x");
            print_vector(y, "y");
            print_vector(grad_f_x, "∇f(x)");
            std::cout << "f(y) = " << f_y << "\n";
            std::cout << "f(x) + ∇f(x)·(y-x) = " << first_order_bound << "\n";
            std::cout << "Difference: " << f_y - first_order_bound << "\n";
            return false;
        }
    }
    
    return true;
}

int main() {
    std::cout << "Convexity in Optimization Examples\n";
    std::cout << "=================================\n";
    
    // Example 1: Check convexity of a simple convex function
    std::cout << "\nExample 1: Convex function f(x,y) = x^2 + y^2\n";
    
    // f(x,y) = x^2 + y^2 (convex)
    auto f1 = [](const Vector& v) {
        double x = v[0];
        double y = v[1];
        return x*x + y*y;
    };
    
    Vector domain_min = {-10.0, -10.0};
    Vector domain_max = {10.0, 10.0};
    
    std::cout << "Testing convexity by definition...\n";
    bool is_convex_def = is_convex_by_definition(f1, domain_min, domain_max);
    std::cout << "Function is " << (is_convex_def ? "convex" : "not convex") << " by definition.\n";
    
    std::cout << "\nTesting convexity by Hessian criterion...\n";
    bool is_convex_hess = is_convex_by_hessian(f1, domain_min, domain_max);
    std::cout << "Function is " << (is_convex_hess ? "convex" : "not convex") << " by Hessian criterion.\n";
    
    std::cout << "\nTesting strict convexity by Hessian criterion...\n";
    bool is_strictly_convex = is_strictly_convex_by_hessian(f1, domain_min, domain_max);
    std::cout << "Function is " << (is_strictly_convex ? "strictly convex" : "not strictly convex") << " by Hessian criterion.\n";
    
    std::cout << "\nTesting convexity by first-order condition...\n";
    bool is_convex_fo = is_convex_by_first_order(f1, domain_min, domain_max);
    std::cout << "Function is " << (is_convex_fo ? "convex" : "not convex") << " by first-order condition.\n";
    
    // Example 2: Check convexity of a non-convex function
    std::cout << "\nExample 2: Non-convex function f(x,y) = x^2 - y^2\n";
    
    // f(x,y) = x^2 - y^2 (not convex)
    auto f2 = [](const Vector& v) {
        double x = v[0];
        double y = v[1];
        return x*x - y*y;
    };
    
    std::cout << "Testing convexity by definition...\n";
    is_convex_def = is_convex_by_definition(f2, domain_min, domain_max);
    std::cout << "Function is " << (is_convex_def ? "convex" : "not convex") << " by definition.\n";
    
    std::cout << "\nTesting convexity by Hessian criterion...\n";
    is_convex_hess = is_convex_by_hessian(f2, domain_min, domain_max);
    std::cout << "Function is " << (is_convex_hess ? "convex" : "not convex") << " by Hessian criterion.\n";
    
    // Example 3: Check convexity of a convex but not strictly convex function
    std::cout << "\nExample 3: Convex but not strictly convex function f(x,y) = |x| + |y|\n";
    
    // f(x,y) = |x| + |y| (convex but not strictly convex)
    auto f3 = [](const Vector& v) {
        double x = v[0];
        double y = v[1];
        return std::abs(x) + std::abs(y);
    };
    
    // Modify domain to avoid the non-differentiable points at x=0 or y=0
    Vector domain_min_mod = {0.1, 0.1};
    Vector domain_max_mod = {10.0, 10.0};
    
    std::cout << "Testing convexity by definition...\n";
    is_convex_def = is_convex_by_definition(f3, domain_min, domain_max);
    std::cout << "Function is " << (is_convex_def ? "convex" : "not convex") << " by definition.\n";
    
    std::cout << "\nTesting convexity by first-order condition...\n";
    is_convex_fo = is_convex_by_first_order(f3, domain_min_mod, domain_max_mod);
    std::cout << "Function is " << (is_convex_fo ? "convex" : "not convex") << " by first-order condition.\n";
    
    // Example 4: Convexity and optimization
    std::cout << "\nExample 4: Convexity and optimization\n";
    
    // For a convex function, any local minimum is a global minimum
    // Let's demonstrate this with gradient descent on a convex function
    
    // f(x,y) = (x-3)^2 + (y+2)^2 (convex with global minimum at (3,-2))
    auto f4 = [](const Vector& v) {
        double x = v[0];
        double y = v[1];
        return std::pow(x-3, 2) + std::pow(y+2, 2);
    };
    
    std::cout << "Function: f(x,y) = (x-3)^2 + (y+2)^2\n";
    std::cout << "Global minimum at (3,-2)\n\n";
    
    // Starting point
    Vector x0 = {0.0, 0.0};
    Vector x = x0;
    double learning_rate = 0.1;
    int max_iterations = 100;
    double tolerance = 1e-6;
    
    std::cout << "Starting gradient descent from ";
    print_vector(x0, "");
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        Vector grad = compute_gradient(f4, x);
        double grad_norm = vector_norm(grad);
        
        if (grad_norm < tolerance) {
            std::cout << "Converged after " << iter << " iterations.\n";
            break;
        }
        
        // Update x using gradient descent
        for (size_t i = 0; i < x.size(); ++i) {
            x[i] -= learning_rate * grad[i];
        }
        
        if (iter % 10 == 0 || iter == max_iterations - 1) {
            std::cout << "Iteration " << iter << ": ";
            print_vector(x, "x");
            std::cout << "f(x) = " << f4(x) << ", ||∇f|| = " << grad_norm << "\n";
        }
    }
    
    std::cout << "\nFinal result: ";
    print_vector(x, "x");
    std::cout << "f(x) = " << f4(x) << "\n";
    std::cout << "Distance from true minimum: " 
              << std::sqrt(std::pow(x[0]-3, 2) + std::pow(x[1]+2, 2)) << "\n";
    
    return 0;
}
