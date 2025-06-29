#include <iostream>
#include <vector>
#include <cmath>
#include <complex>
#include <iomanip>
#include <algorithm>

/**
 * Eigenvalues and Eigenvectors Implementation
 * 
 * This file demonstrates the power iteration method for finding
 * the dominant eigenvalue and eigenvector of a matrix.
 * 
 * Note: This is a simplified implementation for educational purposes.
 * For production use, consider using libraries like Eigen or LAPACK.
 */

using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Print a vector
void print_vector(const Vector& v, const std::string& name = "Vector") {
    std::cout << name << ": [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Print a matrix
void print_matrix(const Matrix& A, const std::string& name = "Matrix") {
    std::cout << name << ":" << std::endl;
    for (const auto& row : A) {
        std::cout << "  [";
        for (size_t j = 0; j < row.size(); ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(4) << row[j];
            if (j < row.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

// Matrix-vector multiplication
Vector matrix_vector_multiply(const Matrix& A, const Vector& v) {
    if (A[0].size() != v.size()) {
        throw std::invalid_argument("Matrix and vector dimensions incompatible for multiplication");
    }
    
    size_t rows = A.size();
    size_t cols = A[0].size();
    Vector result(rows, 0.0);
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i] += A[i][j] * v[j];
        }
    }
    return result;
}

// Vector dot product
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

// Vector norm (magnitude)
double vector_norm(const Vector& v) {
    return std::sqrt(dot_product(v, v));
}

// Normalize a vector
Vector normalize(const Vector& v) {
    double norm = vector_norm(v);
    if (norm < 1e-10) {
        throw std::invalid_argument("Cannot normalize a zero vector");
    }
    
    Vector result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = v[i] / norm;
    }
    return result;
}

// Power iteration method to find the dominant eigenvalue and eigenvector
std::pair<double, Vector> power_iteration(const Matrix& A, int max_iterations = 100, double tolerance = 1e-10) {
    if (A.size() != A[0].size()) {
        throw std::invalid_argument("Matrix must be square for eigenvalue computation");
    }
    
    size_t n = A.size();
    
    // Start with a random vector
    Vector v(n, 1.0);  // Could use random values, but 1.0 works for demonstration
    v = normalize(v);
    
    double eigenvalue = 0.0;
    double prev_eigenvalue = 0.0;
    
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Multiply matrix by current eigenvector estimate
        Vector Av = matrix_vector_multiply(A, v);
        
        // Compute Rayleigh quotient for eigenvalue estimate
        eigenvalue = dot_product(v, Av);
        
        // Normalize the resulting vector
        v = normalize(Av);
        
        // Check for convergence
        if (std::abs(eigenvalue - prev_eigenvalue) < tolerance) {
            std::cout << "Converged after " << iter + 1 << " iterations." << std::endl;
            break;
        }
        
        prev_eigenvalue = eigenvalue;
    }
    
    return {eigenvalue, v};
}

// Characteristic polynomial coefficients for a 2x2 matrix
// Returns coefficients of x^2 + b*x + c
std::pair<double, double> characteristic_polynomial_2x2(const Matrix& A) {
    if (A.size() != 2 || A[0].size() != 2) {
        throw std::invalid_argument("Matrix must be 2x2");
    }
    
    double b = -A[0][0] - A[1][1];  // Negative trace
    double c = A[0][0] * A[1][1] - A[0][1] * A[1][0];  // Determinant
    
    return {b, c};
}

// Solve quadratic equation ax^2 + bx + c = 0
std::pair<std::complex<double>, std::complex<double>> solve_quadratic(double a, double b, double c) {
    double discriminant = b*b - 4*a*c;
    
    if (discriminant >= 0) {
        double sqrt_disc = std::sqrt(discriminant);
        double x1 = (-b + sqrt_disc) / (2*a);
        double x2 = (-b - sqrt_disc) / (2*a);
        return {std::complex<double>(x1, 0), std::complex<double>(x2, 0)};
    } else {
        double real_part = -b / (2*a);
        double imag_part = std::sqrt(-discriminant) / (2*a);
        return {std::complex<double>(real_part, imag_part), std::complex<double>(real_part, -imag_part)};
    }
}

// Find eigenvalues of a 2x2 matrix analytically
std::pair<std::complex<double>, std::complex<double>> eigenvalues_2x2(const Matrix& A) {
    auto [b, c] = characteristic_polynomial_2x2(A);
    return solve_quadratic(1.0, b, c);
}

// Find eigenvector for a given eigenvalue (for 2x2 matrix)
Vector eigenvector_2x2(const Matrix& A, double eigenvalue) {
    // For a 2x2 matrix with eigenvalue λ, the eigenvector satisfies:
    // (A - λI)v = 0
    
    // Compute A - λI
    Matrix A_minus_lambda_I = {
        {A[0][0] - eigenvalue, A[0][1]},
        {A[1][0], A[1][1] - eigenvalue}
    };
    
    // If the first row has larger values, use it
    if (std::abs(A_minus_lambda_I[0][0]) > std::abs(A_minus_lambda_I[1][0]) || 
        std::abs(A_minus_lambda_I[0][1]) > std::abs(A_minus_lambda_I[1][1])) {
        
        if (std::abs(A_minus_lambda_I[0][0]) > 1e-10) {
            return {-A_minus_lambda_I[0][1], A_minus_lambda_I[0][0]};
        } else if (std::abs(A_minus_lambda_I[0][1]) > 1e-10) {
            return {A_minus_lambda_I[0][1], -A_minus_lambda_I[0][0]};
        }
    } else {
        if (std::abs(A_minus_lambda_I[1][0]) > 1e-10) {
            return {-A_minus_lambda_I[1][1], A_minus_lambda_I[1][0]};
        } else if (std::abs(A_minus_lambda_I[1][1]) > 1e-10) {
            return {A_minus_lambda_I[1][1], -A_minus_lambda_I[1][0]};
        }
    }
    
    // Fallback
    return {1.0, 0.0};
}

// Check if a vector is an eigenvector of a matrix
bool is_eigenvector(const Matrix& A, const Vector& v, double eigenvalue, double tolerance = 1e-10) {
    Vector Av = matrix_vector_multiply(A, v);
    Vector lambda_v(v.size());
    
    for (size_t i = 0; i < v.size(); ++i) {
        lambda_v[i] = eigenvalue * v[i];
    }
    
    double diff_norm = 0.0;
    for (size_t i = 0; i < v.size(); ++i) {
        diff_norm += std::pow(Av[i] - lambda_v[i], 2);
    }
    diff_norm = std::sqrt(diff_norm);
    
    return diff_norm < tolerance;
}

int main() {
    std::cout << "Eigenvalues and Eigenvectors Examples\n";
    std::cout << "===================================\n";
    
    // Example 1: A simple 2x2 matrix with real eigenvalues
    Matrix A1 = {
        {4.0, 1.0},
        {1.0, 3.0}
    };
    
    std::cout << "Example 1: 2x2 Matrix with real eigenvalues\n";
    print_matrix(A1, "Matrix A1");
    
    // Find eigenvalues analytically
    auto [lambda1, lambda2] = eigenvalues_2x2(A1);
    std::cout << "\nAnalytical eigenvalues:\n";
    std::cout << "λ1 = " << lambda1.real() << " + " << lambda1.imag() << "i\n";
    std::cout << "λ2 = " << lambda2.real() << " + " << lambda2.imag() << "i\n";
    
    // Find eigenvectors for real eigenvalues
    if (std::abs(lambda1.imag()) < 1e-10 && std::abs(lambda2.imag()) < 1e-10) {
        Vector v1 = eigenvector_2x2(A1, lambda1.real());
        Vector v2 = eigenvector_2x2(A1, lambda2.real());
        
        v1 = normalize(v1);
        v2 = normalize(v2);
        
        std::cout << "\nEigenvectors:\n";
        print_vector(v1, "v1 for λ1");
        print_vector(v2, "v2 for λ2");
        
        // Verify eigenvectors
        std::cout << "\nVerification:\n";
        std::cout << "Is v1 an eigenvector with λ1? " 
                  << (is_eigenvector(A1, v1, lambda1.real()) ? "Yes" : "No") << std::endl;
        std::cout << "Is v2 an eigenvector with λ2? " 
                  << (is_eigenvector(A1, v2, lambda2.real()) ? "Yes" : "No") << std::endl;
    }
    
    // Example 2: A matrix with a dominant eigenvalue
    Matrix A2 = {
        {5.0, 2.0, 1.0},
        {2.0, 3.0, 1.0},
        {1.0, 1.0, 2.0}
    };
    
    std::cout << "\nExample 2: 3x3 Matrix with power iteration\n";
    print_matrix(A2, "Matrix A2");
    
    // Find dominant eigenvalue and eigenvector using power iteration
    auto [eigenvalue, eigenvector] = power_iteration(A2);
    
    std::cout << "\nPower iteration results:\n";
    std::cout << "Dominant eigenvalue: " << eigenvalue << std::endl;
    print_vector(eigenvector, "Dominant eigenvector");
    
    // Verify
    std::cout << "\nVerification:\n";
    std::cout << "Is the vector an eigenvector? " 
              << (is_eigenvector(A2, eigenvector, eigenvalue) ? "Yes" : "No") << std::endl;
    
    // Example 3: A matrix with complex eigenvalues
    Matrix A3 = {
        {3.0, -2.0},
        {4.0, -1.0}
    };
    
    std::cout << "\nExample 3: 2x2 Matrix with complex eigenvalues\n";
    print_matrix(A3, "Matrix A3");
    
    // Find eigenvalues analytically
    auto [lambda3, lambda4] = eigenvalues_2x2(A3);
    std::cout << "\nAnalytical eigenvalues:\n";
    std::cout << "λ1 = " << lambda3.real() << " + " << lambda3.imag() << "i\n";
    std::cout << "λ2 = " << lambda4.real() << " + " << lambda4.imag() << "i\n";
    
    return 0;
}
