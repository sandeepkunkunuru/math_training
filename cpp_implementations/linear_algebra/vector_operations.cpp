#include <iostream>
#include <vector>
#include <cmath>
#include <numeric>
#include <algorithm>

/**
 * Vector Operations Implementation
 * 
 * This file demonstrates fundamental vector operations in C++ that are
 * essential for optimization mathematics.
 */

// Vector representation using std::vector
using Vector = std::vector<double>;

// Print a vector
void print_vector(const Vector& v, const std::string& name = "Vector") {
    std::cout << name << ": [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Vector addition: v1 + v2
Vector vector_add(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimensions must match for addition");
    }
    
    Vector result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] + v2[i];
    }
    return result;
}

// Vector subtraction: v1 - v2
Vector vector_subtract(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimensions must match for subtraction");
    }
    
    Vector result(v1.size());
    for (size_t i = 0; i < v1.size(); ++i) {
        result[i] = v1[i] - v2[i];
    }
    return result;
}

// Scalar multiplication: c * v
Vector scalar_multiply(double c, const Vector& v) {
    Vector result(v.size());
    for (size_t i = 0; i < v.size(); ++i) {
        result[i] = c * v[i];
    }
    return result;
}

// Dot product: v1 · v2
double dot_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != v2.size()) {
        throw std::invalid_argument("Vector dimensions must match for dot product");
    }
    
    return std::inner_product(v1.begin(), v1.end(), v2.begin(), 0.0);
}

// Vector norm (magnitude): |v|
double vector_norm(const Vector& v) {
    return std::sqrt(dot_product(v, v));
}

// Vector normalization: v / |v|
Vector normalize(const Vector& v) {
    double norm = vector_norm(v);
    if (norm < 1e-10) {
        throw std::invalid_argument("Cannot normalize a zero vector");
    }
    
    return scalar_multiply(1.0 / norm, v);
}

// Cross product (only for 3D vectors): v1 × v2
Vector cross_product(const Vector& v1, const Vector& v2) {
    if (v1.size() != 3 || v2.size() != 3) {
        throw std::invalid_argument("Cross product is only defined for 3D vectors");
    }
    
    Vector result(3);
    result[0] = v1[1] * v2[2] - v1[2] * v2[1];
    result[1] = v1[2] * v2[0] - v1[0] * v2[2];
    result[2] = v1[0] * v2[1] - v1[1] * v2[0];
    return result;
}

// Projection of v1 onto v2
Vector projection(const Vector& v1, const Vector& v2) {
    double dot = dot_product(v1, v2);
    double normSquared = dot_product(v2, v2);
    
    if (normSquared < 1e-10) {
        throw std::invalid_argument("Cannot project onto a zero vector");
    }
    
    return scalar_multiply(dot / normSquared, v2);
}

// Check if vectors are orthogonal (perpendicular)
bool are_orthogonal(const Vector& v1, const Vector& v2) {
    return std::abs(dot_product(v1, v2)) < 1e-10;
}

// Check if vectors are parallel
bool are_parallel(const Vector& v1, const Vector& v2) {
    // Normalize both vectors and check if they're the same or opposites
    if (v1.empty() || v2.empty()) {
        throw std::invalid_argument("Cannot check parallelism of empty vectors");
    }
    
    if (vector_norm(v1) < 1e-10 || vector_norm(v2) < 1e-10) {
        throw std::invalid_argument("Zero vectors are parallel to all vectors");
    }
    
    Vector n1 = normalize(v1);
    Vector n2 = normalize(v2);
    
    double dot = std::abs(dot_product(n1, n2));
    return std::abs(dot - 1.0) < 1e-10;
}

int main() {
    // Example vectors
    Vector v1 = {1.0, 2.0, 3.0};
    Vector v2 = {4.0, 5.0, 6.0};
    Vector v3 = {0.0, 0.0, 5.0};
    Vector v4 = {0.0, 0.0, -10.0};
    
    std::cout << "Vector Operations Examples\n";
    std::cout << "=========================\n";
    
    print_vector(v1, "v1");
    print_vector(v2, "v2");
    
    std::cout << "\nBasic Operations:\n";
    print_vector(vector_add(v1, v2), "v1 + v2");
    print_vector(vector_subtract(v1, v2), "v1 - v2");
    print_vector(scalar_multiply(2.0, v1), "2 * v1");
    
    std::cout << "\nDot Product: v1 · v2 = " << dot_product(v1, v2) << std::endl;
    std::cout << "Vector Norm: |v1| = " << vector_norm(v1) << std::endl;
    
    std::cout << "\nNormalized Vectors:\n";
    print_vector(normalize(v1), "normalize(v1)");
    print_vector(normalize(v2), "normalize(v2)");
    
    std::cout << "\nCross Product:\n";
    print_vector(cross_product(v1, v2), "v1 × v2");
    
    std::cout << "\nProjection:\n";
    print_vector(projection(v1, v2), "projection of v1 onto v2");
    
    std::cout << "\nVector Relationships:\n";
    std::cout << "v1 and v2 are orthogonal: " << (are_orthogonal(v1, v2) ? "Yes" : "No") << std::endl;
    std::cout << "v3 and v4 are parallel: " << (are_parallel(v3, v4) ? "Yes" : "No") << std::endl;
    
    return 0;
}
