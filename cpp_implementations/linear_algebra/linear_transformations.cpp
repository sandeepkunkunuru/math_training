#include <iostream>
#include <vector>
#include <iomanip>
#include <cmath>
#include <stdexcept>

/**
 * Linear Transformations Implementation
 * 
 * This file demonstrates linear transformations in C++ that are
 * essential for optimization mathematics.
 */

// Matrix and vector representations
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

// Print a vector
void print_vector(const Vector& v, const std::string& name = "Vector") {
    std::cout << name << ": [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(2) << v[i];
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
            std::cout << std::setw(6) << std::fixed << std::setprecision(2) << row[j];
            if (j < row.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

// Matrix-vector multiplication (linear transformation)
Vector transform(const Matrix& A, const Vector& v) {
    if (A[0].size() != v.size()) {
        throw std::invalid_argument("Matrix and vector dimensions incompatible for transformation");
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

// Create a 2D rotation matrix for angle theta (in radians)
Matrix rotation_matrix_2d(double theta) {
    Matrix R = {
        {std::cos(theta), -std::sin(theta)},
        {std::sin(theta), std::cos(theta)}
    };
    return R;
}

// Create a 2D scaling matrix with factors sx and sy
Matrix scaling_matrix_2d(double sx, double sy) {
    Matrix S = {
        {sx, 0.0},
        {0.0, sy}
    };
    return S;
}

// Create a 2D shear matrix with shear factors kx and ky
Matrix shear_matrix_2d(double kx, double ky) {
    Matrix H = {
        {1.0, kx},
        {ky, 1.0}
    };
    return H;
}

// Create a 3D rotation matrix around the x-axis
Matrix rotation_matrix_3d_x(double theta) {
    Matrix R = {
        {1.0, 0.0, 0.0},
        {0.0, std::cos(theta), -std::sin(theta)},
        {0.0, std::sin(theta), std::cos(theta)}
    };
    return R;
}

// Create a 3D rotation matrix around the y-axis
Matrix rotation_matrix_3d_y(double theta) {
    Matrix R = {
        {std::cos(theta), 0.0, std::sin(theta)},
        {0.0, 1.0, 0.0},
        {-std::sin(theta), 0.0, std::cos(theta)}
    };
    return R;
}

// Create a 3D rotation matrix around the z-axis
Matrix rotation_matrix_3d_z(double theta) {
    Matrix R = {
        {std::cos(theta), -std::sin(theta), 0.0},
        {std::sin(theta), std::cos(theta), 0.0},
        {0.0, 0.0, 1.0}
    };
    return R;
}

// Create a 3D scaling matrix
Matrix scaling_matrix_3d(double sx, double sy, double sz) {
    Matrix S = {
        {sx, 0.0, 0.0},
        {0.0, sy, 0.0},
        {0.0, 0.0, sz}
    };
    return S;
}

// Matrix multiplication for composing transformations
Matrix matrix_multiply(const Matrix& A, const Matrix& B) {
    if (A[0].size() != B.size()) {
        throw std::invalid_argument("Matrix dimensions incompatible for multiplication");
    }
    
    size_t rows_A = A.size();
    size_t cols_A = A[0].size();
    size_t cols_B = B[0].size();
    
    Matrix result(rows_A, std::vector<double>(cols_B, 0.0));
    
    for (size_t i = 0; i < rows_A; ++i) {
        for (size_t j = 0; j < cols_B; ++j) {
            for (size_t k = 0; k < cols_A; ++k) {
                result[i][j] += A[i][k] * B[k][j];
            }
        }
    }
    return result;
}

// Compute the determinant of a 2x2 matrix
double determinant_2x2(const Matrix& A) {
    return A[0][0] * A[1][1] - A[0][1] * A[1][0];
}

// Check if a transformation preserves area/volume
bool is_area_preserving(const Matrix& A) {
    // For 2x2 matrices, check if determinant is 1
    if (A.size() == 2 && A[0].size() == 2) {
        return std::abs(determinant_2x2(A) - 1.0) < 1e-10;
    }
    
    // For other sizes, this would be more complex
    throw std::invalid_argument("Area preservation check only implemented for 2x2 matrices");
}

int main() {
    std::cout << "Linear Transformations Examples\n";
    std::cout << "==============================\n";
    
    // 2D examples
    Vector v2d = {3.0, 2.0};
    std::cout << "2D Transformations:\n";
    print_vector(v2d, "Original 2D vector");
    
    // Rotation
    double angle = M_PI / 4.0;  // 45 degrees
    Matrix R2d = rotation_matrix_2d(angle);
    print_matrix(R2d, "2D Rotation matrix (45 degrees)");
    Vector v2d_rotated = transform(R2d, v2d);
    print_vector(v2d_rotated, "Rotated vector");
    
    // Scaling
    Matrix S2d = scaling_matrix_2d(2.0, 0.5);
    print_matrix(S2d, "2D Scaling matrix (2x, 0.5y)");
    Vector v2d_scaled = transform(S2d, v2d);
    print_vector(v2d_scaled, "Scaled vector");
    
    // Shearing
    Matrix H2d = shear_matrix_2d(0.5, 0.0);
    print_matrix(H2d, "2D Shear matrix (x-direction)");
    Vector v2d_sheared = transform(H2d, v2d);
    print_vector(v2d_sheared, "Sheared vector");
    
    // Composition of transformations
    std::cout << "\nComposition of transformations:\n";
    // First rotate, then scale
    Matrix RS = matrix_multiply(S2d, R2d);
    print_matrix(RS, "Scale after rotate");
    Vector v2d_rs = transform(RS, v2d);
    print_vector(v2d_rs, "Rotated then scaled vector");
    
    // First scale, then rotate
    Matrix SR = matrix_multiply(R2d, S2d);
    print_matrix(SR, "Rotate after scale");
    Vector v2d_sr = transform(SR, v2d);
    print_vector(v2d_sr, "Scaled then rotated vector");
    
    std::cout << "\nNote that RS != SR, showing that order matters in transformations!\n";
    
    // 3D examples
    std::cout << "\n3D Transformations:\n";
    Vector v3d = {1.0, 2.0, 3.0};
    print_vector(v3d, "Original 3D vector");
    
    // 3D rotation around x-axis
    Matrix Rx = rotation_matrix_3d_x(M_PI / 2.0);  // 90 degrees
    print_matrix(Rx, "3D Rotation matrix around x-axis (90 degrees)");
    Vector v3d_rotated_x = transform(Rx, v3d);
    print_vector(v3d_rotated_x, "Vector rotated around x-axis");
    
    // 3D scaling
    Matrix S3d = scaling_matrix_3d(2.0, 2.0, 2.0);
    print_matrix(S3d, "3D Scaling matrix (uniform 2x)");
    Vector v3d_scaled = transform(S3d, v3d);
    print_vector(v3d_scaled, "Scaled 3D vector");
    
    // Area preservation
    std::cout << "\nArea Preservation:\n";
    std::cout << "Rotation preserves area: " << (is_area_preserving(R2d) ? "Yes" : "No") << std::endl;
    std::cout << "Scaling preserves area: " << (is_area_preserving(S2d) ? "Yes" : "No") << std::endl;
    
    return 0;
}
