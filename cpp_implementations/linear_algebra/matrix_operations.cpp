#include <iostream>
#include <vector>
#include <iomanip>
#include <stdexcept>
#include <cmath>

/**
 * Matrix Operations Implementation
 * 
 * This file demonstrates fundamental matrix operations in C++ that are
 * essential for optimization mathematics.
 */

// Matrix representation using 2D vector
using Matrix = std::vector<std::vector<double>>;
using Vector = std::vector<double>;

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

// Create an identity matrix of size n×n
Matrix identity_matrix(size_t n) {
    Matrix result(n, std::vector<double>(n, 0.0));
    for (size_t i = 0; i < n; ++i) {
        result[i][i] = 1.0;
    }
    return result;
}

// Matrix addition: A + B
Matrix matrix_add(const Matrix& A, const Matrix& B) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        throw std::invalid_argument("Matrix dimensions must match for addition");
    }
    
    size_t rows = A.size();
    size_t cols = A[0].size();
    Matrix result(rows, std::vector<double>(cols, 0.0));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] + B[i][j];
        }
    }
    return result;
}

// Matrix subtraction: A - B
Matrix matrix_subtract(const Matrix& A, const Matrix& B) {
    if (A.size() != B.size() || A[0].size() != B[0].size()) {
        throw std::invalid_argument("Matrix dimensions must match for subtraction");
    }
    
    size_t rows = A.size();
    size_t cols = A[0].size();
    Matrix result(rows, std::vector<double>(cols, 0.0));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = A[i][j] - B[i][j];
        }
    }
    return result;
}

// Scalar multiplication: c * A
Matrix scalar_multiply(double c, const Matrix& A) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    Matrix result(rows, std::vector<double>(cols, 0.0));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[i][j] = c * A[i][j];
        }
    }
    return result;
}

// Matrix multiplication: A * B
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

// Matrix-vector multiplication: A * v
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

// Transpose of a matrix
Matrix transpose(const Matrix& A) {
    size_t rows = A.size();
    size_t cols = A[0].size();
    Matrix result(cols, std::vector<double>(rows, 0.0));
    
    for (size_t i = 0; i < rows; ++i) {
        for (size_t j = 0; j < cols; ++j) {
            result[j][i] = A[i][j];
        }
    }
    return result;
}

// Calculate determinant of a matrix (recursive implementation)
double determinant(const Matrix& A) {
    size_t n = A.size();
    
    if (n != A[0].size()) {
        throw std::invalid_argument("Matrix must be square to calculate determinant");
    }
    
    // Base case for 1x1 matrix
    if (n == 1) {
        return A[0][0];
    }
    
    // Base case for 2x2 matrix
    if (n == 2) {
        return A[0][0] * A[1][1] - A[0][1] * A[1][0];
    }
    
    double det = 0.0;
    
    // For each element in the first row
    for (size_t j = 0; j < n; ++j) {
        // Create submatrix by excluding first row and current column
        Matrix submatrix(n - 1, std::vector<double>(n - 1, 0.0));
        
        for (size_t row = 1; row < n; ++row) {
            size_t col_sub = 0;
            for (size_t col = 0; col < n; ++col) {
                if (col == j) continue;
                submatrix[row - 1][col_sub] = A[row][col];
                ++col_sub;
            }
        }
        
        // Add or subtract the determinant of the submatrix
        det += (j % 2 == 0 ? 1 : -1) * A[0][j] * determinant(submatrix);
    }
    
    return det;
}

// Calculate the trace of a matrix (sum of diagonal elements)
double trace(const Matrix& A) {
    if (A.size() != A[0].size()) {
        throw std::invalid_argument("Matrix must be square to calculate trace");
    }
    
    double tr = 0.0;
    for (size_t i = 0; i < A.size(); ++i) {
        tr += A[i][i];
    }
    return tr;
}

// Calculate the inverse of a 2x2 matrix
Matrix inverse_2x2(const Matrix& A) {
    if (A.size() != 2 || A[0].size() != 2) {
        throw std::invalid_argument("Matrix must be 2x2 to use this inverse function");
    }
    
    double det = determinant(A);
    if (std::abs(det) < 1e-10) {
        throw std::invalid_argument("Matrix is singular, cannot compute inverse");
    }
    
    Matrix result(2, std::vector<double>(2, 0.0));
    result[0][0] = A[1][1] / det;
    result[0][1] = -A[0][1] / det;
    result[1][0] = -A[1][0] / det;
    result[1][1] = A[0][0] / det;
    
    return result;
}

// Check if a matrix is symmetric (A = A^T)
bool is_symmetric(const Matrix& A) {
    if (A.size() != A[0].size()) {
        return false;  // Non-square matrices cannot be symmetric
    }
    
    size_t n = A.size();
    for (size_t i = 0; i < n; ++i) {
        for (size_t j = i + 1; j < n; ++j) {
            if (std::abs(A[i][j] - A[j][i]) > 1e-10) {
                return false;
            }
        }
    }
    return true;
}

int main() {
    // Example matrices
    Matrix A = {
        {1.0, 2.0, 3.0},
        {4.0, 5.0, 6.0},
        {7.0, 8.0, 9.0}
    };
    
    Matrix B = {
        {9.0, 8.0, 7.0},
        {6.0, 5.0, 4.0},
        {3.0, 2.0, 1.0}
    };
    
    Matrix C = {
        {1.0, 2.0},
        {3.0, 4.0}
    };
    
    Matrix D = {
        {1.0, 2.0},
        {2.0, 3.0}
    };
    
    Vector v = {1.0, 2.0, 3.0};
    
    std::cout << "Matrix Operations Examples\n";
    std::cout << "=========================\n";
    
    print_matrix(A, "Matrix A");
    print_matrix(B, "Matrix B");
    
    std::cout << "\nBasic Operations:\n";
    print_matrix(matrix_add(A, B), "A + B");
    print_matrix(matrix_subtract(A, B), "A - B");
    print_matrix(scalar_multiply(2.0, A), "2 * A");
    
    std::cout << "\nMatrix Multiplication:\n";
    print_matrix(matrix_multiply(A, B), "A * B");
    
    std::cout << "\nMatrix-Vector Multiplication:\n";
    Vector Av = matrix_vector_multiply(A, v);
    std::cout << "A * v = [";
    for (size_t i = 0; i < Av.size(); ++i) {
        std::cout << Av[i];
        if (i < Av.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
    
    std::cout << "\nMatrix Transpose:\n";
    print_matrix(transpose(A), "A^T");
    
    std::cout << "\nMatrix Properties:\n";
    std::cout << "Determinant of C = " << determinant(C) << std::endl;
    std::cout << "Trace of A = " << trace(A) << std::endl;
    
    std::cout << "\nMatrix Inverse (2x2):\n";
    print_matrix(inverse_2x2(C), "C^(-1)");
    
    std::cout << "\nMatrix Symmetry:\n";
    std::cout << "Is D symmetric? " << (is_symmetric(D) ? "Yes" : "No") << std::endl;
    std::cout << "Is A symmetric? " << (is_symmetric(A) ? "Yes" : "No") << std::endl;
    
    std::cout << "\nIdentity Matrix:\n";
    print_matrix(identity_matrix(3), "I_3");
    
    return 0;
}
