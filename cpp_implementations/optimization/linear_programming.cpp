#include <iostream>
#include <vector>
#include <cmath>
#include <iomanip>
#include <algorithm>
#include <limits>
#include <string>

/**
 * Linear Programming Implementation
 * 
 * This file demonstrates the implementation of linear programming algorithms
 * including the Simplex method and related concepts.
 */

// Type definitions for clarity
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

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
void print_matrix(const Matrix& m, const std::string& name = "Matrix") {
    std::cout << name << ":" << std::endl;
    for (size_t i = 0; i < m.size(); ++i) {
        std::cout << "  [";
        for (size_t j = 0; j < m[i].size(); ++j) {
            std::cout << std::fixed << std::setprecision(4) << m[i][j];
            if (j < m[i].size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
    }
}

// Print the current tableau
void print_tableau(const Matrix& tableau, const std::vector<int>& basic_vars, 
                  const std::vector<int>& non_basic_vars) {
    std::cout << "Current Tableau:" << std::endl;
    
    // Print header row with variable names
    std::cout << std::setw(8) << "Basic";
    for (int var : non_basic_vars) {
        std::cout << std::setw(8) << "x" + std::to_string(var);
    }
    std::cout << std::setw(8) << "RHS" << std::endl;
    
    // Print separator line
    std::cout << std::string(8 * (non_basic_vars.size() + 2), '-') << std::endl;
    
    // Print each row of the tableau
    for (size_t i = 0; i < tableau.size() - 1; ++i) {
        std::cout << std::setw(8) << "x" + std::to_string(basic_vars[i]);
        for (size_t j = 0; j < tableau[i].size() - 1; ++j) {
            std::cout << std::setw(8) << std::fixed << std::setprecision(2) << tableau[i][j];
        }
        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << tableau[i].back() << std::endl;
    }
    
    // Print objective row
    std::cout << std::setw(8) << "z";
    for (size_t j = 0; j < tableau.back().size() - 1; ++j) {
        std::cout << std::setw(8) << std::fixed << std::setprecision(2) << tableau.back()[j];
    }
    std::cout << std::setw(8) << std::fixed << std::setprecision(2) << tableau.back().back() << std::endl;
}

// Convert a linear program to standard form
// Standard form: Maximize c^T x subject to Ax = b, x >= 0
void convert_to_standard_form(Matrix& A, Vector& b, Vector& c, bool is_maximization) {
    // If minimization problem, negate the objective function to convert to maximization
    if (!is_maximization) {
        for (double& val : c) {
            val = -val;
        }
    }
    
    // Ensure b is non-negative (multiply constraints by -1 if b_i < 0)
    for (size_t i = 0; i < b.size(); ++i) {
        if (b[i] < 0) {
            b[i] = -b[i];
            for (size_t j = 0; j < A[i].size(); ++j) {
                A[i][j] = -A[i][j];
            }
        }
    }
}

// Initialize the simplex tableau from the standard form LP
Matrix initialize_tableau(const Matrix& A, const Vector& b, const Vector& c) {
    size_t m = A.size();    // Number of constraints
    size_t n = A[0].size(); // Number of variables
    
    // Create tableau with dimensions (m+1) x (n+1)
    Matrix tableau(m + 1, Vector(n + 1, 0.0));
    
    // Fill in the constraint coefficients and RHS
    for (size_t i = 0; i < m; ++i) {
        for (size_t j = 0; j < n; ++j) {
            tableau[i][j] = A[i][j];
        }
        tableau[i][n] = b[i]; // RHS
    }
    
    // Fill in the objective function coefficients (negated)
    for (size_t j = 0; j < n; ++j) {
        tableau[m][j] = -c[j];
    }
    // The bottom-right cell is the objective value (initially 0)
    tableau[m][n] = 0.0;
    
    return tableau;
}

// Find the pivot column (most negative coefficient in the objective row)
int find_pivot_column(const Matrix& tableau) {
    size_t n = tableau[0].size() - 1; // Number of variables
    int pivot_col = -1;
    double min_value = -1e-10; // Small negative threshold to account for numerical errors
    
    for (size_t j = 0; j < n; ++j) {
        if (tableau.back()[j] < min_value && tableau.back()[j] < 0) {
            min_value = tableau.back()[j];
            pivot_col = j;
        }
    }
    
    return pivot_col;
}

// Find the pivot row using the minimum ratio test
int find_pivot_row(const Matrix& tableau, int pivot_col) {
    size_t m = tableau.size() - 1; // Number of constraints
    int pivot_row = -1;
    double min_ratio = std::numeric_limits<double>::max();
    
    for (size_t i = 0; i < m; ++i) {
        if (tableau[i][pivot_col] > 1e-10) { // Positive coefficient
            double ratio = tableau[i].back() / tableau[i][pivot_col];
            if (ratio < min_ratio && ratio >= 0) {
                min_ratio = ratio;
                pivot_row = i;
            }
        }
    }
    
    return pivot_row;
}

// Perform pivot operation
void pivot(Matrix& tableau, int pivot_row, int pivot_col) {
    size_t m = tableau.size();
    size_t n = tableau[0].size();
    
    // Scale the pivot row so that the pivot element becomes 1
    double pivot_element = tableau[pivot_row][pivot_col];
    for (size_t j = 0; j < n; ++j) {
        tableau[pivot_row][j] /= pivot_element;
    }
    
    // Eliminate the pivot column elements in other rows
    for (size_t i = 0; i < m; ++i) {
        if (i != pivot_row) {
            double factor = tableau[i][pivot_col];
            for (size_t j = 0; j < n; ++j) {
                tableau[i][j] -= factor * tableau[pivot_row][j];
            }
        }
    }
}

// Simplex method for solving linear programs in standard form
Vector simplex_method(Matrix A, Vector b, Vector c, bool is_maximization = true) {
    // Convert to standard form
    convert_to_standard_form(A, b, c, is_maximization);
    
    size_t m = A.size();    // Number of constraints
    size_t n = A[0].size(); // Number of variables
    
    // Initialize tableau
    Matrix tableau = initialize_tableau(A, b, c);
    
    // Initialize basic and non-basic variable indices
    std::vector<int> basic_vars(m);
    std::vector<int> non_basic_vars(n);
    
    // Initially, slack variables are basic and original variables are non-basic
    for (size_t i = 0; i < m; ++i) {
        basic_vars[i] = n + i;
    }
    for (size_t j = 0; j < n; ++j) {
        non_basic_vars[j] = j;
    }
    
    std::cout << "Starting Simplex Method" << std::endl;
    std::cout << "======================" << std::endl;
    
    // Main simplex loop
    int iteration = 0;
    while (true) {
        std::cout << "\nIteration " << iteration << std::endl;
        print_tableau(tableau, basic_vars, non_basic_vars);
        
        // Find the pivot column
        int pivot_col = find_pivot_column(tableau);
        if (pivot_col == -1) {
            std::cout << "Optimal solution found!" << std::endl;
            break;
        }
        
        // Find the pivot row
        int pivot_row = find_pivot_row(tableau, pivot_col);
        if (pivot_row == -1) {
            std::cout << "Unbounded solution!" << std::endl;
            break;
        }
        
        std::cout << "Pivot: row=" << pivot_row << ", col=" << pivot_col << std::endl;
        
        // Update basic and non-basic variable sets
        int entering_var = non_basic_vars[pivot_col];
        int leaving_var = basic_vars[pivot_row];
        basic_vars[pivot_row] = entering_var;
        non_basic_vars[pivot_col] = leaving_var;
        
        // Perform pivot operation
        pivot(tableau, pivot_row, pivot_col);
        
        iteration++;
    }
    
    // Extract solution
    Vector solution(n, 0.0);
    for (size_t i = 0; i < m; ++i) {
        if (basic_vars[i] < n) {
            solution[basic_vars[i]] = tableau[i].back();
        }
    }
    
    double objective_value = tableau.back().back();
    if (!is_maximization) {
        objective_value = -objective_value;
    }
    
    std::cout << "Objective value: " << objective_value << std::endl;
    
    return solution;
}

// Two-phase simplex method for problems that need artificial variables
Vector two_phase_simplex(Matrix A, Vector b, Vector c, bool is_maximization = true) {
    // Convert to standard form
    convert_to_standard_form(A, b, c, is_maximization);
    
    size_t m = A.size();    // Number of constraints
    size_t n = A[0].size(); // Number of variables
    
    std::cout << "Starting Two-Phase Simplex Method" << std::endl;
    std::cout << "===============================" << std::endl;
    
    // Phase I: Find a basic feasible solution
    std::cout << "\nPhase I: Finding a basic feasible solution" << std::endl;
    
    // Create an auxiliary problem with artificial variables
    Matrix A_aux = A;
    Vector c_aux(n + m, 0.0);
    
    // Add artificial variables to A
    for (size_t i = 0; i < m; ++i) {
        A_aux[i].resize(n + m, 0.0);
        A_aux[i][n + i] = 1.0;  // Coefficient of artificial variable
        c_aux[n + i] = -1.0;    // Objective is to minimize sum of artificial variables
    }
    
    // Solve the auxiliary problem
    Vector aux_solution = simplex_method(A_aux, b, c_aux, false);
    
    // Check if the auxiliary objective is zero (feasible original problem)
    double aux_objective = 0.0;
    for (size_t i = n; i < n + m; ++i) {
        aux_objective += aux_solution[i];
    }
    
    if (std::abs(aux_objective) > 1e-10) {
        std::cout << "The original problem is infeasible!" << std::endl;
        return Vector(n, 0.0);
    }
    
    // Phase II: Solve the original problem
    std::cout << "\nPhase II: Solving the original problem" << std::endl;
    
    // Remove artificial variables and solve the original problem
    Matrix A_orig = A;
    Vector solution = simplex_method(A_orig, b, c, is_maximization);
    
    return solution;
}

int main() {
    std::cout << "Linear Programming Examples" << std::endl;
    std::cout << "==========================" << std::endl;
    
    // Example 1: Simple maximization problem
    // Maximize: 3x + 4y
    // Subject to:
    //   x + 2y <= 8
    //   3x + 2y <= 12
    //   x, y >= 0
    std::cout << "\nExample 1: Simple Maximization Problem" << std::endl;
    Matrix A1 = {
        {1.0, 2.0},
        {3.0, 2.0}
    };
    Vector b1 = {8.0, 12.0};
    Vector c1 = {3.0, 4.0};
    
    // Add slack variables to convert to standard form
    for (size_t i = 0; i < A1.size(); ++i) {
        for (size_t j = 0; j < A1.size(); ++j) {
            if (i == j) {
                A1[i].push_back(1.0);  // Slack variable coefficient
            } else {
                A1[i].push_back(0.0);
            }
        }
    }
    c1.resize(c1.size() + A1.size(), 0.0);  // Extend c with zeros for slack variables
    
    Vector solution1 = simplex_method(A1, b1, c1, true);
    std::cout << "Solution:" << std::endl;
    print_vector(solution1, "x");
    
    // Example 2: Minimization problem
    // Minimize: 2x + 3y + 4z
    // Subject to:
    //   3x + 2y + z >= 10
    //   2x + 5y + 3z >= 15
    //   x, y, z >= 0
    std::cout << "\nExample 2: Minimization Problem" << std::endl;
    Matrix A2 = {
        {3.0, 2.0, 1.0},
        {2.0, 5.0, 3.0}
    };
    Vector b2 = {10.0, 15.0};
    Vector c2 = {2.0, 3.0, 4.0};
    
    // Convert >= constraints to <= by multiplying by -1
    for (size_t i = 0; i < A2.size(); ++i) {
        for (size_t j = 0; j < A2[i].size(); ++j) {
            A2[i][j] = -A2[i][j];
        }
        b2[i] = -b2[i];
    }
    
    // Add slack variables
    for (size_t i = 0; i < A2.size(); ++i) {
        for (size_t j = 0; j < A2.size(); ++j) {
            if (i == j) {
                A2[i].push_back(1.0);
            } else {
                A2[i].push_back(0.0);
            }
        }
    }
    c2.resize(c2.size() + A2.size(), 0.0);
    
    Vector solution2 = simplex_method(A2, b2, c2, false);
    std::cout << "Solution:" << std::endl;
    print_vector(solution2, "x");
    
    // Example 3: Problem requiring two-phase simplex
    // Maximize: 2x + 3y
    // Subject to:
    //   x + y = 4
    //   2x + y <= 5
    //   x, y >= 0
    std::cout << "\nExample 3: Problem with Equality Constraint" << std::endl;
    Matrix A3 = {
        {1.0, 1.0},
        {2.0, 1.0}
    };
    Vector b3 = {4.0, 5.0};
    Vector c3 = {2.0, 3.0};
    
    // Convert equality to two inequalities
    // x + y <= 4
    // x + y >= 4 (or -x - y <= -4)
    Matrix A3_expanded = {
        {1.0, 1.0},
        {-1.0, -1.0},
        {2.0, 1.0}
    };
    Vector b3_expanded = {4.0, -4.0, 5.0};
    
    // Add slack variables
    for (size_t i = 0; i < A3_expanded.size(); ++i) {
        for (size_t j = 0; j < A3_expanded.size(); ++j) {
            if (i == j) {
                A3_expanded[i].push_back(1.0);
            } else {
                A3_expanded[i].push_back(0.0);
            }
        }
    }
    c3.resize(c3.size() + A3_expanded.size(), 0.0);
    
    Vector solution3 = two_phase_simplex(A3_expanded, b3_expanded, c3, true);
    std::cout << "Solution:" << std::endl;
    print_vector(solution3, "x");
    
    return 0;
}
