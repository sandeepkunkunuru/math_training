#include <iostream>
#include <vector>
#include <queue>
#include <limits>
#include <algorithm>
#include <iomanip>
#include <memory>
#include <cmath>

/**
 * Branch and Bound Algorithm for Integer Linear Programming
 * 
 * This file demonstrates the implementation of the branch and bound algorithm
 * for solving integer linear programming problems.
 */

// Type definitions
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;

// Structure to represent a node in the branch and bound tree
struct BranchNode {
    Matrix A;                    // Constraint matrix
    Vector b;                    // Right-hand side
    Vector c;                    // Objective coefficients
    Vector lower_bounds;         // Variable lower bounds
    Vector upper_bounds;         // Variable upper bounds
    std::vector<bool> is_integer; // Which variables must be integer
    double objective_value;      // LP relaxation objective value
    Vector solution;             // LP relaxation solution
    int depth;                   // Depth in the branch and bound tree
    int branching_variable;      // Variable that was branched on
    double branching_value;      // Value used for branching
    bool is_feasible;           // Whether LP relaxation is feasible
    
    BranchNode() : objective_value(-std::numeric_limits<double>::infinity()), 
                   depth(0), branching_variable(-1), branching_value(0.0), 
                   is_feasible(false) {}
};

// Simple LP solver using a basic implementation (for demonstration)
// In practice, you would use a more sophisticated LP solver
class SimplexSolver {
private:
    static const double EPSILON;
    
public:
    struct LPResult {
        bool is_feasible;
        bool is_unbounded;
        double objective_value;
        Vector solution;
        
        LPResult() : is_feasible(false), is_unbounded(false), 
                    objective_value(-std::numeric_limits<double>::infinity()) {}
    };
    
    // Solve LP: maximize c^T x subject to Ax <= b, lower_bounds <= x <= upper_bounds
    static LPResult solve_lp(const Matrix& A, const Vector& b, const Vector& c,
                            const Vector& lower_bounds, const Vector& upper_bounds) {
        LPResult result;
        
        // This is a simplified LP solver for demonstration
        // In practice, use a robust LP solver like the simplex method
        size_t n = c.size();
        size_t m = A.size();
        
        // For this example, we'll use a simple feasibility check and
        // approximate solution for demonstration purposes
        Vector x(n, 0.0);
        
        // Simple heuristic: try to satisfy constraints greedily
        for (size_t i = 0; i < n; ++i) {
            x[i] = std::max(lower_bounds[i], std::min(upper_bounds[i], 1.0));
        }
        
        // Check feasibility
        bool feasible = true;
        for (size_t i = 0; i < m; ++i) {
            double lhs = 0.0;
            for (size_t j = 0; j < n; ++j) {
                lhs += A[i][j] * x[j];
            }
            if (lhs > b[i] + EPSILON) {
                feasible = false;
                break;
            }
        }
        
        if (feasible) {
            result.is_feasible = true;
            result.solution = x;
            result.objective_value = 0.0;
            for (size_t i = 0; i < n; ++i) {
                result.objective_value += c[i] * x[i];
            }
        }
        
        return result;
    }
};

const double SimplexSolver::EPSILON = 1e-9;

// Branch and Bound solver for Integer Linear Programming
class BranchAndBoundSolver {
private:
    double best_integer_value;
    Vector best_integer_solution;
    bool has_integer_solution;
    int nodes_explored;
    int max_nodes;
    
    // Priority queue comparator (maximize objective value)
    struct NodeComparator {
        bool operator()(const std::shared_ptr<BranchNode>& a, 
                       const std::shared_ptr<BranchNode>& b) const {
            return a->objective_value < b->objective_value; // Max heap
        }
    };
    
public:
    BranchAndBoundSolver() : best_integer_value(-std::numeric_limits<double>::infinity()),
                            has_integer_solution(false), nodes_explored(0), max_nodes(1000) {}
    
    // Check if a solution satisfies integer constraints
    bool is_integer_feasible(const Vector& solution, const std::vector<bool>& is_integer) const {
        for (size_t i = 0; i < solution.size(); ++i) {
            if (is_integer[i] && std::abs(solution[i] - std::round(solution[i])) > SimplexSolver::EPSILON) {
                return false;
            }
        }
        return true;
    }
    
    // Find the most fractional variable for branching
    int select_branching_variable(const Vector& solution, const std::vector<bool>& is_integer) const {
        int best_var = -1;
        double max_fractionality = 0.0;
        
        for (size_t i = 0; i < solution.size(); ++i) {
            if (is_integer[i]) {
                double fractional_part = std::abs(solution[i] - std::round(solution[i]));
                if (fractional_part > max_fractionality) {
                    max_fractionality = fractional_part;
                    best_var = i;
                }
            }
        }
        
        return best_var;
    }
    
    // Create child nodes by branching on a variable
    std::pair<std::shared_ptr<BranchNode>, std::shared_ptr<BranchNode>> 
    create_child_nodes(const BranchNode& parent, int var_index) const {
        double var_value = parent.solution[var_index];
        
        // Left child: x[var_index] <= floor(var_value)
        auto left_child = std::make_shared<BranchNode>();
        *left_child = parent;
        left_child->upper_bounds[var_index] = std::floor(var_value);
        left_child->depth = parent.depth + 1;
        left_child->branching_variable = var_index;
        left_child->branching_value = std::floor(var_value);
        
        // Right child: x[var_index] >= ceil(var_value)
        auto right_child = std::make_shared<BranchNode>();
        *right_child = parent;
        right_child->lower_bounds[var_index] = std::ceil(var_value);
        right_child->depth = parent.depth + 1;
        right_child->branching_variable = var_index;
        right_child->branching_value = std::ceil(var_value);
        
        return {left_child, right_child};
    }
    
    // Solve the integer linear program using branch and bound
    bool solve(const Matrix& A, const Vector& b, const Vector& c,
              const Vector& lower_bounds, const Vector& upper_bounds,
              const std::vector<bool>& is_integer) {
        
        std::cout << "Starting Branch and Bound Algorithm" << std::endl;
        std::cout << "===================================" << std::endl;
        
        // Initialize the root node
        auto root = std::make_shared<BranchNode>();
        root->A = A;
        root->b = b;
        root->c = c;
        root->lower_bounds = lower_bounds;
        root->upper_bounds = upper_bounds;
        root->is_integer = is_integer;
        
        // Solve LP relaxation at root
        auto lp_result = SimplexSolver::solve_lp(A, b, c, lower_bounds, upper_bounds);
        if (!lp_result.is_feasible) {
            std::cout << "Root LP relaxation is infeasible!" << std::endl;
            return false;
        }
        
        root->is_feasible = true;
        root->objective_value = lp_result.objective_value;
        root->solution = lp_result.solution;
        
        std::cout << "Root LP relaxation value: " << root->objective_value << std::endl;
        
        // Initialize priority queue with root node
        std::priority_queue<std::shared_ptr<BranchNode>, 
                           std::vector<std::shared_ptr<BranchNode>>, 
                           NodeComparator> node_queue;
        node_queue.push(root);
        
        // Main branch and bound loop
        while (!node_queue.empty() && nodes_explored < max_nodes) {
            auto current_node = node_queue.top();
            node_queue.pop();
            nodes_explored++;
            
            std::cout << "\nExploring node " << nodes_explored 
                     << " (depth " << current_node->depth 
                     << ", obj = " << std::fixed << std::setprecision(4) 
                     << current_node->objective_value << ")" << std::endl;
            
            // Pruning: if current node's bound is worse than best integer solution
            if (has_integer_solution && current_node->objective_value <= best_integer_value + SimplexSolver::EPSILON) {
                std::cout << "  Pruned by bound" << std::endl;
                continue;
            }
            
            // Check if current solution is integer feasible
            if (is_integer_feasible(current_node->solution, current_node->is_integer)) {
                std::cout << "  Integer feasible solution found!" << std::endl;
                if (!has_integer_solution || current_node->objective_value > best_integer_value) {
                    best_integer_value = current_node->objective_value;
                    best_integer_solution = current_node->solution;
                    has_integer_solution = true;
                    std::cout << "  New best integer solution: " << best_integer_value << std::endl;
                }
                continue;
            }
            
            // Select branching variable
            int branch_var = select_branching_variable(current_node->solution, current_node->is_integer);
            if (branch_var == -1) {
                std::cout << "  No fractional integer variables found" << std::endl;
                continue;
            }
            
            std::cout << "  Branching on variable " << branch_var 
                     << " (value = " << current_node->solution[branch_var] << ")" << std::endl;
            
            // Create child nodes
            auto children = create_child_nodes(*current_node, branch_var);
            
            // Solve LP relaxations for child nodes and add to queue
            for (auto& child : {children.first, children.second}) {
                auto child_lp = SimplexSolver::solve_lp(child->A, child->b, child->c,
                                                       child->lower_bounds, child->upper_bounds);
                
                if (child_lp.is_feasible) {
                    child->is_feasible = true;
                    child->objective_value = child_lp.objective_value;
                    child->solution = child_lp.solution;
                    
                    // Only add to queue if not pruned by bound
                    if (!has_integer_solution || child->objective_value > best_integer_value + SimplexSolver::EPSILON) {
                        node_queue.push(child);
                    }
                }
            }
        }
        
        std::cout << "\nBranch and Bound completed!" << std::endl;
        std::cout << "Nodes explored: " << nodes_explored << std::endl;
        
        return has_integer_solution;
    }
    
    void print_solution() const {
        if (has_integer_solution) {
            std::cout << "\nOptimal integer solution found:" << std::endl;
            std::cout << "Objective value: " << std::fixed << std::setprecision(6) 
                     << best_integer_value << std::endl;
            std::cout << "Solution vector: [";
            for (size_t i = 0; i < best_integer_solution.size(); ++i) {
                std::cout << std::fixed << std::setprecision(4) << best_integer_solution[i];
                if (i < best_integer_solution.size() - 1) std::cout << ", ";
            }
            std::cout << "]" << std::endl;
        } else {
            std::cout << "\nNo integer solution found!" << std::endl;
        }
    }
    
    double get_objective_value() const { return best_integer_value; }
    Vector get_solution() const { return best_integer_solution; }
    bool has_solution() const { return has_integer_solution; }
};

// Example problems
void solve_integer_knapsack() {
    std::cout << "Integer Knapsack Problem" << std::endl;
    std::cout << "========================" << std::endl;
    
    // Maximize: 3x1 + 4x2 + 2x3
    // Subject to: 2x1 + 3x2 + x3 <= 5
    //            x1, x2, x3 ∈ {0, 1}
    
    Matrix A = {{2.0, 3.0, 1.0}};
    Vector b = {5.0};
    Vector c = {3.0, 4.0, 2.0};
    Vector lower_bounds = {0.0, 0.0, 0.0};
    Vector upper_bounds = {1.0, 1.0, 1.0};
    std::vector<bool> is_integer = {true, true, true};
    
    BranchAndBoundSolver solver;
    if (solver.solve(A, b, c, lower_bounds, upper_bounds, is_integer)) {
        solver.print_solution();
    }
    std::cout << std::endl;
}

void solve_mixed_integer_problem() {
    std::cout << "Mixed Integer Programming Problem" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Maximize: 2x1 + 3x2
    // Subject to: x1 + 2x2 <= 3
    //            2x1 + x2 <= 3
    //            x1 ∈ Z+, x2 ∈ R+
    
    Matrix A = {
        {1.0, 2.0},
        {2.0, 1.0}
    };
    Vector b = {3.0, 3.0};
    Vector c = {2.0, 3.0};
    Vector lower_bounds = {0.0, 0.0};
    Vector upper_bounds = {10.0, 10.0}; // Large upper bounds
    std::vector<bool> is_integer = {true, false}; // x1 integer, x2 continuous
    
    BranchAndBoundSolver solver;
    if (solver.solve(A, b, c, lower_bounds, upper_bounds, is_integer)) {
        solver.print_solution();
    }
    std::cout << std::endl;
}

void solve_facility_location() {
    std::cout << "Simple Facility Location Problem" << std::endl;
    std::cout << "=================================" << std::endl;
    
    // Minimize: 10y1 + 15y2 + 3x11 + 5x12 + 4x21 + 2x22
    // Subject to: x11 + x12 = 1  (demand point 1)
    //            x21 + x22 = 1  (demand point 2)
    //            x11 + x21 <= 2*y1  (facility 1 capacity)
    //            x12 + x22 <= 2*y2  (facility 2 capacity)
    //            y1, y2 ∈ {0, 1}, x_ij >= 0
    
    // Convert to maximization by negating objective
    Matrix A = {
        {1.0, 1.0, 0.0, 0.0, 0.0, 0.0},  // x11 + x12 <= 1
        {0.0, 0.0, 1.0, 1.0, 0.0, 0.0},  // x21 + x22 <= 1
        {-1.0, 0.0, -1.0, 0.0, 2.0, 0.0}, // -x11 - x21 + 2*y1 >= 0
        {0.0, -1.0, 0.0, -1.0, 0.0, 2.0}  // -x12 - x22 + 2*y2 >= 0
    };
    Vector b = {1.0, 1.0, 0.0, 0.0};
    Vector c = {-3.0, -5.0, -4.0, -2.0, -10.0, -15.0}; // Negated for maximization
    Vector lower_bounds = {0.0, 0.0, 0.0, 0.0, 0.0, 0.0};
    Vector upper_bounds = {1.0, 1.0, 1.0, 1.0, 1.0, 1.0};
    std::vector<bool> is_integer = {false, false, false, false, true, true};
    
    BranchAndBoundSolver solver;
    if (solver.solve(A, b, c, lower_bounds, upper_bounds, is_integer)) {
        solver.print_solution();
        std::cout << "Note: Objective was negated for maximization. True minimum = " 
                 << -solver.get_objective_value() << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "Branch and Bound Algorithm Examples" << std::endl;
    std::cout << "===================================" << std::endl << std::endl;
    
    // Example 1: Integer knapsack problem
    solve_integer_knapsack();
    
    // Example 2: Mixed integer programming
    solve_mixed_integer_problem();
    
    // Example 3: Simple facility location
    solve_facility_location();
    
    return 0;
}
