#include <iostream>
#include <vector>
#include <unordered_map>
#include <unordered_set>
#include <algorithm>
#include <functional>
#include <string>

/**
 * Constraint Satisfaction Problem (CSP) Implementation
 * 
 * This file demonstrates the implementation of constraint satisfaction problems
 * including basic CSP solving algorithms like backtracking with constraint propagation.
 */

// Forward declarations
class Variable;
class Constraint;
class CSP;

// Domain type - set of possible values for a variable
using Domain = std::unordered_set<int>;

// Variable class representing a CSP variable
class Variable {
public:
    std::string name;
    Domain domain;
    int value;
    bool assigned;
    
    Variable(const std::string& n, const Domain& d) 
        : name(n), domain(d), value(-1), assigned(false) {}
    
    void assign(int val) {
        if (domain.find(val) != domain.end()) {
            value = val;
            assigned = true;
        }
    }
    
    void unassign() {
        value = -1;
        assigned = false;
    }
    
    bool is_consistent_with(int val) const {
        return domain.find(val) != domain.end();
    }
    
    void print() const {
        std::cout << name << ": ";
        if (assigned) {
            std::cout << value;
        } else {
            std::cout << "unassigned, domain = {";
            bool first = true;
            for (int val : domain) {
                if (!first) std::cout << ", ";
                std::cout << val;
                first = false;
            }
            std::cout << "}";
        }
        std::cout << std::endl;
    }
};

// Abstract base class for constraints
class Constraint {
public:
    std::vector<Variable*> variables;
    
    Constraint(const std::vector<Variable*>& vars) : variables(vars) {}
    virtual ~Constraint() = default;
    
    virtual bool is_satisfied() const = 0;
    virtual bool is_consistent() const = 0;
    virtual std::string get_name() const = 0;
    
    // Check if constraint involves a specific variable
    bool involves(Variable* var) const {
        return std::find(variables.begin(), variables.end(), var) != variables.end();
    }
};

// Binary constraint: two variables must have different values
class AllDifferentConstraint : public Constraint {
public:
    AllDifferentConstraint(Variable* var1, Variable* var2) 
        : Constraint({var1, var2}) {}
    
    bool is_satisfied() const override {
        if (!variables[0]->assigned || !variables[1]->assigned) {
            return true; // Not yet violated
        }
        return variables[0]->value != variables[1]->value;
    }
    
    bool is_consistent() const override {
        // Check if there's still a way to satisfy this constraint
        if (variables[0]->assigned && variables[1]->assigned) {
            return variables[0]->value != variables[1]->value;
        }
        
        if (variables[0]->assigned) {
            return variables[1]->domain.find(variables[0]->value) != variables[1]->domain.end() ? 
                   variables[1]->domain.size() > 1 : true;
        }
        
        if (variables[1]->assigned) {
            return variables[0]->domain.find(variables[1]->value) != variables[0]->domain.end() ? 
                   variables[0]->domain.size() > 1 : true;
        }
        
        return true; // Both unassigned
    }
    
    std::string get_name() const override {
        return variables[0]->name + " != " + variables[1]->name;
    }
};

// Arithmetic constraint: var1 + var2 = sum
class SumConstraint : public Constraint {
private:
    int target_sum;
    
public:
    SumConstraint(Variable* var1, Variable* var2, int sum) 
        : Constraint({var1, var2}), target_sum(sum) {}
    
    bool is_satisfied() const override {
        if (!variables[0]->assigned || !variables[1]->assigned) {
            return true; // Not yet violated
        }
        return variables[0]->value + variables[1]->value == target_sum;
    }
    
    bool is_consistent() const override {
        if (variables[0]->assigned && variables[1]->assigned) {
            return variables[0]->value + variables[1]->value == target_sum;
        }
        
        if (variables[0]->assigned) {
            int needed = target_sum - variables[0]->value;
            return variables[1]->domain.find(needed) != variables[1]->domain.end();
        }
        
        if (variables[1]->assigned) {
            int needed = target_sum - variables[1]->value;
            return variables[0]->domain.find(needed) != variables[0]->domain.end();
        }
        
        // Both unassigned - check if any combination works
        for (int val1 : variables[0]->domain) {
            int needed = target_sum - val1;
            if (variables[1]->domain.find(needed) != variables[1]->domain.end()) {
                return true;
            }
        }
        return false;
    }
    
    std::string get_name() const override {
        return variables[0]->name + " + " + variables[1]->name + " = " + std::to_string(target_sum);
    }
};

// CSP class that manages variables and constraints
class CSP {
private:
    std::vector<Variable> variables;
    std::vector<std::unique_ptr<Constraint>> constraints;
    
public:
    void add_variable(const std::string& name, const Domain& domain) {
        variables.emplace_back(name, domain);
    }
    
    Variable* get_variable(const std::string& name) {
        auto it = std::find_if(variables.begin(), variables.end(),
                              [&name](const Variable& v) { return v.name == name; });
        return it != variables.end() ? &(*it) : nullptr;
    }
    
    void add_constraint(std::unique_ptr<Constraint> constraint) {
        constraints.push_back(std::move(constraint));
    }
    
    bool is_complete() const {
        return std::all_of(variables.begin(), variables.end(),
                          [](const Variable& v) { return v.assigned; });
    }
    
    bool is_consistent() const {
        return std::all_of(constraints.begin(), constraints.end(),
                          [](const std::unique_ptr<Constraint>& c) { return c->is_consistent(); });
    }
    
    bool is_solution() const {
        return is_complete() && std::all_of(constraints.begin(), constraints.end(),
                                           [](const std::unique_ptr<Constraint>& c) { return c->is_satisfied(); });
    }
    
    Variable* select_unassigned_variable() {
        // MRV (Minimum Remaining Values) heuristic
        Variable* best = nullptr;
        size_t min_domain_size = SIZE_MAX;
        
        for (auto& var : variables) {
            if (!var.assigned && var.domain.size() < min_domain_size) {
                min_domain_size = var.domain.size();
                best = &var;
            }
        }
        return best;
    }
    
    std::vector<int> order_domain_values(Variable* var) {
        // For now, just return domain values in arbitrary order
        // Could implement LCV (Least Constraining Value) heuristic here
        std::vector<int> values(var->domain.begin(), var->domain.end());
        std::sort(values.begin(), values.end());
        return values;
    }
    
    // Forward checking: remove inconsistent values from unassigned variables
    bool forward_check(Variable* assigned_var) {
        for (auto& constraint : constraints) {
            if (!constraint->involves(assigned_var)) continue;
            
            for (auto* var : constraint->variables) {
                if (var->assigned || var == assigned_var) continue;
                
                // Check which values in var's domain are still consistent
                Domain new_domain;
                for (int val : var->domain) {
                    var->assign(val);
                    if (constraint->is_consistent()) {
                        new_domain.insert(val);
                    }
                    var->unassign();
                }
                
                if (new_domain.empty()) {
                    return false; // Domain wipeout
                }
                var->domain = new_domain;
            }
        }
        return true;
    }
    
    // Backtracking search with forward checking
    bool backtrack_search() {
        if (is_solution()) {
            return true;
        }
        
        Variable* var = select_unassigned_variable();
        if (!var) return false;
        
        std::vector<int> values = order_domain_values(var);
        
        for (int value : values) {
            // Save current domains for backtracking
            std::vector<Domain> saved_domains;
            for (auto& v : variables) {
                saved_domains.push_back(v.domain);
            }
            
            var->assign(value);
            
            if (is_consistent() && forward_check(var)) {
                if (backtrack_search()) {
                    return true;
                }
            }
            
            // Backtrack
            var->unassign();
            for (size_t i = 0; i < variables.size(); ++i) {
                variables[i].domain = saved_domains[i];
            }
        }
        
        return false;
    }
    
    void print_state() const {
        std::cout << "Current CSP State:" << std::endl;
        std::cout << "Variables:" << std::endl;
        for (const auto& var : variables) {
            std::cout << "  ";
            var.print();
        }
        
        std::cout << "Constraints:" << std::endl;
        for (const auto& constraint : constraints) {
            std::cout << "  " << constraint->get_name() << std::endl;
        }
        std::cout << std::endl;
    }
    
    void print_solution() const {
        if (is_solution()) {
            std::cout << "Solution found:" << std::endl;
            for (const auto& var : variables) {
                std::cout << "  " << var.name << " = " << var.value << std::endl;
            }
        } else {
            std::cout << "No solution exists or problem not solved." << std::endl;
        }
    }
};

// Example problems
void solve_n_queens(int n) {
    std::cout << "Solving " << n << "-Queens Problem" << std::endl;
    std::cout << "==============================" << std::endl;
    
    CSP csp;
    Domain domain;
    for (int i = 1; i <= n; ++i) {
        domain.insert(i);
    }
    
    // Add variables (one for each column, value represents row)
    for (int i = 1; i <= n; ++i) {
        csp.add_variable("Q" + std::to_string(i), domain);
    }
    
    // Add constraints
    for (int i = 1; i <= n; ++i) {
        for (int j = i + 1; j <= n; ++j) {
            Variable* qi = csp.get_variable("Q" + std::to_string(i));
            Variable* qj = csp.get_variable("Q" + std::to_string(j));
            
            // No two queens in same row
            csp.add_constraint(std::make_unique<AllDifferentConstraint>(qi, qj));
            
            // No two queens on same diagonal
            // This would require a custom diagonal constraint - simplified here
        }
    }
    
    csp.print_state();
    
    if (csp.backtrack_search()) {
        csp.print_solution();
    } else {
        std::cout << "No solution found!" << std::endl;
    }
    std::cout << std::endl;
}

void solve_map_coloring() {
    std::cout << "Solving Map Coloring Problem" << std::endl;
    std::cout << "============================" << std::endl;
    
    CSP csp;
    Domain colors = {1, 2, 3}; // Red=1, Green=2, Blue=3
    
    // Add variables (regions)
    csp.add_variable("WA", colors);  // Western Australia
    csp.add_variable("NT", colors);  // Northern Territory
    csp.add_variable("SA", colors);  // South Australia
    csp.add_variable("Q", colors);   // Queensland
    csp.add_variable("NSW", colors); // New South Wales
    csp.add_variable("V", colors);   // Victoria
    csp.add_variable("T", colors);   // Tasmania
    
    // Add adjacency constraints (adjacent regions must have different colors)
    std::vector<std::pair<std::string, std::string>> adjacencies = {
        {"WA", "NT"}, {"WA", "SA"}, {"NT", "SA"}, {"NT", "Q"},
        {"SA", "Q"}, {"SA", "NSW"}, {"SA", "V"}, {"Q", "NSW"}, {"NSW", "V"}
    };
    
    for (const auto& adj : adjacencies) {
        Variable* var1 = csp.get_variable(adj.first);
        Variable* var2 = csp.get_variable(adj.second);
        csp.add_constraint(std::make_unique<AllDifferentConstraint>(var1, var2));
    }
    
    csp.print_state();
    
    if (csp.backtrack_search()) {
        csp.print_solution();
        std::cout << "Color mapping: 1=Red, 2=Green, 3=Blue" << std::endl;
    } else {
        std::cout << "No solution found!" << std::endl;
    }
    std::cout << std::endl;
}

void solve_sudoku_subset() {
    std::cout << "Solving 4x4 Sudoku Subset" << std::endl;
    std::cout << "=========================" << std::endl;
    
    CSP csp;
    Domain domain = {1, 2, 3, 4};
    
    // Add variables for a 4x4 grid
    for (int i = 1; i <= 4; ++i) {
        for (int j = 1; j <= 4; ++j) {
            csp.add_variable("X" + std::to_string(i) + std::to_string(j), domain);
        }
    }
    
    // Add row constraints
    for (int i = 1; i <= 4; ++i) {
        for (int j = 1; j <= 4; ++j) {
            for (int k = j + 1; k <= 4; ++k) {
                Variable* var1 = csp.get_variable("X" + std::to_string(i) + std::to_string(j));
                Variable* var2 = csp.get_variable("X" + std::to_string(i) + std::to_string(k));
                csp.add_constraint(std::make_unique<AllDifferentConstraint>(var1, var2));
            }
        }
    }
    
    // Add column constraints
    for (int j = 1; j <= 4; ++j) {
        for (int i = 1; i <= 4; ++i) {
            for (int k = i + 1; k <= 4; ++k) {
                Variable* var1 = csp.get_variable("X" + std::to_string(i) + std::to_string(j));
                Variable* var2 = csp.get_variable("X" + std::to_string(k) + std::to_string(j));
                csp.add_constraint(std::make_unique<AllDifferentConstraint>(var1, var2));
            }
        }
    }
    
    // Pre-assign some values to make it interesting
    csp.get_variable("X11")->assign(1);
    csp.get_variable("X22")->assign(2);
    
    csp.print_state();
    
    if (csp.backtrack_search()) {
        csp.print_solution();
        
        // Print as grid
        std::cout << "Grid representation:" << std::endl;
        for (int i = 1; i <= 4; ++i) {
            for (int j = 1; j <= 4; ++j) {
                Variable* var = csp.get_variable("X" + std::to_string(i) + std::to_string(j));
                std::cout << var->value << " ";
            }
            std::cout << std::endl;
        }
    } else {
        std::cout << "No solution found!" << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "Constraint Satisfaction Problem Examples" << std::endl;
    std::cout << "=======================================" << std::endl << std::endl;
    
    // Example 1: Map coloring problem
    solve_map_coloring();
    
    // Example 2: 4-Queens problem (simplified)
    solve_n_queens(4);
    
    // Example 3: 4x4 Sudoku subset
    solve_sudoku_subset();
    
    return 0;
}
