#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <algorithm>
#include <chrono>
#include <iomanip>
#include <functional>

/**
 * Simulated Annealing Implementation
 * 
 * This file demonstrates simulated annealing for various optimization problems
 * including TSP, function optimization, and combinatorial problems.
 */

// Base class for optimization problems
template<typename Solution>
class OptimizationProblem {
public:
    virtual ~OptimizationProblem() = default;
    virtual double evaluate(const Solution& solution) const = 0;
    virtual Solution generate_neighbor(const Solution& current, std::mt19937& rng) const = 0;
    virtual Solution generate_random_solution(std::mt19937& rng) const = 0;
    virtual void print_solution(const Solution& solution) const = 0;
    virtual bool is_minimization() const { return true; }
};

// Simulated Annealing Algorithm
template<typename Solution>
class SimulatedAnnealing {
private:
    const OptimizationProblem<Solution>& problem;
    std::mt19937 rng;
    
    // Cooling schedule parameters
    double initial_temperature;
    double final_temperature;
    double cooling_rate;
    int max_iterations;
    int iterations_per_temperature;
    
public:
    struct Result {
        Solution best_solution;
        double best_value;
        std::vector<double> temperature_history;
        std::vector<double> value_history;
        int total_iterations;
        int accepted_moves;
        int rejected_moves;
    };
    
    SimulatedAnnealing(const OptimizationProblem<Solution>& prob, 
                      double init_temp = 1000.0,
                      double final_temp = 0.01,
                      double cool_rate = 0.95,
                      int max_iter = 10000,
                      int iter_per_temp = 100)
        : problem(prob), rng(std::chrono::steady_clock::now().time_since_epoch().count()),
          initial_temperature(init_temp), final_temperature(final_temp),
          cooling_rate(cool_rate), max_iterations(max_iter),
          iterations_per_temperature(iter_per_temp) {}
    
    Result solve() {
        std::cout << "Running Simulated Annealing" << std::endl;
        std::cout << "===========================" << std::endl;
        
        Result result;
        
        // Initialize with random solution
        Solution current = problem.generate_random_solution(rng);
        double current_value = problem.evaluate(current);
        
        result.best_solution = current;
        result.best_value = current_value;
        result.total_iterations = 0;
        result.accepted_moves = 0;
        result.rejected_moves = 0;
        
        double temperature = initial_temperature;
        std::uniform_real_distribution<> uniform(0.0, 1.0);
        
        std::cout << "Initial solution value: " << std::fixed << std::setprecision(4) 
                 << current_value << std::endl;
        std::cout << "Initial temperature: " << initial_temperature << std::endl;
        
        while (temperature > final_temperature && result.total_iterations < max_iterations) {
            for (int i = 0; i < iterations_per_temperature && result.total_iterations < max_iterations; ++i) {
                // Generate neighbor
                Solution neighbor = problem.generate_neighbor(current, rng);
                double neighbor_value = problem.evaluate(neighbor);
                
                // Calculate acceptance probability
                double delta = neighbor_value - current_value;
                if (!problem.is_minimization()) {
                    delta = -delta; // For maximization problems
                }
                
                bool accept = false;
                if (delta < 0) {
                    // Better solution - always accept
                    accept = true;
                } else {
                    // Worse solution - accept with probability
                    double probability = std::exp(-delta / temperature);
                    accept = uniform(rng) < probability;
                }
                
                if (accept) {
                    current = neighbor;
                    current_value = neighbor_value;
                    result.accepted_moves++;
                    
                    // Update best solution
                    bool is_better = problem.is_minimization() ? 
                                   (neighbor_value < result.best_value) :
                                   (neighbor_value > result.best_value);
                    if (is_better) {
                        result.best_solution = neighbor;
                        result.best_value = neighbor_value;
                    }
                } else {
                    result.rejected_moves++;
                }
                
                result.total_iterations++;
                
                // Record history
                if (result.total_iterations % 100 == 0) {
                    result.temperature_history.push_back(temperature);
                    result.value_history.push_back(result.best_value);
                }
            }
            
            // Cool down
            temperature *= cooling_rate;
        }
        
        std::cout << "Final temperature: " << std::fixed << std::setprecision(6) 
                 << temperature << std::endl;
        std::cout << "Total iterations: " << result.total_iterations << std::endl;
        std::cout << "Accepted moves: " << result.accepted_moves << std::endl;
        std::cout << "Rejected moves: " << result.rejected_moves << std::endl;
        std::cout << "Acceptance rate: " << std::fixed << std::setprecision(2) 
                 << (100.0 * result.accepted_moves / result.total_iterations) << "%" << std::endl;
        std::cout << "Best solution value: " << std::fixed << std::setprecision(4) 
                 << result.best_value << std::endl;
        
        return result;
    }
};

// TSP Problem for Simulated Annealing
class TSPProblem : public OptimizationProblem<std::vector<int>> {
private:
    std::vector<std::vector<double>> distance_matrix;
    int n;
    
public:
    TSPProblem(const std::vector<std::vector<double>>& distances) 
        : distance_matrix(distances), n(distances.size()) {}
    
    double evaluate(const std::vector<int>& tour) const override {
        double total_distance = 0.0;
        for (int i = 0; i < n; ++i) {
            int from = tour[i];
            int to = tour[(i + 1) % n];
            total_distance += distance_matrix[from][to];
        }
        return total_distance;
    }
    
    std::vector<int> generate_neighbor(const std::vector<int>& current, std::mt19937& rng) const override {
        std::vector<int> neighbor = current;
        std::uniform_int_distribution<> dist(0, n - 1);
        
        // 2-opt move: reverse a segment
        int i = dist(rng);
        int j = dist(rng);
        if (i > j) std::swap(i, j);
        
        std::reverse(neighbor.begin() + i, neighbor.begin() + j + 1);
        return neighbor;
    }
    
    std::vector<int> generate_random_solution(std::mt19937& rng) const override {
        std::vector<int> solution(n);
        std::iota(solution.begin(), solution.end(), 0);
        std::shuffle(solution.begin(), solution.end(), rng);
        return solution;
    }
    
    void print_solution(const std::vector<int>& solution) const override {
        std::cout << "Tour: ";
        for (int i = 0; i < n; ++i) {
            std::cout << solution[i];
            if (i < n - 1) std::cout << " -> ";
        }
        std::cout << " -> " << solution[0] << std::endl;
        std::cout << "Distance: " << std::fixed << std::setprecision(4) 
                 << evaluate(solution) << std::endl;
    }
};

// Function Optimization Problem
class FunctionOptimization : public OptimizationProblem<std::vector<double>> {
private:
    std::function<double(const std::vector<double>&)> objective_function;
    std::vector<std::pair<double, double>> bounds;
    double step_size;
    
public:
    FunctionOptimization(std::function<double(const std::vector<double>&)> func,
                        const std::vector<std::pair<double, double>>& bounds,
                        double step = 0.1)
        : objective_function(func), bounds(bounds), step_size(step) {}
    
    double evaluate(const std::vector<double>& solution) const override {
        return objective_function(solution);
    }
    
    std::vector<double> generate_neighbor(const std::vector<double>& current, std::mt19937& rng) const override {
        std::vector<double> neighbor = current;
        std::normal_distribution<> normal(0.0, step_size);
        std::uniform_int_distribution<> var_dist(0, current.size() - 1);
        
        // Perturb one random variable
        int var_index = var_dist(rng);
        neighbor[var_index] += normal(rng);
        
        // Ensure bounds
        neighbor[var_index] = std::max(bounds[var_index].first, 
                                     std::min(bounds[var_index].second, neighbor[var_index]));
        
        return neighbor;
    }
    
    std::vector<double> generate_random_solution(std::mt19937& rng) const override {
        std::vector<double> solution;
        for (const auto& bound : bounds) {
            std::uniform_real_distribution<> dist(bound.first, bound.second);
            solution.push_back(dist(rng));
        }
        return solution;
    }
    
    void print_solution(const std::vector<double>& solution) const override {
        std::cout << "Solution: [";
        for (size_t i = 0; i < solution.size(); ++i) {
            std::cout << std::fixed << std::setprecision(4) << solution[i];
            if (i < solution.size() - 1) std::cout << ", ";
        }
        std::cout << "]" << std::endl;
        std::cout << "Value: " << std::fixed << std::setprecision(6) 
                 << evaluate(solution) << std::endl;
    }
};

// Knapsack Problem
struct Item {
    int weight;
    int value;
    Item(int w, int v) : weight(w), value(v) {}
};

class KnapsackProblem : public OptimizationProblem<std::vector<bool>> {
private:
    std::vector<Item> items;
    int capacity;
    
public:
    KnapsackProblem(const std::vector<Item>& items, int cap) 
        : items(items), capacity(cap) {}
    
    bool is_minimization() const override { return false; } // Maximization problem
    
    double evaluate(const std::vector<bool>& solution) const override {
        int total_weight = 0;
        int total_value = 0;
        
        for (size_t i = 0; i < solution.size(); ++i) {
            if (solution[i]) {
                total_weight += items[i].weight;
                total_value += items[i].value;
            }
        }
        
        // Penalty for exceeding capacity
        if (total_weight > capacity) {
            return total_value - 1000.0 * (total_weight - capacity);
        }
        
        return total_value;
    }
    
    std::vector<bool> generate_neighbor(const std::vector<bool>& current, std::mt19937& rng) const override {
        std::vector<bool> neighbor = current;
        std::uniform_int_distribution<> dist(0, items.size() - 1);
        
        // Flip one random bit
        int index = dist(rng);
        neighbor[index] = !neighbor[index];
        
        return neighbor;
    }
    
    std::vector<bool> generate_random_solution(std::mt19937& rng) const override {
        std::vector<bool> solution(items.size());
        std::uniform_real_distribution<> prob(0.0, 1.0);
        
        for (size_t i = 0; i < items.size(); ++i) {
            solution[i] = prob(rng) < 0.5;
        }
        
        return solution;
    }
    
    void print_solution(const std::vector<bool>& solution) const override {
        int total_weight = 0;
        int total_value = 0;
        
        std::cout << "Selected items: ";
        bool first = true;
        for (size_t i = 0; i < solution.size(); ++i) {
            if (solution[i]) {
                if (!first) std::cout << ", ";
                std::cout << i;
                total_weight += items[i].weight;
                total_value += items[i].value;
                first = false;
            }
        }
        std::cout << std::endl;
        std::cout << "Total weight: " << total_weight << "/" << capacity << std::endl;
        std::cout << "Total value: " << total_value << std::endl;
    }
};

// Example problems
void solve_tsp_example() {
    std::cout << "=== TSP EXAMPLE ===" << std::endl;
    
    // Small TSP instance
    std::vector<std::vector<double>> distances = {
        {0, 10, 15, 20},
        {10, 0, 35, 25},
        {15, 35, 0, 30},
        {20, 25, 30, 0}
    };
    
    TSPProblem tsp(distances);
    SimulatedAnnealing<std::vector<int>> sa(tsp, 100.0, 0.01, 0.95, 5000, 50);
    
    auto result = sa.solve();
    std::cout << "\nBest TSP solution found:" << std::endl;
    tsp.print_solution(result.best_solution);
    std::cout << std::endl;
}

void solve_function_optimization() {
    std::cout << "=== FUNCTION OPTIMIZATION EXAMPLE ===" << std::endl;
    
    // Minimize Rosenbrock function: f(x,y) = (1-x)^2 + 100(y-x^2)^2
    auto rosenbrock = [](const std::vector<double>& x) {
        double term1 = 1.0 - x[0];
        double term2 = x[1] - x[0] * x[0];
        return term1 * term1 + 100.0 * term2 * term2;
    };
    
    std::vector<std::pair<double, double>> bounds = {{-5.0, 5.0}, {-5.0, 5.0}};
    FunctionOptimization func_opt(rosenbrock, bounds, 0.2);
    
    SimulatedAnnealing<std::vector<double>> sa(func_opt, 10.0, 0.001, 0.99, 10000, 100);
    
    auto result = sa.solve();
    std::cout << "\nBest function optimization solution:" << std::endl;
    func_opt.print_solution(result.best_solution);
    std::cout << "Global minimum is at (1, 1) with value 0" << std::endl;
    std::cout << std::endl;
}

void solve_knapsack_example() {
    std::cout << "=== KNAPSACK EXAMPLE ===" << std::endl;
    
    std::vector<Item> items = {
        Item(10, 60), Item(20, 100), Item(30, 120),
        Item(40, 160), Item(50, 200), Item(15, 50),
        Item(25, 80), Item(35, 110)
    };
    int capacity = 100;
    
    std::cout << "Items (weight, value):" << std::endl;
    for (size_t i = 0; i < items.size(); ++i) {
        std::cout << "Item " << i << ": (" << items[i].weight 
                 << ", " << items[i].value << ")" << std::endl;
    }
    std::cout << "Capacity: " << capacity << std::endl << std::endl;
    
    KnapsackProblem knapsack(items, capacity);
    SimulatedAnnealing<std::vector<bool>> sa(knapsack, 50.0, 0.01, 0.95, 5000, 100);
    
    auto result = sa.solve();
    std::cout << "\nBest knapsack solution:" << std::endl;
    knapsack.print_solution(result.best_solution);
    std::cout << std::endl;
}

int main() {
    std::cout << "Simulated Annealing Examples" << std::endl;
    std::cout << "============================" << std::endl << std::endl;
    
    // Example 1: TSP
    solve_tsp_example();
    
    // Example 2: Function optimization
    solve_function_optimization();
    
    // Example 3: Knapsack problem
    solve_knapsack_example();
    
    return 0;
}
