#include <iostream>
#include <vector>
#include <cmath>
#include <random>
#include <algorithm>
#include <functional>
#include <iomanip>
#include <limits>
#include <chrono>

/**
 * Metaheuristics Implementation
 * 
 * This file demonstrates various metaheuristic optimization algorithms
 * including Simulated Annealing, Genetic Algorithms, and Particle Swarm Optimization.
 */

// Type definitions for clarity
using Vector = std::vector<double>;
using Matrix = std::vector<std::vector<double>>;
using Function = std::function<double(const Vector&)>;
using Neighborhood = std::function<Vector(const Vector&, std::mt19937&)>;

// Random number generator
std::mt19937 rng(std::chrono::system_clock::now().time_since_epoch().count());

// Print a vector
void print_vector(const Vector& v, const std::string& name = "Vector") {
    std::cout << name << ": [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Print iteration information
void print_iteration(int iter, const Vector& x, double f_x) {
    std::cout << "Iteration " << std::setw(4) << iter 
              << " | f(x) = " << std::setw(12) << std::fixed << std::setprecision(6) << f_x
              << " | x = [";
    
    for (size_t i = 0; i < x.size(); ++i) {
        std::cout << std::fixed << std::setprecision(6) << x[i];
        if (i < x.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Generate a random vector within bounds
Vector random_vector(const Vector& lower_bounds, const Vector& upper_bounds) {
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    Vector result(lower_bounds.size());
    for (size_t i = 0; i < lower_bounds.size(); ++i) {
        result[i] = lower_bounds[i] + dist(rng) * (upper_bounds[i] - lower_bounds[i]);
    }
    
    return result;
}

// Simulated Annealing algorithm
Vector simulated_annealing(
    const Function& objective,
    const Vector& initial_point,
    const Vector& lower_bounds,
    const Vector& upper_bounds,
    double initial_temp = 100.0,
    double cooling_rate = 0.95,
    int max_iterations = 1000,
    int iterations_per_temp = 50
) {
    Vector current_solution = initial_point;
    double current_energy = objective(current_solution);
    
    Vector best_solution = current_solution;
    double best_energy = current_energy;
    
    double temperature = initial_temp;
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    std::cout << "\nStarting Simulated Annealing" << std::endl;
    std::cout << "===========================" << std::endl;
    std::cout << "Initial temperature: " << initial_temp << std::endl;
    std::cout << "Cooling rate: " << cooling_rate << std::endl;
    
    int iteration = 0;
    
    // Main simulated annealing loop
    while (iteration < max_iterations) {
        for (int i = 0; i < iterations_per_temp; ++i) {
            // Generate a neighbor solution
            Vector neighbor_solution(current_solution.size());
            for (size_t j = 0; j < current_solution.size(); ++j) {
                // Perturb the current solution
                double perturbation = dist(rng) * temperature * (upper_bounds[j] - lower_bounds[j]) * 0.1;
                if (dist(rng) < 0.5) perturbation = -perturbation;
                
                neighbor_solution[j] = current_solution[j] + perturbation;
                
                // Ensure bounds are respected
                neighbor_solution[j] = std::max(neighbor_solution[j], lower_bounds[j]);
                neighbor_solution[j] = std::min(neighbor_solution[j], upper_bounds[j]);
            }
            
            // Evaluate the neighbor solution
            double neighbor_energy = objective(neighbor_solution);
            
            // Decide whether to accept the neighbor solution
            bool accept = false;
            
            if (neighbor_energy < current_energy) {
                // Always accept better solutions
                accept = true;
            } else {
                // Accept worse solutions with a probability that decreases with temperature
                double acceptance_probability = std::exp((current_energy - neighbor_energy) / temperature);
                if (dist(rng) < acceptance_probability) {
                    accept = true;
                }
            }
            
            // Update current solution if accepted
            if (accept) {
                current_solution = neighbor_solution;
                current_energy = neighbor_energy;
                
                // Update best solution if needed
                if (current_energy < best_energy) {
                    best_solution = current_solution;
                    best_energy = current_energy;
                }
            }
            
            // Print iteration information periodically
            if (iteration % 50 == 0) {
                print_iteration(iteration, best_solution, best_energy);
            }
            
            iteration++;
            if (iteration >= max_iterations) break;
        }
        
        // Cool down the temperature
        temperature *= cooling_rate;
        
        // Print temperature information periodically
        if (iteration % 100 == 0) {
            std::cout << "Temperature: " << temperature << std::endl;
        }
    }
    
    std::cout << "\nSimulated Annealing completed" << std::endl;
    std::cout << "Final temperature: " << temperature << std::endl;
    std::cout << "Best solution found:" << std::endl;
    print_vector(best_solution, "x*");
    std::cout << "Objective value: " << best_energy << std::endl;
    
    return best_solution;
}

// Genetic Algorithm implementation
Vector genetic_algorithm(
    const Function& objective,
    const Vector& lower_bounds,
    const Vector& upper_bounds,
    int population_size = 50,
    int max_generations = 100,
    double crossover_rate = 0.8,
    double mutation_rate = 0.1
) {
    int dimension = lower_bounds.size();
    
    // Initialize population randomly within bounds
    std::vector<Vector> population(population_size);
    std::vector<double> fitness(population_size);
    
    for (int i = 0; i < population_size; ++i) {
        population[i] = random_vector(lower_bounds, upper_bounds);
        fitness[i] = -objective(population[i]);  // Negative because we want to maximize fitness
    }
    
    Vector best_solution = population[0];
    double best_objective = objective(best_solution);
    
    std::cout << "\nStarting Genetic Algorithm" << std::endl;
    std::cout << "========================" << std::endl;
    std::cout << "Population size: " << population_size << std::endl;
    std::cout << "Crossover rate: " << crossover_rate << std::endl;
    std::cout << "Mutation rate: " << mutation_rate << std::endl;
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // Main GA loop
    for (int generation = 0; generation < max_generations; ++generation) {
        // Create new generation
        std::vector<Vector> new_population(population_size);
        
        // Elitism: Keep the best individual
        int best_idx = 0;
        for (int i = 1; i < population_size; ++i) {
            if (fitness[i] > fitness[best_idx]) {
                best_idx = i;
            }
        }
        new_population[0] = population[best_idx];
        
        // Selection, crossover, and mutation
        for (int i = 1; i < population_size; ++i) {
            // Tournament selection
            int parent1_idx = rand() % population_size;
            int parent2_idx = rand() % population_size;
            
            for (int j = 0; j < 3; ++j) {  // Tournament size = 3
                int competitor_idx = rand() % population_size;
                if (fitness[competitor_idx] > fitness[parent1_idx]) {
                    parent1_idx = competitor_idx;
                }
            }
            
            for (int j = 0; j < 3; ++j) {  // Tournament size = 3
                int competitor_idx = rand() % population_size;
                if (fitness[competitor_idx] > fitness[parent2_idx]) {
                    parent2_idx = competitor_idx;
                }
            }
            
            Vector& parent1 = population[parent1_idx];
            Vector& parent2 = population[parent2_idx];
            
            // Crossover
            Vector child = parent1;
            if (dist(rng) < crossover_rate) {
                // Single-point crossover
                int crossover_point = rand() % dimension;
                for (int j = crossover_point; j < dimension; ++j) {
                    child[j] = parent2[j];
                }
            }
            
            // Mutation
            for (int j = 0; j < dimension; ++j) {
                if (dist(rng) < mutation_rate) {
                    // Random mutation within bounds
                    child[j] = lower_bounds[j] + dist(rng) * (upper_bounds[j] - lower_bounds[j]);
                }
            }
            
            new_population[i] = child;
        }
        
        // Update population and evaluate fitness
        population = new_population;
        for (int i = 0; i < population_size; ++i) {
            fitness[i] = -objective(population[i]);
        }
        
        // Find the best solution in this generation
        for (int i = 0; i < population_size; ++i) {
            double obj_value = objective(population[i]);
            if (obj_value < best_objective) {
                best_solution = population[i];
                best_objective = obj_value;
            }
        }
        
        // Print progress every few generations
        if (generation % 10 == 0 || generation == max_generations - 1) {
            std::cout << "Generation " << std::setw(4) << generation 
                      << " | Best objective: " << std::setw(12) << std::fixed 
                      << std::setprecision(6) << best_objective << std::endl;
        }
    }
    
    std::cout << "\nGenetic Algorithm completed" << std::endl;
    std::cout << "Best solution found:" << std::endl;
    print_vector(best_solution, "x*");
    std::cout << "Objective value: " << best_objective << std::endl;
    
    return best_solution;
}

// Particle Swarm Optimization (PSO) implementation
Vector particle_swarm_optimization(
    const Function& objective,
    const Vector& lower_bounds,
    const Vector& upper_bounds,
    int swarm_size = 30,
    int max_iterations = 100,
    double w = 0.7,      // Inertia weight
    double c1 = 1.5,     // Cognitive coefficient
    double c2 = 1.5      // Social coefficient
) {
    int dimension = lower_bounds.size();
    
    // Initialize particles randomly within bounds
    std::vector<Vector> positions(swarm_size);
    std::vector<Vector> velocities(swarm_size);
    std::vector<Vector> personal_best_positions(swarm_size);
    std::vector<double> personal_best_values(swarm_size);
    
    // Initialize global best
    Vector global_best_position;
    double global_best_value = std::numeric_limits<double>::max();
    
    std::uniform_real_distribution<double> dist(0.0, 1.0);
    
    // Initialize particles
    for (int i = 0; i < swarm_size; ++i) {
        // Random position within bounds
        positions[i] = random_vector(lower_bounds, upper_bounds);
        
        // Random initial velocity (between -range and +range, where range is upper-lower)
        velocities[i].resize(dimension);
        for (int j = 0; j < dimension; ++j) {
            double range = upper_bounds[j] - lower_bounds[j];
            velocities[i][j] = (dist(rng) * 2.0 - 1.0) * range * 0.1;  // 10% of range
        }
        
        // Evaluate initial position
        double value = objective(positions[i]);
        
        // Initialize personal best
        personal_best_positions[i] = positions[i];
        personal_best_values[i] = value;
        
        // Update global best if needed
        if (value < global_best_value) {
            global_best_value = value;
            global_best_position = positions[i];
        }
    }
    
    std::cout << "\nStarting Particle Swarm Optimization" << std::endl;
    std::cout << "=================================" << std::endl;
    std::cout << "Swarm size: " << swarm_size << std::endl;
    std::cout << "Parameters: w = " << w << ", c1 = " << c1 << ", c2 = " << c2 << std::endl;
    
    // Main PSO loop
    for (int iter = 0; iter < max_iterations; ++iter) {
        // Update each particle
        for (int i = 0; i < swarm_size; ++i) {
            // Update velocity
            for (int j = 0; j < dimension; ++j) {
                // Cognitive component (attraction to personal best)
                double cognitive = c1 * dist(rng) * (personal_best_positions[i][j] - positions[i][j]);
                
                // Social component (attraction to global best)
                double social = c2 * dist(rng) * (global_best_position[j] - positions[i][j]);
                
                // Update velocity with inertia
                velocities[i][j] = w * velocities[i][j] + cognitive + social;
                
                // Optional: Velocity clamping
                double v_max = 0.1 * (upper_bounds[j] - lower_bounds[j]);
                velocities[i][j] = std::max(std::min(velocities[i][j], v_max), -v_max);
            }
            
            // Update position
            for (int j = 0; j < dimension; ++j) {
                positions[i][j] += velocities[i][j];
                
                // Ensure position is within bounds
                positions[i][j] = std::max(positions[i][j], lower_bounds[j]);
                positions[i][j] = std::min(positions[i][j], upper_bounds[j]);
            }
            
            // Evaluate new position
            double value = objective(positions[i]);
            
            // Update personal best if needed
            if (value < personal_best_values[i]) {
                personal_best_values[i] = value;
                personal_best_positions[i] = positions[i];
                
                // Update global best if needed
                if (value < global_best_value) {
                    global_best_value = value;
                    global_best_position = positions[i];
                }
            }
        }
        
        // Print progress every few iterations
        if (iter % 10 == 0 || iter == max_iterations - 1) {
            std::cout << "Iteration " << std::setw(4) << iter 
                      << " | Best objective: " << std::setw(12) << std::fixed 
                      << std::setprecision(6) << global_best_value << std::endl;
        }
        
        // Optional: Dynamic parameter adjustment
        // Linearly decrease inertia weight
        w = w * 0.99;
    }
    
    std::cout << "\nParticle Swarm Optimization completed" << std::endl;
    std::cout << "Best solution found:" << std::endl;
    print_vector(global_best_position, "x*");
    std::cout << "Objective value: " << global_best_value << std::endl;
    
    return global_best_position;
}

// Example test functions for optimization

// Rosenbrock function (banana function)
// f(x,y) = (1-x)² + 100(y-x²)²
// Global minimum at (1,1) with f(1,1) = 0
double rosenbrock(const Vector& x) {
    return std::pow(1.0 - x[0], 2) + 100.0 * std::pow(x[1] - x[0] * x[0], 2);
}

// Sphere function
// f(x) = sum(x_i²)
// Global minimum at origin with f(0,...,0) = 0
double sphere(const Vector& x) {
    double sum = 0.0;
    for (double xi : x) {
        sum += xi * xi;
    }
    return sum;
}

// Rastrigin function
// f(x) = 10n + sum(x_i² - 10cos(2πx_i))
// Global minimum at origin with f(0,...,0) = 0
double rastrigin(const Vector& x) {
    double sum = 10.0 * x.size();
    for (double xi : x) {
        sum += xi * xi - 10.0 * std::cos(2.0 * M_PI * xi);
    }
    return sum;
}

// Ackley function
// f(x) = -20exp(-0.2sqrt(0.5sum(x_i²))) - exp(0.5sum(cos(2πx_i))) + e + 20
// Global minimum at origin with f(0,...,0) = 0
double ackley(const Vector& x) {
    double sum1 = 0.0;
    double sum2 = 0.0;
    
    for (double xi : x) {
        sum1 += xi * xi;
        sum2 += std::cos(2.0 * M_PI * xi);
    }
    
    double term1 = -20.0 * std::exp(-0.2 * std::sqrt(sum1 / x.size()));
    double term2 = -std::exp(sum2 / x.size());
    
    return term1 + term2 + std::exp(1.0) + 20.0;
}

int main() {
    std::cout << "===== Metaheuristic Optimization Algorithms =====" << std::endl;
    
    // Define problem dimensions and bounds
    int dimension = 2;
    Vector lower_bounds(dimension, -5.0);
    Vector upper_bounds(dimension, 5.0);
    
    std::cout << "\n\n=== PROBLEM 1: Rosenbrock Function ===" << std::endl;
    std::cout << "Global minimum at (1,1) with f(1,1) = 0" << std::endl;
    
    // Initial point for simulated annealing
    Vector initial_point = {-1.0, -1.0};
    
    // Simulated Annealing
    std::cout << "\n1. Simulated Annealing:" << std::endl;
    Vector sa_result = simulated_annealing(
        rosenbrock, initial_point, lower_bounds, upper_bounds,
        100.0, 0.95, 1000, 20
    );
    
    // Genetic Algorithm
    std::cout << "\n2. Genetic Algorithm:" << std::endl;
    Vector ga_result = genetic_algorithm(
        rosenbrock, lower_bounds, upper_bounds,
        30, 100, 0.8, 0.1
    );
    
    // Particle Swarm Optimization
    std::cout << "\n3. Particle Swarm Optimization:" << std::endl;
    Vector pso_result = particle_swarm_optimization(
        rosenbrock, lower_bounds, upper_bounds,
        20, 100, 0.7, 1.5, 1.5
    );
    
    // Compare results
    std::cout << "\nComparison of results for Rosenbrock Function:" << std::endl;
    std::cout << "True minimum: (1.0, 1.0) with f(x*) = 0.0" << std::endl;
    std::cout << "SA  result: f(x*) = " << rosenbrock(sa_result) << std::endl;
    std::cout << "GA  result: f(x*) = " << rosenbrock(ga_result) << std::endl;
    std::cout << "PSO result: f(x*) = " << rosenbrock(pso_result) << std::endl;
    
    // Second test problem: Rastrigin function
    std::cout << "\n\n=== PROBLEM 2: Rastrigin Function ===" << std::endl;
    std::cout << "Global minimum at origin with f(0,0) = 0" << std::endl;
    
    // Adjust bounds for Rastrigin
    lower_bounds = Vector(dimension, -5.12);
    upper_bounds = Vector(dimension, 5.12);
    initial_point = {2.0, 3.0};
    
    // Simulated Annealing
    std::cout << "\n1. Simulated Annealing:" << std::endl;
    sa_result = simulated_annealing(
        rastrigin, initial_point, lower_bounds, upper_bounds,
        100.0, 0.95, 1000, 20
    );
    
    // Genetic Algorithm
    std::cout << "\n2. Genetic Algorithm:" << std::endl;
    ga_result = genetic_algorithm(
        rastrigin, lower_bounds, upper_bounds,
        30, 100, 0.8, 0.2
    );
    
    // Particle Swarm Optimization
    std::cout << "\n3. Particle Swarm Optimization:" << std::endl;
    pso_result = particle_swarm_optimization(
        rastrigin, lower_bounds, upper_bounds,
        20, 100, 0.7, 1.5, 1.5
    );
    
    // Compare results
    std::cout << "\nComparison of results for Rastrigin Function:" << std::endl;
    std::cout << "True minimum: (0.0, 0.0) with f(x*) = 0.0" << std::endl;
    std::cout << "SA  result: f(x*) = " << rastrigin(sa_result) << std::endl;
    std::cout << "GA  result: f(x*) = " << rastrigin(ga_result) << std::endl;
    std::cout << "PSO result: f(x*) = " << rastrigin(pso_result) << std::endl;
    
    return 0;
}
