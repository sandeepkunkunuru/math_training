#include <iostream>
#include <vector>
#include <algorithm>
#include <limits>
#include <cmath>
#include <random>
#include <chrono>
#include <iomanip>

/**
 * Traveling Salesman Problem (TSP) Implementation
 * 
 * This file demonstrates various algorithms for solving the TSP including
 * brute force, dynamic programming, nearest neighbor heuristic, and 2-opt local search.
 */

const double INF = std::numeric_limits<double>::infinity();

// Point structure for 2D coordinates
struct Point {
    double x, y;
    std::string name;
    
    Point(double x = 0, double y = 0, const std::string& name = "") : x(x), y(y), name(name) {}
    
    double distance_to(const Point& other) const {
        return std::sqrt((x - other.x) * (x - other.x) + (y - other.y) * (y - other.y));
    }
};

// TSP instance representation
class TSPInstance {
public:
    std::vector<Point> cities;
    std::vector<std::vector<double>> distance_matrix;
    int n;
    
    TSPInstance(const std::vector<Point>& cities) : cities(cities), n(cities.size()) {
        build_distance_matrix();
    }
    
    void build_distance_matrix() {
        distance_matrix.assign(n, std::vector<double>(n, 0.0));
        for (int i = 0; i < n; ++i) {
            for (int j = 0; j < n; ++j) {
                if (i != j) {
                    distance_matrix[i][j] = cities[i].distance_to(cities[j]);
                } else {
                    distance_matrix[i][j] = 0.0;
                }
            }
        }
    }
    
    double tour_length(const std::vector<int>& tour) const {
        double length = 0.0;
        for (int i = 0; i < n; ++i) {
            int from = tour[i];
            int to = tour[(i + 1) % n];
            length += distance_matrix[from][to];
        }
        return length;
    }
    
    void print_instance() const {
        std::cout << "TSP Instance with " << n << " cities:" << std::endl;
        for (int i = 0; i < n; ++i) {
            std::cout << "  " << i << ": " << cities[i].name 
                     << " (" << std::fixed << std::setprecision(2) 
                     << cities[i].x << ", " << cities[i].y << ")" << std::endl;
        }
        std::cout << std::endl;
    }
    
    void print_tour(const std::vector<int>& tour, const std::string& method) const {
        std::cout << method << " Tour:" << std::endl;
        std::cout << "Route: ";
        for (int i = 0; i < n; ++i) {
            std::cout << cities[tour[i]].name;
            if (i < n - 1) std::cout << " -> ";
        }
        std::cout << " -> " << cities[tour[0]].name << std::endl;
        std::cout << "Total distance: " << std::fixed << std::setprecision(2) 
                 << tour_length(tour) << std::endl << std::endl;
    }
};

// Brute force TSP solver (for small instances)
class BruteForceTSP {
public:
    static std::vector<int> solve(const TSPInstance& instance) {
        std::cout << "Running Brute Force TSP" << std::endl;
        std::cout << "=======================" << std::endl;
        
        if (instance.n > 10) {
            std::cout << "Instance too large for brute force (n=" << instance.n << ")" << std::endl;
            return {};
        }
        
        std::vector<int> cities(instance.n);
        std::iota(cities.begin(), cities.end(), 0);
        
        std::vector<int> best_tour = cities;
        double best_length = instance.tour_length(cities);
        int permutations_checked = 0;
        
        do {
            double length = instance.tour_length(cities);
            permutations_checked++;
            
            if (length < best_length) {
                best_length = length;
                best_tour = cities;
            }
        } while (std::next_permutation(cities.begin() + 1, cities.end()));
        
        std::cout << "Checked " << permutations_checked << " permutations" << std::endl;
        std::cout << "Best tour length: " << std::fixed << std::setprecision(2) 
                 << best_length << std::endl;
        
        return best_tour;
    }
};

// Dynamic Programming TSP solver (Held-Karp algorithm)
class DynamicProgrammingTSP {
private:
    static int bit_count(int mask) {
        return __builtin_popcount(mask);
    }
    
public:
    static std::vector<int> solve(const TSPInstance& instance) {
        std::cout << "Running Dynamic Programming TSP (Held-Karp)" << std::endl;
        std::cout << "===========================================" << std::endl;
        
        int n = instance.n;
        if (n > 20) {
            std::cout << "Instance too large for DP (n=" << n << ")" << std::endl;
            return {};
        }
        
        // dp[mask][i] = minimum cost to visit all cities in mask ending at city i
        std::vector<std::vector<double>> dp(1 << n, std::vector<double>(n, INF));
        std::vector<std::vector<int>> parent(1 << n, std::vector<int>(n, -1));
        
        // Base case: start at city 0
        dp[1][0] = 0;
        
        // Fill DP table
        for (int mask = 1; mask < (1 << n); ++mask) {
            for (int u = 0; u < n; ++u) {
                if (!(mask & (1 << u)) || dp[mask][u] == INF) continue;
                
                for (int v = 0; v < n; ++v) {
                    if (mask & (1 << v)) continue;
                    
                    int new_mask = mask | (1 << v);
                    double new_cost = dp[mask][u] + instance.distance_matrix[u][v];
                    
                    if (new_cost < dp[new_mask][v]) {
                        dp[new_mask][v] = new_cost;
                        parent[new_mask][v] = u;
                    }
                }
            }
        }
        
        // Find minimum cost to return to start
        int final_mask = (1 << n) - 1;
        double min_cost = INF;
        int last_city = -1;
        
        for (int i = 1; i < n; ++i) {
            double cost = dp[final_mask][i] + instance.distance_matrix[i][0];
            if (cost < min_cost) {
                min_cost = cost;
                last_city = i;
            }
        }
        
        // Reconstruct tour
        std::vector<int> tour;
        int mask = final_mask;
        int current = last_city;
        
        while (current != -1) {
            tour.push_back(current);
            int prev = parent[mask][current];
            mask ^= (1 << current);
            current = prev;
        }
        
        std::reverse(tour.begin(), tour.end());
        
        std::cout << "Optimal tour length: " << std::fixed << std::setprecision(2) 
                 << min_cost << std::endl;
        
        return tour;
    }
};

// Nearest Neighbor heuristic
class NearestNeighborTSP {
public:
    static std::vector<int> solve(const TSPInstance& instance, int start_city = 0) {
        std::cout << "Running Nearest Neighbor Heuristic" << std::endl;
        std::cout << "===================================" << std::endl;
        
        int n = instance.n;
        std::vector<bool> visited(n, false);
        std::vector<int> tour;
        
        int current = start_city;
        tour.push_back(current);
        visited[current] = true;
        
        for (int step = 1; step < n; ++step) {
            double min_distance = INF;
            int next_city = -1;
            
            for (int i = 0; i < n; ++i) {
                if (!visited[i] && instance.distance_matrix[current][i] < min_distance) {
                    min_distance = instance.distance_matrix[current][i];
                    next_city = i;
                }
            }
            
            tour.push_back(next_city);
            visited[next_city] = true;
            current = next_city;
        }
        
        double tour_length = instance.tour_length(tour);
        std::cout << "Heuristic tour length: " << std::fixed << std::setprecision(2) 
                 << tour_length << std::endl;
        
        return tour;
    }
    
    // Try all starting cities and return best
    static std::vector<int> solve_best_start(const TSPInstance& instance) {
        std::cout << "Running Nearest Neighbor with Best Starting City" << std::endl;
        std::cout << "================================================" << std::endl;
        
        std::vector<int> best_tour;
        double best_length = INF;
        int best_start = 0;
        
        for (int start = 0; start < instance.n; ++start) {
            auto tour = solve(instance, start);
            double length = instance.tour_length(tour);
            
            if (length < best_length) {
                best_length = length;
                best_tour = tour;
                best_start = start;
            }
        }
        
        std::cout << "Best starting city: " << instance.cities[best_start].name << std::endl;
        std::cout << "Best tour length: " << std::fixed << std::setprecision(2) 
                 << best_length << std::endl;
        
        return best_tour;
    }
};

// 2-opt local search improvement
class TwoOptTSP {
private:
    static void two_opt_swap(std::vector<int>& tour, int i, int k) {
        std::reverse(tour.begin() + i, tour.begin() + k + 1);
    }
    
public:
    static std::vector<int> improve(const TSPInstance& instance, std::vector<int> tour) {
        std::cout << "Running 2-opt Local Search" << std::endl;
        std::cout << "==========================" << std::endl;
        
        int n = instance.n;
        double best_length = instance.tour_length(tour);
        bool improved = true;
        int iterations = 0;
        
        std::cout << "Initial tour length: " << std::fixed << std::setprecision(2) 
                 << best_length << std::endl;
        
        while (improved) {
            improved = false;
            iterations++;
            
            for (int i = 0; i < n - 1; ++i) {
                for (int k = i + 1; k < n; ++k) {
                    // Calculate change in tour length
                    int city1 = tour[i];
                    int city2 = tour[(i + 1) % n];
                    int city3 = tour[k];
                    int city4 = tour[(k + 1) % n];
                    
                    double old_distance = instance.distance_matrix[city1][city2] + 
                                        instance.distance_matrix[city3][city4];
                    double new_distance = instance.distance_matrix[city1][city3] + 
                                        instance.distance_matrix[city2][city4];
                    
                    if (new_distance < old_distance) {
                        two_opt_swap(tour, i + 1, k);
                        best_length = instance.tour_length(tour);
                        improved = true;
                        break;
                    }
                }
                if (improved) break;
            }
        }
        
        std::cout << "Improved tour length: " << std::fixed << std::setprecision(2) 
                 << best_length << std::endl;
        std::cout << "Iterations: " << iterations << std::endl;
        
        return tour;
    }
};

// Example TSP instances
TSPInstance create_small_example() {
    std::vector<Point> cities = {
        Point(0, 0, "A"),
        Point(1, 3, "B"),
        Point(4, 3, "C"),
        Point(6, 1, "D"),
        Point(3, 0, "E")
    };
    return TSPInstance(cities);
}

TSPInstance create_medium_example() {
    std::vector<Point> cities = {
        Point(60, 200, "City1"),
        Point(180, 200, "City2"),
        Point(80, 180, "City3"),
        Point(140, 180, "City4"),
        Point(20, 160, "City5"),
        Point(100, 160, "City6"),
        Point(200, 160, "City7"),
        Point(140, 140, "City8"),
        Point(40, 120, "City9"),
        Point(100, 120, "City10")
    };
    return TSPInstance(cities);
}

TSPInstance create_random_instance(int n, int seed = 42) {
    std::mt19937 gen(seed);
    std::uniform_real_distribution<> dis(0.0, 100.0);
    
    std::vector<Point> cities;
    for (int i = 0; i < n; ++i) {
        cities.emplace_back(dis(gen), dis(gen), "City" + std::to_string(i + 1));
    }
    
    return TSPInstance(cities);
}

int main() {
    std::cout << "Traveling Salesman Problem Algorithms" << std::endl;
    std::cout << "=====================================" << std::endl << std::endl;
    
    // Example 1: Small instance - compare all methods
    std::cout << "=== SMALL INSTANCE COMPARISON ===" << std::endl;
    auto small_instance = create_small_example();
    small_instance.print_instance();
    
    // Brute force (optimal)
    auto bf_tour = BruteForceTSP::solve(small_instance);
    if (!bf_tour.empty()) {
        small_instance.print_tour(bf_tour, "Brute Force (Optimal)");
    }
    
    // Dynamic programming (optimal)
    auto dp_tour = DynamicProgrammingTSP::solve(small_instance);
    if (!dp_tour.empty()) {
        small_instance.print_tour(dp_tour, "Dynamic Programming (Optimal)");
    }
    
    // Nearest neighbor heuristic
    auto nn_tour = NearestNeighborTSP::solve(small_instance);
    small_instance.print_tour(nn_tour, "Nearest Neighbor");
    
    // 2-opt improvement
    auto improved_tour = TwoOptTSP::improve(small_instance, nn_tour);
    small_instance.print_tour(improved_tour, "2-opt Improved");
    
    std::cout << std::endl;
    
    // Example 2: Medium instance - heuristics only
    std::cout << "=== MEDIUM INSTANCE (HEURISTICS) ===" << std::endl;
    auto medium_instance = create_medium_example();
    medium_instance.print_instance();
    
    // Nearest neighbor with best start
    auto nn_best_tour = NearestNeighborTSP::solve_best_start(medium_instance);
    medium_instance.print_tour(nn_best_tour, "Nearest Neighbor (Best Start)");
    
    // 2-opt improvement
    auto improved_medium = TwoOptTSP::improve(medium_instance, nn_best_tour);
    medium_instance.print_tour(improved_medium, "2-opt Improved");
    
    std::cout << std::endl;
    
    // Example 3: Random instance performance test
    std::cout << "=== RANDOM INSTANCE PERFORMANCE ===" << std::endl;
    auto random_instance = create_random_instance(15);
    std::cout << "Random instance with 15 cities" << std::endl << std::endl;
    
    auto start_time = std::chrono::high_resolution_clock::now();
    auto random_nn = NearestNeighborTSP::solve(random_instance);
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "Nearest Neighbor time: " << duration.count() << " microseconds" << std::endl;
    random_instance.print_tour(random_nn, "Nearest Neighbor");
    
    start_time = std::chrono::high_resolution_clock::now();
    auto random_improved = TwoOptTSP::improve(random_instance, random_nn);
    end_time = std::chrono::high_resolution_clock::now();
    duration = std::chrono::duration_cast<std::chrono::microseconds>(end_time - start_time);
    
    std::cout << "2-opt improvement time: " << duration.count() << " microseconds" << std::endl;
    random_instance.print_tour(random_improved, "2-opt Improved");
    
    return 0;
}
