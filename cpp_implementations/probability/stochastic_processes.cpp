#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <functional>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <chrono>

/**
 * Stochastic Processes Implementation
 * 
 * This file demonstrates stochastic processes and their applications
 * in C++, which are essential for stochastic optimization and modeling.
 */

// Function to print a vector
void print_vector(const std::vector<double>& v, const std::string& name = "Vector") {
    std::cout << name << ": [";
    for (size_t i = 0; i < v.size(); ++i) {
        std::cout << std::fixed << std::setprecision(4) << v[i];
        if (i < v.size() - 1) std::cout << ", ";
    }
    std::cout << "]" << std::endl;
}

// Function to print statistics of a sample
void print_statistics(const std::vector<double>& samples, const std::string& name = "Samples") {
    double sum = std::accumulate(samples.begin(), samples.end(), 0.0);
    double mean = sum / samples.size();
    
    double sq_sum = std::inner_product(samples.begin(), samples.end(), samples.begin(), 0.0);
    double variance = sq_sum / samples.size() - mean * mean;
    double std_dev = std::sqrt(variance);
    
    auto minmax = std::minmax_element(samples.begin(), samples.end());
    
    std::cout << "Statistics for " << name << ":" << std::endl;
    std::cout << "  Mean: " << mean << std::endl;
    std::cout << "  Variance: " << variance << std::endl;
    std::cout << "  Standard Deviation: " << std_dev << std::endl;
    std::cout << "  Min: " << *minmax.first << std::endl;
    std::cout << "  Max: " << *minmax.second << std::endl;
}

// Class to simulate a random walk
class RandomWalk {
public:
    RandomWalk(double start_position = 0.0, double step_size = 1.0)
        : current_position_(start_position), step_size_(step_size) {
        
        path_.push_back(current_position_);
    }
    
    // Take a step in the random walk
    void step() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0.0, 1.0);
        
        // Equal probability of moving left or right
        double direction = (dist(gen) < 0.5) ? -1.0 : 1.0;
        current_position_ += direction * step_size_;
        
        path_.push_back(current_position_);
    }
    
    // Simulate n steps
    void simulate(int n) {
        for (int i = 0; i < n; ++i) {
            step();
        }
    }
    
    // Get the current position
    double position() const {
        return current_position_;
    }
    
    // Get the entire path
    const std::vector<double>& path() const {
        return path_;
    }
    
private:
    double current_position_;
    double step_size_;
    std::vector<double> path_;
};

// Class to simulate a Poisson process
class PoissonProcess {
public:
    PoissonProcess(double rate)
        : rate_(rate) {}
    
    // Generate inter-arrival times
    std::vector<double> generate_inter_arrival_times(int n) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::exponential_distribution<> dist(rate_);
        
        std::vector<double> times(n);
        for (int i = 0; i < n; ++i) {
            times[i] = dist(gen);
        }
        
        return times;
    }
    
    // Generate arrival times
    std::vector<double> generate_arrival_times(int n) {
        std::vector<double> inter_arrival_times = generate_inter_arrival_times(n);
        std::vector<double> arrival_times(n);
        
        double time = 0.0;
        for (int i = 0; i < n; ++i) {
            time += inter_arrival_times[i];
            arrival_times[i] = time;
        }
        
        return arrival_times;
    }
    
    // Generate number of events in time intervals
    std::vector<int> generate_event_counts(const std::vector<double>& intervals) {
        std::random_device rd;
        std::mt19937 gen(rd());
        
        std::vector<int> counts(intervals.size());
        for (size_t i = 0; i < intervals.size(); ++i) {
            std::poisson_distribution<> dist(rate_ * intervals[i]);
            counts[i] = dist(gen);
        }
        
        return counts;
    }
    
private:
    double rate_;  // Rate parameter (events per unit time)
};

// Class to simulate a Brownian motion (Wiener process)
class BrownianMotion {
public:
    BrownianMotion(double start_position = 0.0, double volatility = 1.0)
        : current_position_(start_position), volatility_(volatility) {
        
        path_.push_back(current_position_);
        times_.push_back(0.0);
    }
    
    // Simulate a path with n steps over time interval [0, T]
    void simulate(int n, double T) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 1.0);
        
        double dt = T / n;
        double time = 0.0;
        
        for (int i = 0; i < n; ++i) {
            time += dt;
            double dW = dist(gen) * std::sqrt(dt);
            current_position_ += volatility_ * dW;
            
            path_.push_back(current_position_);
            times_.push_back(time);
        }
    }
    
    // Get the current position
    double position() const {
        return current_position_;
    }
    
    // Get the entire path
    const std::vector<double>& path() const {
        return path_;
    }
    
    // Get the time points
    const std::vector<double>& times() const {
        return times_;
    }
    
private:
    double current_position_;
    double volatility_;
    std::vector<double> path_;
    std::vector<double> times_;
};

// Class to simulate a Geometric Brownian Motion (GBM)
class GeometricBrownianMotion {
public:
    GeometricBrownianMotion(double start_price = 100.0, double drift = 0.05, double volatility = 0.2)
        : current_price_(start_price), drift_(drift), volatility_(volatility) {
        
        path_.push_back(current_price_);
        times_.push_back(0.0);
    }
    
    // Simulate a path with n steps over time interval [0, T]
    void simulate(int n, double T) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 1.0);
        
        double dt = T / n;
        double time = 0.0;
        
        for (int i = 0; i < n; ++i) {
            time += dt;
            double dW = dist(gen) * std::sqrt(dt);
            
            // GBM formula: S(t+dt) = S(t) * exp((μ - σ²/2) * dt + σ * dW)
            current_price_ *= std::exp((drift_ - 0.5 * volatility_ * volatility_) * dt + volatility_ * dW);
            
            path_.push_back(current_price_);
            times_.push_back(time);
        }
    }
    
    // Get the current price
    double price() const {
        return current_price_;
    }
    
    // Get the entire path
    const std::vector<double>& path() const {
        return path_;
    }
    
    // Get the time points
    const std::vector<double>& times() const {
        return times_;
    }
    
private:
    double current_price_;
    double drift_;
    double volatility_;
    std::vector<double> path_;
    std::vector<double> times_;
};

// Class to simulate a Mean-Reverting Process (Ornstein-Uhlenbeck process)
class MeanRevertingProcess {
public:
    MeanRevertingProcess(double start_value = 0.0, double mean = 0.0, 
                        double reversion_speed = 0.1, double volatility = 0.2)
        : current_value_(start_value), mean_(mean), 
          reversion_speed_(reversion_speed), volatility_(volatility) {
        
        path_.push_back(current_value_);
        times_.push_back(0.0);
    }
    
    // Simulate a path with n steps over time interval [0, T]
    void simulate(int n, double T) {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::normal_distribution<> dist(0.0, 1.0);
        
        double dt = T / n;
        double time = 0.0;
        
        for (int i = 0; i < n; ++i) {
            time += dt;
            double dW = dist(gen) * std::sqrt(dt);
            
            // Ornstein-Uhlenbeck process: dX = θ(μ - X)dt + σdW
            double dX = reversion_speed_ * (mean_ - current_value_) * dt + volatility_ * dW;
            current_value_ += dX;
            
            path_.push_back(current_value_);
            times_.push_back(time);
        }
    }
    
    // Get the current value
    double value() const {
        return current_value_;
    }
    
    // Get the entire path
    const std::vector<double>& path() const {
        return path_;
    }
    
    // Get the time points
    const std::vector<double>& times() const {
        return times_;
    }
    
private:
    double current_value_;
    double mean_;
    double reversion_speed_;
    double volatility_;
    std::vector<double> path_;
    std::vector<double> times_;
};

// Class to simulate a Markov Chain
class MarkovChain {
public:
    MarkovChain(const std::vector<std::vector<double>>& transition_matrix, int initial_state = 0)
        : transition_matrix_(transition_matrix), current_state_(initial_state) {
        
        // Validate transition matrix
        for (const auto& row : transition_matrix_) {
            double sum = std::accumulate(row.begin(), row.end(), 0.0);
            if (std::abs(sum - 1.0) > 1e-10) {
                throw std::invalid_argument("Each row of the transition matrix must sum to 1");
            }
        }
        
        path_.push_back(current_state_);
    }
    
    // Take a step in the Markov chain
    void step() {
        std::random_device rd;
        std::mt19937 gen(rd());
        std::uniform_real_distribution<> dist(0.0, 1.0);
        
        double u = dist(gen);
        double cumulative_prob = 0.0;
        
        for (size_t next_state = 0; next_state < transition_matrix_[current_state_].size(); ++next_state) {
            cumulative_prob += transition_matrix_[current_state_][next_state];
            if (u <= cumulative_prob) {
                current_state_ = next_state;
                break;
            }
        }
        
        path_.push_back(current_state_);
    }
    
    // Simulate n steps
    void simulate(int n) {
        for (int i = 0; i < n; ++i) {
            step();
        }
    }
    
    // Get the current state
    int state() const {
        return current_state_;
    }
    
    // Get the entire path
    const std::vector<int>& path() const {
        return path_;
    }
    
    // Compute the stationary distribution (if it exists)
    std::vector<double> compute_stationary_distribution(int max_iterations = 1000, double tolerance = 1e-10) {
        int n = transition_matrix_.size();
        std::vector<double> pi(n, 1.0 / n);  // Start with uniform distribution
        std::vector<double> pi_next(n);
        
        for (int iter = 0; iter < max_iterations; ++iter) {
            // pi_next = pi * P
            for (int j = 0; j < n; ++j) {
                pi_next[j] = 0.0;
                for (int i = 0; i < n; ++i) {
                    pi_next[j] += pi[i] * transition_matrix_[i][j];
                }
            }
            
            // Check convergence
            double max_diff = 0.0;
            for (int i = 0; i < n; ++i) {
                max_diff = std::max(max_diff, std::abs(pi_next[i] - pi[i]));
            }
            
            if (max_diff < tolerance) {
                return pi_next;
            }
            
            pi = pi_next;
        }
        
        std::cout << "Warning: Stationary distribution did not converge within " 
                  << max_iterations << " iterations." << std::endl;
        return pi;
    }
    
private:
    std::vector<std::vector<double>> transition_matrix_;
    int current_state_;
    std::vector<int> path_;
};

int main() {
    std::cout << "Stochastic Processes Examples\n";
    std::cout << "============================\n";
    
    // Example 1: Random Walk
    std::cout << "\nExample 1: Random Walk\n";
    
    RandomWalk random_walk(0.0, 1.0);
    random_walk.simulate(1000);
    
    std::cout << "Random walk after 1000 steps:" << std::endl;
    std::cout << "Final position: " << random_walk.position() << std::endl;
    
    // Compute statistics of the path
    print_statistics(random_walk.path(), "Random Walk Path");
    
    // Example 2: Poisson Process
    std::cout << "\nExample 2: Poisson Process\n";
    
    double rate = 2.0;  // 2 events per unit time
    PoissonProcess poisson_process(rate);
    
    // Generate 10 inter-arrival times
    std::vector<double> inter_arrival_times = poisson_process.generate_inter_arrival_times(10);
    std::cout << "Inter-arrival times:" << std::endl;
    print_vector(inter_arrival_times, "Times");
    print_statistics(inter_arrival_times, "Inter-arrival Times");
    std::cout << "Theoretical mean: " << 1.0 / rate << std::endl;
    
    // Generate 10 arrival times
    std::vector<double> arrival_times = poisson_process.generate_arrival_times(10);
    std::cout << "\nArrival times:" << std::endl;
    print_vector(arrival_times, "Times");
    
    // Generate event counts for different time intervals
    std::vector<double> intervals = {0.5, 1.0, 2.0, 5.0};
    std::vector<int> event_counts = poisson_process.generate_event_counts(intervals);
    
    std::cout << "\nEvent counts for different time intervals:" << std::endl;
    for (size_t i = 0; i < intervals.size(); ++i) {
        std::cout << "Interval: " << intervals[i] << ", Count: " << event_counts[i] 
                  << ", Expected: " << rate * intervals[i] << std::endl;
    }
    
    // Example 3: Brownian Motion (Wiener Process)
    std::cout << "\nExample 3: Brownian Motion (Wiener Process)\n";
    
    BrownianMotion brownian_motion(0.0, 1.0);
    brownian_motion.simulate(1000, 1.0);  // 1000 steps over 1 time unit
    
    std::cout << "Brownian motion after 1000 steps:" << std::endl;
    std::cout << "Final position: " << brownian_motion.position() << std::endl;
    
    // Compute statistics of the path
    print_statistics(brownian_motion.path(), "Brownian Motion Path");
    std::cout << "Theoretical mean: 0" << std::endl;
    std::cout << "Theoretical variance at t=1: " << 1.0 << std::endl;
    
    // Example 4: Geometric Brownian Motion
    std::cout << "\nExample 4: Geometric Brownian Motion\n";
    
    GeometricBrownianMotion gbm(100.0, 0.05, 0.2);
    gbm.simulate(252, 1.0);  // 252 trading days in a year
    
    std::cout << "Geometric Brownian Motion after 252 steps (1 year):" << std::endl;
    std::cout << "Final price: " << gbm.price() << std::endl;
    
    // Compute statistics of the path
    print_statistics(gbm.path(), "GBM Path");
    
    double S0 = 100.0;
    double mu = 0.05;
    double sigma = 0.2;
    double T = 1.0;
    
    std::cout << "Theoretical mean: " << S0 * std::exp(mu * T) << std::endl;
    std::cout << "Theoretical variance: " 
              << S0 * S0 * std::exp(2 * mu * T) * (std::exp(sigma * sigma * T) - 1) << std::endl;
    
    // Example 5: Mean-Reverting Process (Ornstein-Uhlenbeck)
    std::cout << "\nExample 5: Mean-Reverting Process (Ornstein-Uhlenbeck)\n";
    
    MeanRevertingProcess ou_process(1.0, 0.0, 0.5, 0.1);
    ou_process.simulate(1000, 10.0);  // 1000 steps over 10 time units
    
    std::cout << "Ornstein-Uhlenbeck process after 1000 steps:" << std::endl;
    std::cout << "Final value: " << ou_process.value() << std::endl;
    
    // Compute statistics of the path
    print_statistics(ou_process.path(), "OU Process Path");
    
    double mean = 0.0;
    double reversion_speed = 0.5;
    double vol = 0.1;
    double t = 10.0;
    
    std::cout << "Theoretical mean: " << mean << std::endl;
    std::cout << "Theoretical variance: " 
              << (vol * vol) / (2 * reversion_speed) * (1 - std::exp(-2 * reversion_speed * t)) << std::endl;
    
    // Example 6: Markov Chain
    std::cout << "\nExample 6: Markov Chain\n";
    
    // Define a simple weather model:
    // State 0: Sunny, State 1: Cloudy, State 2: Rainy
    std::vector<std::vector<double>> transition_matrix = {
        {0.7, 0.2, 0.1},  // From Sunny
        {0.3, 0.4, 0.3},  // From Cloudy
        {0.2, 0.3, 0.5}   // From Rainy
    };
    
    MarkovChain weather_model(transition_matrix, 0);  // Start with Sunny
    weather_model.simulate(100);
    
    // Count occurrences of each state
    std::vector<int> state_counts(3, 0);
    for (int state : weather_model.path()) {
        state_counts[state]++;
    }
    
    std::cout << "Weather model after 100 days:" << std::endl;
    std::cout << "Final state: " << weather_model.state() 
              << " (" << (weather_model.state() == 0 ? "Sunny" : 
                          (weather_model.state() == 1 ? "Cloudy" : "Rainy")) << ")" << std::endl;
    
    std::cout << "\nState counts:" << std::endl;
    std::cout << "Sunny: " << state_counts[0] << " days" << std::endl;
    std::cout << "Cloudy: " << state_counts[1] << " days" << std::endl;
    std::cout << "Rainy: " << state_counts[2] << " days" << std::endl;
    
    // Compute the stationary distribution
    std::vector<double> stationary_dist = weather_model.compute_stationary_distribution();
    
    std::cout << "\nStationary distribution:" << std::endl;
    std::cout << "Sunny: " << stationary_dist[0] << std::endl;
    std::cout << "Cloudy: " << stationary_dist[1] << std::endl;
    std::cout << "Rainy: " << stationary_dist[2] << std::endl;
    
    return 0;
}
