#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <functional>
#include <iomanip>
#include <algorithm>
#include <chrono>

/**
 * Monte Carlo Methods Implementation
 * 
 * This file demonstrates Monte Carlo simulation techniques in C++,
 * which are essential for stochastic optimization and uncertainty modeling.
 */

// Function to estimate the value of pi using Monte Carlo simulation
double estimate_pi(int num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(0.0, 1.0);
    
    int points_inside_circle = 0;
    
    for (int i = 0; i < num_samples; ++i) {
        double x = dist(gen);
        double y = dist(gen);
        
        // Check if the point is inside the quarter circle
        if (x*x + y*y <= 1.0) {
            points_inside_circle++;
        }
    }
    
    // The ratio of points inside the quarter circle to total points
    // approximates π/4
    return 4.0 * points_inside_circle / num_samples;
}

// Function to estimate a definite integral using Monte Carlo integration
double monte_carlo_integration(const std::function<double(double)>& f, 
                              double a, double b, int num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(a, b);
    
    double sum = 0.0;
    
    for (int i = 0; i < num_samples; ++i) {
        double x = dist(gen);
        sum += f(x);
    }
    
    return (b - a) * sum / num_samples;
}

// Function to estimate a multi-dimensional integral using Monte Carlo integration
double monte_carlo_integration_multi(
    const std::function<double(const std::vector<double>&)>& f,
    const std::vector<double>& lower_bounds,
    const std::vector<double>& upper_bounds,
    int num_samples) {
    
    if (lower_bounds.size() != upper_bounds.size()) {
        throw std::invalid_argument("Bounds dimensions must match");
    }
    
    int dim = lower_bounds.size();
    std::random_device rd;
    std::mt19937 gen(rd());
    
    std::vector<std::uniform_real_distribution<>> dists;
    double volume = 1.0;
    
    for (int i = 0; i < dim; ++i) {
        dists.emplace_back(lower_bounds[i], upper_bounds[i]);
        volume *= (upper_bounds[i] - lower_bounds[i]);
    }
    
    double sum = 0.0;
    
    for (int i = 0; i < num_samples; ++i) {
        std::vector<double> x(dim);
        for (int j = 0; j < dim; ++j) {
            x[j] = dists[j](gen);
        }
        sum += f(x);
    }
    
    return volume * sum / num_samples;
}

// Function to estimate the expected value of a random variable using Monte Carlo
double monte_carlo_expected_value(const std::function<double()>& random_generator, 
                                 int num_samples) {
    double sum = 0.0;
    
    for (int i = 0; i < num_samples; ++i) {
        sum += random_generator();
    }
    
    return sum / num_samples;
}

// Function to estimate the probability of an event using Monte Carlo
double monte_carlo_probability(const std::function<bool()>& event_checker, 
                              int num_samples) {
    int count = 0;
    
    for (int i = 0; i < num_samples; ++i) {
        if (event_checker()) {
            count++;
        }
    }
    
    return static_cast<double>(count) / num_samples;
}

// Function to perform rejection sampling
std::vector<double> rejection_sampling(
    const std::function<double(double)>& target_pdf,
    const std::function<double()>& proposal_sampler,
    const std::function<double(double)>& proposal_pdf,
    double M,  // Scaling factor such that M * proposal_pdf(x) >= target_pdf(x) for all x
    int num_samples) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> u_dist(0.0, 1.0);
    
    std::vector<double> samples;
    samples.reserve(num_samples);
    
    int attempts = 0;
    const int max_attempts = num_samples * 100;  // Avoid infinite loops
    
    while (samples.size() < num_samples && attempts < max_attempts) {
        double x = proposal_sampler();
        double u = u_dist(gen);
        
        // Accept or reject the sample
        if (u <= target_pdf(x) / (M * proposal_pdf(x))) {
            samples.push_back(x);
        }
        
        attempts++;
    }
    
    if (samples.size() < num_samples) {
        std::cout << "Warning: Could not generate requested number of samples. "
                  << "Generated " << samples.size() << " out of " << num_samples << ".\n";
    }
    
    return samples;
}

// Function to perform importance sampling for integral estimation
double importance_sampling(
    const std::function<double(double)>& f,
    const std::function<double()>& importance_sampler,
    const std::function<double(double)>& importance_pdf,
    int num_samples) {
    
    double sum = 0.0;
    
    for (int i = 0; i < num_samples; ++i) {
        double x = importance_sampler();
        sum += f(x) / importance_pdf(x);
    }
    
    return sum / num_samples;
}

// Function to estimate confidence interval for Monte Carlo estimate
std::pair<double, double> confidence_interval(const std::vector<double>& estimates, double confidence_level = 0.95) {
    double mean = 0.0;
    for (double est : estimates) {
        mean += est;
    }
    mean /= estimates.size();
    
    double variance = 0.0;
    for (double est : estimates) {
        double diff = est - mean;
        variance += diff * diff;
    }
    variance /= (estimates.size() - 1);  // Bessel's correction
    
    double std_dev = std::sqrt(variance);
    double std_error = std_dev / std::sqrt(estimates.size());
    
    // For 95% confidence, use 1.96 (assuming normal distribution)
    double z;
    if (std::abs(confidence_level - 0.90) < 1e-6) {
        z = 1.645;
    } else if (std::abs(confidence_level - 0.95) < 1e-6) {
        z = 1.96;
    } else if (std::abs(confidence_level - 0.99) < 1e-6) {
        z = 2.576;
    } else {
        z = 1.96;  // Default to 95%
    }
    
    double margin = z * std_error;
    
    return {mean - margin, mean + margin};
}

// Function to perform a simple Monte Carlo simulation for a stochastic process
std::vector<double> simulate_stochastic_process(
    double initial_value,
    const std::function<double(double, double)>& transition_function,
    int num_steps,
    int num_paths) {
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal_dist(0.0, 1.0);
    
    std::vector<double> final_values(num_paths);
    
    for (int path = 0; path < num_paths; ++path) {
        double current_value = initial_value;
        
        for (int step = 0; step < num_steps; ++step) {
            double random_shock = normal_dist(gen);
            current_value = transition_function(current_value, random_shock);
        }
        
        final_values[path] = current_value;
    }
    
    return final_values;
}

int main() {
    std::cout << "Monte Carlo Methods Examples\n";
    std::cout << "===========================\n";
    
    // Example 1: Estimating π
    std::cout << "\nExample 1: Estimating π using Monte Carlo\n";
    
    std::vector<int> sample_sizes = {1000, 10000, 100000, 1000000};
    
    for (int n : sample_sizes) {
        double pi_estimate = estimate_pi(n);
        double error = std::abs(pi_estimate - M_PI);
        
        std::cout << "Samples: " << std::setw(8) << n 
                  << ", π estimate: " << std::fixed << std::setprecision(6) << pi_estimate 
                  << ", error: " << error 
                  << ", relative error: " << error / M_PI * 100 << "%" << std::endl;
    }
    
    // Example 2: Monte Carlo Integration
    std::cout << "\nExample 2: Monte Carlo Integration\n";
    
    // Integrate f(x) = x^2 from 0 to 1
    // Analytical result: 1/3
    auto f = [](double x) { return x * x; };
    
    for (int n : sample_sizes) {
        double integral_estimate = monte_carlo_integration(f, 0.0, 1.0, n);
        double analytical_result = 1.0 / 3.0;
        double error = std::abs(integral_estimate - analytical_result);
        
        std::cout << "Samples: " << std::setw(8) << n 
                  << ", Integral estimate: " << std::fixed << std::setprecision(6) << integral_estimate 
                  << ", analytical: " << analytical_result 
                  << ", error: " << error << std::endl;
    }
    
    // Example 3: Multi-dimensional Monte Carlo Integration
    std::cout << "\nExample 3: Multi-dimensional Monte Carlo Integration\n";
    
    // Integrate f(x,y) = x^2 + y^2 over the unit square [0,1]×[0,1]
    // Analytical result: 2/3
    auto f_2d = [](const std::vector<double>& x) { 
        return x[0] * x[0] + x[1] * x[1]; 
    };
    
    std::vector<double> lower_bounds = {0.0, 0.0};
    std::vector<double> upper_bounds = {1.0, 1.0};
    
    for (int n : {1000, 10000, 100000}) {
        double integral_estimate = monte_carlo_integration_multi(f_2d, lower_bounds, upper_bounds, n);
        double analytical_result = 2.0 / 3.0;
        double error = std::abs(integral_estimate - analytical_result);
        
        std::cout << "Samples: " << std::setw(8) << n 
                  << ", Integral estimate: " << std::fixed << std::setprecision(6) << integral_estimate 
                  << ", analytical: " << analytical_result 
                  << ", error: " << error << std::endl;
    }
    
    // Example 4: Expected Value Estimation
    std::cout << "\nExample 4: Expected Value Estimation\n";
    
    // Estimate E[X^2] where X ~ N(0,1)
    // Analytical result: Var[X] + E[X]^2 = 1 + 0^2 = 1
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal_dist(0.0, 1.0);
    
    auto squared_normal = [&]() { 
        double x = normal_dist(gen);
        return x * x;
    };
    
    for (int n : sample_sizes) {
        double ev_estimate = monte_carlo_expected_value(squared_normal, n);
        double analytical_result = 1.0;
        double error = std::abs(ev_estimate - analytical_result);
        
        std::cout << "Samples: " << std::setw(8) << n 
                  << ", E[X^2] estimate: " << std::fixed << std::setprecision(6) << ev_estimate 
                  << ", analytical: " << analytical_result 
                  << ", error: " << error << std::endl;
    }
    
    // Example 5: Probability Estimation
    std::cout << "\nExample 5: Probability Estimation\n";
    
    // Estimate P(X > 1) where X ~ N(0,1)
    // Analytical result: 1 - Φ(1) ≈ 0.1587
    auto normal_gt_1 = [&]() { 
        return normal_dist(gen) > 1.0;
    };
    
    double analytical_prob = 0.1587;
    
    for (int n : sample_sizes) {
        double prob_estimate = monte_carlo_probability(normal_gt_1, n);
        double error = std::abs(prob_estimate - analytical_prob);
        
        std::cout << "Samples: " << std::setw(8) << n 
                  << ", P(X > 1) estimate: " << std::fixed << std::setprecision(6) << prob_estimate 
                  << ", analytical: " << analytical_prob 
                  << ", error: " << error << std::endl;
    }
    
    // Example 6: Rejection Sampling
    std::cout << "\nExample 6: Rejection Sampling\n";
    
    // Target distribution: Beta(2,5)
    auto beta_pdf = [](double x) {
        if (x < 0.0 || x > 1.0) return 0.0;
        return 20.0 * x * std::pow(1.0 - x, 4);  // Unnormalized Beta(2,5)
    };
    
    // Proposal distribution: Uniform(0,1)
    auto uniform_sampler = [&]() {
        std::uniform_real_distribution<> dist(0.0, 1.0);
        return dist(gen);
    };
    
    auto uniform_pdf = [](double x) {
        if (x < 0.0 || x > 1.0) return 0.0;
        return 1.0;
    };
    
    // M = 20/16 = 1.25 (maximum value of beta_pdf is 20/16 at x = 1/5)
    double M = 1.25;
    
    std::vector<double> beta_samples = rejection_sampling(beta_pdf, uniform_sampler, uniform_pdf, M, 10000);
    
    double beta_mean = 0.0;
    for (double sample : beta_samples) {
        beta_mean += sample;
    }
    beta_mean /= beta_samples.size();
    
    std::cout << "Generated " << beta_samples.size() << " samples from Beta(2,5) distribution.\n";
    std::cout << "Sample mean: " << beta_mean << " (theoretical: 2/(2+5) = 0.2857)\n";
    
    // Example 7: Confidence Intervals
    std::cout << "\nExample 7: Confidence Intervals\n";
    
    // Perform multiple estimates of π to compute confidence intervals
    int num_experiments = 30;
    int samples_per_experiment = 10000;
    
    std::vector<double> pi_estimates(num_experiments);
    
    for (int i = 0; i < num_experiments; ++i) {
        pi_estimates[i] = estimate_pi(samples_per_experiment);
    }
    
    auto [ci_lower, ci_upper] = confidence_interval(pi_estimates, 0.95);
    
    std::cout << "95% confidence interval for π based on " << num_experiments 
              << " experiments with " << samples_per_experiment << " samples each:\n";
    std::cout << "[" << ci_lower << ", " << ci_upper << "]" << std::endl;
    std::cout << "True value of π: " << M_PI << std::endl;
    std::cout << "Is π in the confidence interval? " 
              << ((M_PI >= ci_lower && M_PI <= ci_upper) ? "Yes" : "No") << std::endl;
    
    // Example 8: Simulating a Stochastic Process (Geometric Brownian Motion)
    std::cout << "\nExample 8: Simulating a Stochastic Process (GBM)\n";
    
    // Parameters for GBM: S(t+1) = S(t) * exp((μ - σ²/2) * dt + σ * √dt * Z)
    double S0 = 100.0;  // Initial stock price
    double mu = 0.05;   // Drift (annual)
    double sigma = 0.2; // Volatility (annual)
    double dt = 1.0/252; // Daily time step (252 trading days per year)
    int num_steps = 252; // Simulate for 1 year
    int num_paths = 1000; // Number of simulation paths
    
    // Transition function for GBM
    auto gbm_transition = [mu, sigma, dt](double S, double Z) {
        return S * std::exp((mu - 0.5 * sigma * sigma) * dt + sigma * std::sqrt(dt) * Z);
    };
    
    auto start_time = std::chrono::high_resolution_clock::now();
    
    std::vector<double> final_prices = simulate_stochastic_process(S0, gbm_transition, num_steps, num_paths);
    
    auto end_time = std::chrono::high_resolution_clock::now();
    auto duration = std::chrono::duration_cast<std::chrono::milliseconds>(end_time - start_time);
    
    // Compute statistics of final prices
    double mean_price = 0.0;
    for (double price : final_prices) {
        mean_price += price;
    }
    mean_price /= final_prices.size();
    
    double var_price = 0.0;
    for (double price : final_prices) {
        double diff = price - mean_price;
        var_price += diff * diff;
    }
    var_price /= (final_prices.size() - 1);
    
    double std_dev_price = std::sqrt(var_price);
    
    // Theoretical mean and standard deviation for GBM
    double theoretical_mean = S0 * std::exp(mu * num_steps * dt);
    double theoretical_std_dev = theoretical_mean * std::sqrt(std::exp(sigma * sigma * num_steps * dt) - 1);
    
    std::cout << "Simulated " << num_paths << " paths of GBM for " << num_steps << " steps in " 
              << duration.count() << " ms.\n";
    std::cout << "Mean final price: " << mean_price << " (theoretical: " << theoretical_mean << ")\n";
    std::cout << "Std dev of final price: " << std_dev_price << " (theoretical: " << theoretical_std_dev << ")\n";
    
    return 0;
}
