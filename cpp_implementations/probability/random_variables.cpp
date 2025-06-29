#include <iostream>
#include <vector>
#include <random>
#include <cmath>
#include <functional>
#include <iomanip>
#include <algorithm>
#include <numeric>
#include <map>

/**
 * Random Variables Implementation
 * 
 * This file demonstrates concepts related to random variables, their transformations,
 * and joint distributions in C++, which are essential for stochastic optimization.
 */

// Function to compute the mean of a vector
double compute_mean(const std::vector<double>& data) {
    if (data.empty()) {
        return 0.0;
    }
    
    double sum = std::accumulate(data.begin(), data.end(), 0.0);
    return sum / data.size();
}

// Function to compute the variance of a vector
double compute_variance(const std::vector<double>& data) {
    if (data.size() <= 1) {
        return 0.0;
    }
    
    double mean = compute_mean(data);
    double sum_squared_diff = 0.0;
    
    for (double value : data) {
        double diff = value - mean;
        sum_squared_diff += diff * diff;
    }
    
    return sum_squared_diff / (data.size() - 1);  // Using Bessel's correction (n-1)
}

// Function to compute the covariance between two random variables
double compute_covariance(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.size() <= 1) {
        throw std::invalid_argument("Vectors must have the same size and at least 2 elements");
    }
    
    double mean_x = compute_mean(x);
    double mean_y = compute_mean(y);
    
    double sum_product_diff = 0.0;
    
    for (size_t i = 0; i < x.size(); ++i) {
        sum_product_diff += (x[i] - mean_x) * (y[i] - mean_y);
    }
    
    return sum_product_diff / (x.size() - 1);  // Using Bessel's correction (n-1)
}

// Function to compute the correlation coefficient between two random variables
double compute_correlation(const std::vector<double>& x, const std::vector<double>& y) {
    double cov = compute_covariance(x, y);
    double std_dev_x = std::sqrt(compute_variance(x));
    double std_dev_y = std::sqrt(compute_variance(y));
    
    if (std_dev_x < 1e-10 || std_dev_y < 1e-10) {
        return 0.0;  // Avoid division by zero
    }
    
    return cov / (std_dev_x * std_dev_y);
}

// Function to transform a random variable using a function
std::vector<double> transform_random_variable(const std::vector<double>& x, 
                                             const std::function<double(double)>& transform) {
    std::vector<double> result(x.size());
    
    for (size_t i = 0; i < x.size(); ++i) {
        result[i] = transform(x[i]);
    }
    
    return result;
}

// Function to generate bivariate normal samples
std::pair<std::vector<double>, std::vector<double>> generate_bivariate_normal(
    double mean_x, double mean_y, double std_dev_x, double std_dev_y, 
    double correlation, int num_samples) {
    
    if (correlation < -1.0 || correlation > 1.0) {
        throw std::invalid_argument("Correlation must be between -1 and 1");
    }
    
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist_z0(0, 1);
    std::normal_distribution<> dist_z1(0, 1);
    
    std::vector<double> x(num_samples);
    std::vector<double> y(num_samples);
    
    for (int i = 0; i < num_samples; ++i) {
        double z0 = dist_z0(gen);
        double z1 = dist_z1(gen);
        
        // Generate correlated normal random variables
        x[i] = mean_x + std_dev_x * z0;
        y[i] = mean_y + std_dev_y * (correlation * z0 + std::sqrt(1 - correlation * correlation) * z1);
    }
    
    return {x, y};
}

// Function to compute the conditional mean E[Y|X=x]
double conditional_mean(const std::vector<double>& x, const std::vector<double>& y, 
                       double x_value, double bandwidth = 0.5) {
    if (x.size() != y.size() || x.size() == 0) {
        throw std::invalid_argument("Vectors must have the same non-zero size");
    }
    
    double sum_weights = 0.0;
    double sum_weighted_y = 0.0;
    
    // Use a Gaussian kernel for weighting
    for (size_t i = 0; i < x.size(); ++i) {
        double distance = std::abs(x[i] - x_value);
        double weight = std::exp(-0.5 * std::pow(distance / bandwidth, 2));
        
        sum_weights += weight;
        sum_weighted_y += weight * y[i];
    }
    
    if (sum_weights < 1e-10) {
        return 0.0;  // Avoid division by zero
    }
    
    return sum_weighted_y / sum_weights;
}

// Function to compute the conditional variance Var[Y|X=x]
double conditional_variance(const std::vector<double>& x, const std::vector<double>& y, 
                           double x_value, double bandwidth = 0.5) {
    if (x.size() != y.size() || x.size() == 0) {
        throw std::invalid_argument("Vectors must have the same non-zero size");
    }
    
    double cond_mean = conditional_mean(x, y, x_value, bandwidth);
    
    double sum_weights = 0.0;
    double sum_weighted_squared_diff = 0.0;
    
    // Use a Gaussian kernel for weighting
    for (size_t i = 0; i < x.size(); ++i) {
        double distance = std::abs(x[i] - x_value);
        double weight = std::exp(-0.5 * std::pow(distance / bandwidth, 2));
        
        double diff = y[i] - cond_mean;
        sum_weights += weight;
        sum_weighted_squared_diff += weight * diff * diff;
    }
    
    if (sum_weights < 1e-10) {
        return 0.0;  // Avoid division by zero
    }
    
    return sum_weighted_squared_diff / sum_weights;
}

// Function to compute the expected value of a random variable
double expected_value(const std::vector<double>& x, const std::vector<double>& probabilities = {}) {
    if (x.empty()) {
        return 0.0;
    }
    
    if (probabilities.empty()) {
        // If no probabilities are provided, assume uniform distribution
        return compute_mean(x);
    }
    
    if (x.size() != probabilities.size()) {
        throw std::invalid_argument("Values and probabilities must have the same size");
    }
    
    double sum_prob = std::accumulate(probabilities.begin(), probabilities.end(), 0.0);
    if (std::abs(sum_prob - 1.0) > 1e-10) {
        throw std::invalid_argument("Probabilities must sum to 1");
    }
    
    double result = 0.0;
    for (size_t i = 0; i < x.size(); ++i) {
        result += x[i] * probabilities[i];
    }
    
    return result;
}

// Function to compute the expected value of a function of a random variable
double expected_value_of_function(const std::vector<double>& x, 
                                 const std::function<double(double)>& g,
                                 const std::vector<double>& probabilities = {}) {
    std::vector<double> g_x = transform_random_variable(x, g);
    return expected_value(g_x, probabilities);
}

// Function to print the joint distribution of two discrete random variables
void print_joint_distribution(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) {
        throw std::invalid_argument("Vectors must have the same non-zero size");
    }
    
    // Create a map of value pairs to count occurrences
    std::map<std::pair<double, double>, int> joint_counts;
    
    for (size_t i = 0; i < x.size(); ++i) {
        joint_counts[{x[i], y[i]}]++;
    }
    
    // Compute the total count
    int total_count = x.size();
    
    // Print the joint distribution
    std::cout << "Joint Distribution P(X,Y):" << std::endl;
    std::cout << std::setw(10) << "X" << std::setw(10) << "Y" 
              << std::setw(15) << "Count" << std::setw(15) << "Probability" << std::endl;
    
    for (const auto& entry : joint_counts) {
        double prob = static_cast<double>(entry.second) / total_count;
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << entry.first.first
                  << std::setw(10) << entry.first.second
                  << std::setw(15) << entry.second
                  << std::setw(15) << prob << std::endl;
    }
}

// Function to compute the marginal distribution of X
std::map<double, double> compute_marginal_x(const std::vector<double>& x, const std::vector<double>& y) {
    if (x.size() != y.size() || x.empty()) {
        throw std::invalid_argument("Vectors must have the same non-zero size");
    }
    
    // Count occurrences of each x value
    std::map<double, int> x_counts;
    
    for (size_t i = 0; i < x.size(); ++i) {
        x_counts[x[i]]++;
    }
    
    // Compute probabilities
    std::map<double, double> marginal_x;
    int total_count = x.size();
    
    for (const auto& entry : x_counts) {
        marginal_x[entry.first] = static_cast<double>(entry.second) / total_count;
    }
    
    return marginal_x;
}

// Function to print a marginal distribution
void print_marginal_distribution(const std::map<double, double>& marginal, const std::string& variable_name) {
    std::cout << "Marginal Distribution P(" << variable_name << "):" << std::endl;
    std::cout << std::setw(10) << variable_name << std::setw(15) << "Probability" << std::endl;
    
    for (const auto& entry : marginal) {
        std::cout << std::fixed << std::setprecision(2)
                  << std::setw(10) << entry.first
                  << std::setw(15) << entry.second << std::endl;
    }
}

int main() {
    std::cout << "Random Variables Examples\n";
    std::cout << "========================\n";
    
    const int num_samples = 10000;
    
    // Example 1: Transformations of Random Variables
    std::cout << "\nExample 1: Transformations of Random Variables\n";
    
    // Generate samples from a normal distribution
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> normal_dist(0, 1);
    
    std::vector<double> normal_samples(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        normal_samples[i] = normal_dist(gen);
    }
    
    std::cout << "Original normal distribution (mean = 0, std_dev = 1):" << std::endl;
    std::cout << "Mean: " << compute_mean(normal_samples) << std::endl;
    std::cout << "Variance: " << compute_variance(normal_samples) << std::endl;
    
    // Linear transformation: Y = 2X + 3
    auto linear_transform = [](double x) { return 2 * x + 3; };
    std::vector<double> linear_transformed = transform_random_variable(normal_samples, linear_transform);
    
    std::cout << "\nLinear transformation Y = 2X + 3:" << std::endl;
    std::cout << "Theoretical mean: " << 2 * 0 + 3 << " (= 2*E[X] + 3)" << std::endl;
    std::cout << "Theoretical variance: " << 2 * 2 * 1 << " (= 2^2 * Var[X])" << std::endl;
    std::cout << "Empirical mean: " << compute_mean(linear_transformed) << std::endl;
    std::cout << "Empirical variance: " << compute_variance(linear_transformed) << std::endl;
    
    // Nonlinear transformation: Y = X^2
    auto square_transform = [](double x) { return x * x; };
    std::vector<double> square_transformed = transform_random_variable(normal_samples, square_transform);
    
    std::cout << "\nNonlinear transformation Y = X^2:" << std::endl;
    std::cout << "Theoretical mean: 1 (= E[X^2] = Var[X] + E[X]^2 = 1 + 0^2 = 1)" << std::endl;
    std::cout << "Empirical mean: " << compute_mean(square_transformed) << std::endl;
    std::cout << "Empirical variance: " << compute_variance(square_transformed) << std::endl;
    
    // Example 2: Joint Distributions and Correlation
    std::cout << "\nExample 2: Joint Distributions and Correlation\n";
    
    // Generate bivariate normal samples with correlation 0.7
    double correlation = 0.7;
    auto [x_samples, y_samples] = generate_bivariate_normal(0, 0, 1, 1, correlation, num_samples);
    
    std::cout << "Bivariate normal with correlation " << correlation << ":" << std::endl;
    std::cout << "Mean of X: " << compute_mean(x_samples) << std::endl;
    std::cout << "Mean of Y: " << compute_mean(y_samples) << std::endl;
    std::cout << "Variance of X: " << compute_variance(x_samples) << std::endl;
    std::cout << "Variance of Y: " << compute_variance(y_samples) << std::endl;
    std::cout << "Covariance: " << compute_covariance(x_samples, y_samples) << std::endl;
    std::cout << "Correlation: " << compute_correlation(x_samples, y_samples) 
              << " (theoretical: " << correlation << ")" << std::endl;
    
    // Example 3: Conditional Distributions
    std::cout << "\nExample 3: Conditional Distributions\n";
    
    // For bivariate normal, E[Y|X=x] = μ_Y + ρ*(σ_Y/σ_X)*(x - μ_X)
    // For our case: E[Y|X=x] = 0 + 0.7*(1/1)*(x - 0) = 0.7*x
    
    double x_value = 1.5;
    double theoretical_cond_mean = 0.7 * x_value;
    double empirical_cond_mean = conditional_mean(x_samples, y_samples, x_value);
    
    std::cout << "Conditional mean E[Y|X=" << x_value << "]:" << std::endl;
    std::cout << "Theoretical: " << theoretical_cond_mean << std::endl;
    std::cout << "Empirical: " << empirical_cond_mean << std::endl;
    
    // For bivariate normal, Var[Y|X=x] = σ_Y^2 * (1 - ρ^2)
    // For our case: Var[Y|X=x] = 1 * (1 - 0.7^2) = 0.51
    
    double theoretical_cond_var = 1 * (1 - correlation * correlation);
    double empirical_cond_var = conditional_variance(x_samples, y_samples, x_value);
    
    std::cout << "Conditional variance Var[Y|X=" << x_value << "]:" << std::endl;
    std::cout << "Theoretical: " << theoretical_cond_var << std::endl;
    std::cout << "Empirical: " << empirical_cond_var << std::endl;
    
    // Example 4: Expected Values
    std::cout << "\nExample 4: Expected Values\n";
    
    // Generate discrete random variable
    std::vector<double> discrete_values = {1, 2, 3, 4, 5};
    std::vector<double> probabilities = {0.1, 0.2, 0.4, 0.2, 0.1};
    
    std::cout << "Discrete random variable X:" << std::endl;
    for (size_t i = 0; i < discrete_values.size(); ++i) {
        std::cout << "P(X = " << discrete_values[i] << ") = " << probabilities[i] << std::endl;
    }
    
    double ev = expected_value(discrete_values, probabilities);
    std::cout << "E[X] = " << ev << std::endl;
    
    // Expected value of g(X) = X^2
    auto g = [](double x) { return x * x; };
    double ev_g = expected_value_of_function(discrete_values, g, probabilities);
    
    std::cout << "E[X^2] = " << ev_g << std::endl;
    std::cout << "Var[X] = E[X^2] - E[X]^2 = " << ev_g - ev * ev << std::endl;
    
    // Example 5: Discrete Joint Distribution
    std::cout << "\nExample 5: Discrete Joint Distribution\n";
    
    // Generate samples from a discrete joint distribution
    std::vector<double> discrete_x;
    std::vector<double> discrete_y;
    
    // Die rolls: X = first die, Y = second die
    std::uniform_int_distribution<> die(1, 6);
    
    for (int i = 0; i < 1000; ++i) {
        discrete_x.push_back(die(gen));
        discrete_y.push_back(die(gen));
    }
    
    std::cout << "Simulating 1000 rolls of two dice:" << std::endl;
    
    // Print the joint distribution
    print_joint_distribution(discrete_x, discrete_y);
    
    // Compute and print the marginal distribution of X
    auto marginal_x = compute_marginal_x(discrete_x, discrete_y);
    print_marginal_distribution(marginal_x, "X");
    
    return 0;
}
