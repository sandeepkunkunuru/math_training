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
 * Probability Distributions Implementation
 * 
 * This file demonstrates common probability distributions and their properties
 * in C++, which are essential for stochastic optimization and simulation.
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

// Function to compute the standard deviation of a vector
double compute_std_dev(const std::vector<double>& data) {
    return std::sqrt(compute_variance(data));
}

// Function to compute the median of a vector
double compute_median(std::vector<double> data) {
    if (data.empty()) {
        return 0.0;
    }
    
    std::sort(data.begin(), data.end());
    
    size_t n = data.size();
    if (n % 2 == 0) {
        return (data[n/2 - 1] + data[n/2]) / 2.0;
    } else {
        return data[n/2];
    }
}

// Function to compute the histogram of a vector
std::map<int, int> compute_histogram(const std::vector<double>& data, int num_bins, double min_val, double max_val) {
    std::map<int, int> histogram;
    
    double bin_width = (max_val - min_val) / num_bins;
    
    for (double value : data) {
        int bin = static_cast<int>((value - min_val) / bin_width);
        
        // Handle edge cases
        if (bin < 0) bin = 0;
        if (bin >= num_bins) bin = num_bins - 1;
        
        histogram[bin]++;
    }
    
    return histogram;
}

// Function to print a histogram
void print_histogram(const std::map<int, int>& histogram, int num_bins, double min_val, double max_val) {
    double bin_width = (max_val - min_val) / num_bins;
    
    std::cout << "Histogram:" << std::endl;
    
    int max_count = 0;
    for (const auto& bin : histogram) {
        max_count = std::max(max_count, bin.second);
    }
    
    const int max_bar_width = 50;
    
    for (int i = 0; i < num_bins; ++i) {
        double bin_start = min_val + i * bin_width;
        double bin_end = bin_start + bin_width;
        
        int count = histogram.count(i) ? histogram.at(i) : 0;
        
        int bar_width = max_count > 0 ? static_cast<int>(static_cast<double>(count) / max_count * max_bar_width) : 0;
        
        std::cout << std::fixed << std::setprecision(2) 
                  << "[" << bin_start << ", " << bin_end << "): " 
                  << std::string(bar_width, '*') << " " << count << std::endl;
    }
}

// Function to generate samples from a uniform distribution
std::vector<double> generate_uniform_samples(double a, double b, int num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::uniform_real_distribution<> dist(a, b);
    
    std::vector<double> samples(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        samples[i] = dist(gen);
    }
    
    return samples;
}

// Function to generate samples from a normal distribution
std::vector<double> generate_normal_samples(double mean, double std_dev, int num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::normal_distribution<> dist(mean, std_dev);
    
    std::vector<double> samples(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        samples[i] = dist(gen);
    }
    
    return samples;
}

// Function to generate samples from an exponential distribution
std::vector<double> generate_exponential_samples(double lambda, int num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::exponential_distribution<> dist(lambda);
    
    std::vector<double> samples(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        samples[i] = dist(gen);
    }
    
    return samples;
}

// Function to generate samples from a Poisson distribution
std::vector<int> generate_poisson_samples(double lambda, int num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::poisson_distribution<> dist(lambda);
    
    std::vector<int> samples(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        samples[i] = dist(gen);
    }
    
    return samples;
}

// Function to generate samples from a binomial distribution
std::vector<int> generate_binomial_samples(int n, double p, int num_samples) {
    std::random_device rd;
    std::mt19937 gen(rd());
    std::binomial_distribution<> dist(n, p);
    
    std::vector<int> samples(num_samples);
    for (int i = 0; i < num_samples; ++i) {
        samples[i] = dist(gen);
    }
    
    return samples;
}

// Function to compute the PDF of a normal distribution
double normal_pdf(double x, double mean, double std_dev) {
    static const double inv_sqrt_2pi = 0.3989422804014327;  // 1 / sqrt(2 * PI)
    double z = (x - mean) / std_dev;
    return inv_sqrt_2pi / std_dev * std::exp(-0.5 * z * z);
}

// Function to compute the CDF of a normal distribution
double normal_cdf(double x, double mean, double std_dev) {
    double z = (x - mean) / std_dev;
    return 0.5 * (1 + std::erf(z / std::sqrt(2)));
}

// Function to compute the PDF of an exponential distribution
double exponential_pdf(double x, double lambda) {
    if (x < 0) {
        return 0.0;
    }
    return lambda * std::exp(-lambda * x);
}

// Function to compute the CDF of an exponential distribution
double exponential_cdf(double x, double lambda) {
    if (x < 0) {
        return 0.0;
    }
    return 1.0 - std::exp(-lambda * x);
}

// Function to print statistics of a sample
void print_statistics(const std::vector<double>& samples, const std::string& distribution_name) {
    std::cout << "Statistics for " << distribution_name << " distribution:" << std::endl;
    std::cout << "Number of samples: " << samples.size() << std::endl;
    std::cout << "Mean: " << compute_mean(samples) << std::endl;
    std::cout << "Variance: " << compute_variance(samples) << std::endl;
    std::cout << "Standard deviation: " << compute_std_dev(samples) << std::endl;
    std::cout << "Median: " << compute_median(samples) << std::endl;
    std::cout << "Min: " << *std::min_element(samples.begin(), samples.end()) << std::endl;
    std::cout << "Max: " << *std::max_element(samples.begin(), samples.end()) << std::endl;
    std::cout << std::endl;
}

int main() {
    std::cout << "Probability Distributions Examples\n";
    std::cout << "================================\n";
    
    const int num_samples = 10000;
    
    // Example 1: Uniform Distribution
    std::cout << "\nExample 1: Uniform Distribution\n";
    
    double a = 0.0;  // Lower bound
    double b = 1.0;  // Upper bound
    
    std::vector<double> uniform_samples = generate_uniform_samples(a, b, num_samples);
    
    std::cout << "Uniform distribution parameters: a = " << a << ", b = " << b << std::endl;
    std::cout << "Theoretical mean: " << (a + b) / 2 << std::endl;
    std::cout << "Theoretical variance: " << std::pow(b - a, 2) / 12 << std::endl;
    
    print_statistics(uniform_samples, "Uniform");
    
    // Compute and print histogram
    auto uniform_histogram = compute_histogram(uniform_samples, 20, a, b);
    print_histogram(uniform_histogram, 20, a, b);
    
    // Example 2: Normal Distribution
    std::cout << "\nExample 2: Normal Distribution\n";
    
    double mean = 0.0;
    double std_dev = 1.0;
    
    std::vector<double> normal_samples = generate_normal_samples(mean, std_dev, num_samples);
    
    std::cout << "Normal distribution parameters: mean = " << mean << ", std_dev = " << std_dev << std::endl;
    std::cout << "Theoretical mean: " << mean << std::endl;
    std::cout << "Theoretical variance: " << std_dev * std_dev << std::endl;
    
    print_statistics(normal_samples, "Normal");
    
    // Compute and print histogram
    auto normal_histogram = compute_histogram(normal_samples, 20, mean - 4 * std_dev, mean + 4 * std_dev);
    print_histogram(normal_histogram, 20, mean - 4 * std_dev, mean + 4 * std_dev);
    
    // Example 3: Exponential Distribution
    std::cout << "\nExample 3: Exponential Distribution\n";
    
    double lambda = 2.0;
    
    std::vector<double> exponential_samples = generate_exponential_samples(lambda, num_samples);
    
    std::cout << "Exponential distribution parameter: lambda = " << lambda << std::endl;
    std::cout << "Theoretical mean: " << 1.0 / lambda << std::endl;
    std::cout << "Theoretical variance: " << 1.0 / (lambda * lambda) << std::endl;
    
    print_statistics(exponential_samples, "Exponential");
    
    // Compute and print histogram
    auto exponential_histogram = compute_histogram(exponential_samples, 20, 0, 5.0 / lambda);
    print_histogram(exponential_histogram, 20, 0, 5.0 / lambda);
    
    // Example 4: Poisson Distribution
    std::cout << "\nExample 4: Poisson Distribution\n";
    
    double poisson_lambda = 5.0;
    
    std::vector<int> poisson_samples = generate_poisson_samples(poisson_lambda, num_samples);
    
    // Convert to double for our statistics functions
    std::vector<double> poisson_samples_double(poisson_samples.begin(), poisson_samples.end());
    
    std::cout << "Poisson distribution parameter: lambda = " << poisson_lambda << std::endl;
    std::cout << "Theoretical mean: " << poisson_lambda << std::endl;
    std::cout << "Theoretical variance: " << poisson_lambda << std::endl;
    
    print_statistics(poisson_samples_double, "Poisson");
    
    // Compute and print histogram
    auto poisson_histogram = compute_histogram(poisson_samples_double, 20, 0, poisson_lambda * 3);
    print_histogram(poisson_histogram, 20, 0, poisson_lambda * 3);
    
    // Example 5: Binomial Distribution
    std::cout << "\nExample 5: Binomial Distribution\n";
    
    int n = 20;
    double p = 0.3;
    
    std::vector<int> binomial_samples = generate_binomial_samples(n, p, num_samples);
    
    // Convert to double for our statistics functions
    std::vector<double> binomial_samples_double(binomial_samples.begin(), binomial_samples.end());
    
    std::cout << "Binomial distribution parameters: n = " << n << ", p = " << p << std::endl;
    std::cout << "Theoretical mean: " << n * p << std::endl;
    std::cout << "Theoretical variance: " << n * p * (1 - p) << std::endl;
    
    print_statistics(binomial_samples_double, "Binomial");
    
    // Compute and print histogram
    auto binomial_histogram = compute_histogram(binomial_samples_double, n + 1, -0.5, n + 0.5);
    print_histogram(binomial_histogram, n + 1, -0.5, n + 0.5);
    
    // Example 6: PDF and CDF calculations
    std::cout << "\nExample 6: PDF and CDF Calculations\n";
    
    std::cout << "Normal distribution (mean = 0, std_dev = 1):" << std::endl;
    std::cout << "PDF at x = -1: " << normal_pdf(-1, 0, 1) << std::endl;
    std::cout << "PDF at x = 0: " << normal_pdf(0, 0, 1) << std::endl;
    std::cout << "PDF at x = 1: " << normal_pdf(1, 0, 1) << std::endl;
    std::cout << "CDF at x = -1: " << normal_cdf(-1, 0, 1) << std::endl;
    std::cout << "CDF at x = 0: " << normal_cdf(0, 0, 1) << std::endl;
    std::cout << "CDF at x = 1: " << normal_cdf(1, 0, 1) << std::endl;
    
    std::cout << "\nExponential distribution (lambda = 2):" << std::endl;
    std::cout << "PDF at x = 0: " << exponential_pdf(0, 2) << std::endl;
    std::cout << "PDF at x = 0.5: " << exponential_pdf(0.5, 2) << std::endl;
    std::cout << "PDF at x = 1: " << exponential_pdf(1, 2) << std::endl;
    std::cout << "CDF at x = 0: " << exponential_cdf(0, 2) << std::endl;
    std::cout << "CDF at x = 0.5: " << exponential_cdf(0.5, 2) << std::endl;
    std::cout << "CDF at x = 1: " << exponential_cdf(1, 2) << std::endl;
    
    return 0;
}
