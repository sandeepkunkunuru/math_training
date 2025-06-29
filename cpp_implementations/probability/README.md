# Probability and Statistics for Optimization

This directory contains C++ implementations of key probability and statistics concepts that are fundamental to optimization algorithms and techniques. These implementations serve as educational examples to understand the mathematical foundations of stochastic optimization methods.

## Contents

1. **Probability Distributions** (`probability_distributions.cpp`)
   - Implementation of common probability distributions (uniform, normal, exponential, Poisson, binomial)
   - Computation of statistical measures (mean, variance, median)
   - PDF and CDF calculations
   - Visualization through text-based histograms

2. **Random Variables** (`random_variables.cpp`)
   - Concepts of random variables and their transformations
   - Joint and marginal distributions
   - Computation of covariance and correlation
   - Conditional probability and expected values
   - Bivariate normal distribution sampling

3. **Monte Carlo Methods** (`monte_carlo.cpp`)
   - Estimation of mathematical constants (π)
   - Monte Carlo integration for single and multi-dimensional integrals
   - Expected value estimation
   - Probability estimation
   - Rejection sampling
   - Importance sampling
   - Confidence interval estimation
   - Simulation of stochastic processes (Geometric Brownian Motion)

4. **Stochastic Processes** (`stochastic_processes.cpp`)
   - Random Walk simulation
   - Poisson Process
   - Brownian Motion (Wiener Process)
   - Geometric Brownian Motion
   - Mean-Reverting Process (Ornstein-Uhlenbeck)
   - Markov Chains and stationary distributions

## Relevance to Optimization

### Probability Distributions
Understanding probability distributions is essential for:
- Modeling uncertainty in optimization problems
- Formulating stochastic programming problems
- Analyzing the robustness of solutions
- Implementing sampling-based optimization algorithms

### Random Variables
Random variables are fundamental to:
- Stochastic optimization models
- Risk assessment in optimization solutions
- Modeling correlations between uncertain parameters
- Scenario generation for stochastic programming

### Monte Carlo Methods
Monte Carlo techniques are widely used in:
- Stochastic gradient descent and its variants
- Simulation-based optimization
- Approximating complex integrals in objective functions
- Bayesian optimization
- Reinforcement learning algorithms
- Estimating expected values in stochastic programming

### Stochastic Processes
Stochastic processes provide models for:
- Time-dependent optimization problems
- Financial optimization (portfolio optimization)
- Dynamic programming with uncertainty
- Markov Decision Processes
- Queueing optimization problems
- Inventory and supply chain optimization

## Connection to OR-Tools

These probability and statistics implementations provide the mathematical foundation for several advanced features in Google's OR-Tools:

1. **Constraint Programming with Uncertainty**:
   - Modeling random variables in constraint satisfaction problems
   - Chance-constrained programming

2. **Vehicle Routing with Stochastic Demands**:
   - Modeling uncertain demands using probability distributions
   - Monte Carlo simulation for evaluating routing policies

3. **Stochastic Linear Programming**:
   - Scenario generation using Monte Carlo methods
   - Sampling-based approaches to stochastic optimization

4. **Reinforcement Learning Integration**:
   - Markov Decision Process modeling
   - Monte Carlo policy evaluation

## Building and Running

To build all the examples in this directory:

```bash
cd /path/to/math_training
mkdir build && cd build
cmake ..
make
```

To run a specific example:

```bash
./cpp_implementations/probability/probability_distributions
./cpp_implementations/probability/random_variables
./cpp_implementations/probability/monte_carlo
./cpp_implementations/probability/stochastic_processes
```

## Further Reading

- "Probability and Statistics for Engineering and the Sciences" by Jay L. Devore
- "Monte Carlo Methods in Financial Engineering" by Paul Glasserman
- "Introduction to Stochastic Processes" by Gregory F. Lawler
- "Stochastic Programming" by John R. Birge and François Louveaux
