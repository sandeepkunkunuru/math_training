# Scheduling Algorithms Implementation

This directory contains C++ implementations of scheduling algorithms and resource allocation problems.

## Files

### `job_shop_scheduling.cpp`
Complete job shop scheduling implementation:

- **Problem Representation**:
  - **Job Structure**: Sequence of operations with precedence constraints
  - **Machine Structure**: Resource scheduling and conflict detection
  - **Operation Structure**: Processing times and machine requirements
- **Scheduling Algorithms**:
  - **Priority Rules**: SPT, LPT, FCFS heuristics
  - **Local Search**: Operation swapping for improvement
- **Schedule Validation**: Precedence and resource constraint checking

### Example Problems

1. **Simple 3x3 Instance**: 3 jobs, 3 machines with different priority rules
2. **Larger 4x3 Instance**: More complex scheduling with comparison analysis

## Key Concepts Demonstrated

- **Job Shop Modeling**: Jobs, operations, machines, and constraints
- **Priority-Based Scheduling**: Different dispatching rules and their effects
- **Local Search**: Neighborhood-based improvement methods
- **Schedule Representation**: Gantt chart-style output and makespan calculation

## Priority Rules Implemented

- **SPT (Shortest Processing Time)**: Prioritize operations with shorter duration
- **LPT (Longest Processing Time)**: Prioritize operations with longer duration  
- **FCFS (First Come First Served)**: Process operations in job order

## Algorithm Features

- **Constraint Handling**: Automatic precedence and resource conflict checking
- **Makespan Optimization**: Minimize total completion time
- **Solution Validation**: Verify feasibility of generated schedules
- **Performance Comparison**: Multiple algorithms on same instance

## Usage

```bash
g++ -std=c++17 -O2 job_shop_scheduling.cpp -o jsp_solver
./jsp_solver
```

## Learning Objectives

- Understand job shop scheduling formulation
- Learn priority-based dispatching rules
- Practice constraint satisfaction in scheduling
- Explore local search for schedule improvement

This implementation demonstrates core concepts used in OR-Tools scheduling module and provides foundation for more advanced scheduling algorithms.
