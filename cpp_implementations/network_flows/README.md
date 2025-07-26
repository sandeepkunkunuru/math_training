# Network Flows Implementation

This directory contains C++ implementations of network flow algorithms and applications.

## Files

### `max_flow.cpp`
Comprehensive implementation of maximum flow algorithms:

- **FlowNetwork Class**: Graph representation with residual capacity tracking
- **Algorithms Implemented**:
  - **Ford-Fulkerson**: DFS-based augmenting path search
  - **Edmonds-Karp**: BFS-based (shortest augmenting path)
  - **Dinic's Algorithm**: Level-based blocking flows
- **Min-Cut Finding**: Identifies minimum cut after max flow computation

### Example Applications

1. **Simple Network**: Basic max flow demonstration
2. **Bipartite Matching**: Maximum matching via max flow reduction
3. **Supply-Demand Network**: Multi-commodity flow with supersource/supersink

## Key Concepts Demonstrated

- **Residual Networks**: Forward and backward edge management
- **Augmenting Paths**: Different strategies for path finding
- **Max-Flow Min-Cut Theorem**: Relationship between flow and cuts
- **Algorithm Complexity**: Comparison of different approaches

## Algorithm Features

- **Ford-Fulkerson**: O(E * max_flow) time complexity
- **Edmonds-Karp**: O(V * E²) time complexity  
- **Dinic's Algorithm**: O(V² * E) time complexity
- **Min-Cut**: Linear time after max flow computation

## Applications Shown

- **Network Capacity**: Basic flow maximization
- **Matching Problems**: Bipartite graph matching
- **Resource Allocation**: Supply and demand satisfaction

## Usage

```bash
g++ -std=c++17 -O2 max_flow.cpp -o max_flow
./max_flow
```

## Learning Objectives

- Understand flow network modeling
- Learn different max flow algorithms
- Explore practical applications
- Practice graph algorithm implementation

This implementation covers the fundamental algorithms used in OR-Tools' graph module and provides a foundation for more advanced network optimization problems.
