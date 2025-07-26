#include <iostream>
#include <vector>
#include <queue>
#include <algorithm>
#include <limits>
#include <unordered_map>
#include <string>

/**
 * Maximum Flow Algorithms Implementation
 * 
 * This file demonstrates various algorithms for solving the maximum flow problem
 * including Ford-Fulkerson with DFS, Edmonds-Karp with BFS, and Dinic's algorithm.
 */

const int INF = std::numeric_limits<int>::max();

// Edge structure for flow networks
struct Edge {
    int to;        // Destination vertex
    int capacity;  // Edge capacity
    int flow;      // Current flow
    int rev;       // Index of reverse edge in adjacency list
    
    Edge(int to, int cap, int rev) : to(to), capacity(cap), flow(0), rev(rev) {}
    
    int residual_capacity() const {
        return capacity - flow;
    }
};

// Flow network representation
class FlowNetwork {
public:
    int n;  // Number of vertices
    std::vector<std::vector<Edge>> graph;
    std::vector<std::string> vertex_names;
    
    FlowNetwork(int vertices) : n(vertices), graph(vertices) {
        for (int i = 0; i < n; ++i) {
            vertex_names.push_back("v" + std::to_string(i));
        }
    }
    
    void set_vertex_name(int v, const std::string& name) {
        if (v >= 0 && v < n) {
            vertex_names[v] = name;
        }
    }
    
    void add_edge(int from, int to, int capacity) {
        graph[from].emplace_back(to, capacity, graph[to].size());
        graph[to].emplace_back(from, 0, graph[from].size() - 1); // Reverse edge with 0 capacity
    }
    
    void print_network() const {
        std::cout << "Flow Network:" << std::endl;
        for (int u = 0; u < n; ++u) {
            for (const auto& edge : graph[u]) {
                if (edge.capacity > 0) { // Only print forward edges
                    std::cout << "  " << vertex_names[u] << " -> " << vertex_names[edge.to] 
                             << " (capacity: " << edge.capacity << ", flow: " << edge.flow << ")" << std::endl;
                }
            }
        }
    }
    
    void reset_flows() {
        for (int u = 0; u < n; ++u) {
            for (auto& edge : graph[u]) {
                edge.flow = 0;
            }
        }
    }
    
    int total_flow_from_source(int source) const {
        int total = 0;
        for (const auto& edge : graph[source]) {
            total += edge.flow;
        }
        return total;
    }
};

// Ford-Fulkerson algorithm with DFS
class FordFulkerson {
private:
    std::vector<bool> visited;
    
    int dfs(FlowNetwork& network, int u, int sink, int min_capacity) {
        if (u == sink) return min_capacity;
        
        visited[u] = true;
        
        for (auto& edge : network.graph[u]) {
            if (!visited[edge.to] && edge.residual_capacity() > 0) {
                int bottleneck = std::min(min_capacity, edge.residual_capacity());
                int flow = dfs(network, edge.to, sink, bottleneck);
                
                if (flow > 0) {
                    edge.flow += flow;
                    network.graph[edge.to][edge.rev].flow -= flow;
                    return flow;
                }
            }
        }
        
        return 0;
    }
    
public:
    int max_flow(FlowNetwork& network, int source, int sink) {
        std::cout << "Running Ford-Fulkerson Algorithm" << std::endl;
        std::cout << "================================" << std::endl;
        
        network.reset_flows();
        int total_flow = 0;
        int iteration = 0;
        
        while (true) {
            visited.assign(network.n, false);
            int path_flow = dfs(network, source, sink, INF);
            
            if (path_flow == 0) break;
            
            total_flow += path_flow;
            iteration++;
            
            std::cout << "Iteration " << iteration << ": Found augmenting path with flow " 
                     << path_flow << " (total: " << total_flow << ")" << std::endl;
        }
        
        std::cout << "Maximum flow: " << total_flow << std::endl;
        return total_flow;
    }
};

// Edmonds-Karp algorithm (Ford-Fulkerson with BFS)
class EdmondsKarp {
private:
    std::vector<int> parent;
    std::vector<int> parent_edge;
    
    bool bfs(const FlowNetwork& network, int source, int sink) {
        parent.assign(network.n, -1);
        parent_edge.assign(network.n, -1);
        std::vector<bool> visited(network.n, false);
        
        std::queue<int> q;
        q.push(source);
        visited[source] = true;
        
        while (!q.empty() && !visited[sink]) {
            int u = q.front();
            q.pop();
            
            for (int i = 0; i < network.graph[u].size(); ++i) {
                const auto& edge = network.graph[u][i];
                if (!visited[edge.to] && edge.residual_capacity() > 0) {
                    parent[edge.to] = u;
                    parent_edge[edge.to] = i;
                    visited[edge.to] = true;
                    q.push(edge.to);
                }
            }
        }
        
        return visited[sink];
    }
    
public:
    int max_flow(FlowNetwork& network, int source, int sink) {
        std::cout << "Running Edmonds-Karp Algorithm" << std::endl;
        std::cout << "==============================" << std::endl;
        
        network.reset_flows();
        int total_flow = 0;
        int iteration = 0;
        
        while (bfs(network, source, sink)) {
            // Find minimum capacity along the path
            int path_flow = INF;
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                int edge_idx = parent_edge[v];
                path_flow = std::min(path_flow, network.graph[u][edge_idx].residual_capacity());
            }
            
            // Update flows along the path
            for (int v = sink; v != source; v = parent[v]) {
                int u = parent[v];
                int edge_idx = parent_edge[v];
                network.graph[u][edge_idx].flow += path_flow;
                network.graph[v][network.graph[u][edge_idx].rev].flow -= path_flow;
            }
            
            total_flow += path_flow;
            iteration++;
            
            std::cout << "Iteration " << iteration << ": Found augmenting path with flow " 
                     << path_flow << " (total: " << total_flow << ")" << std::endl;
        }
        
        std::cout << "Maximum flow: " << total_flow << std::endl;
        return total_flow;
    }
};

// Dinic's algorithm
class Dinic {
private:
    std::vector<int> level;
    std::vector<int> iter;
    
    bool bfs(const FlowNetwork& network, int source, int sink) {
        level.assign(network.n, -1);
        level[source] = 0;
        
        std::queue<int> q;
        q.push(source);
        
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            
            for (const auto& edge : network.graph[u]) {
                if (level[edge.to] < 0 && edge.residual_capacity() > 0) {
                    level[edge.to] = level[u] + 1;
                    q.push(edge.to);
                }
            }
        }
        
        return level[sink] >= 0;
    }
    
    int dfs(FlowNetwork& network, int u, int sink, int pushed) {
        if (u == sink || pushed == 0) return pushed;
        
        for (int& cid = iter[u]; cid < network.graph[u].size(); ++cid) {
            auto& edge = network.graph[u][cid];
            if (level[edge.to] != level[u] + 1 || edge.residual_capacity() == 0) {
                continue;
            }
            
            int tr = dfs(network, edge.to, sink, std::min(pushed, edge.residual_capacity()));
            if (tr > 0) {
                edge.flow += tr;
                network.graph[edge.to][edge.rev].flow -= tr;
                return tr;
            }
        }
        
        return 0;
    }
    
public:
    int max_flow(FlowNetwork& network, int source, int sink) {
        std::cout << "Running Dinic's Algorithm" << std::endl;
        std::cout << "=========================" << std::endl;
        
        network.reset_flows();
        int total_flow = 0;
        int phase = 0;
        
        while (bfs(network, source, sink)) {
            iter.assign(network.n, 0);
            phase++;
            
            int phase_flow = 0;
            while (int pushed = dfs(network, source, sink, INF)) {
                total_flow += pushed;
                phase_flow += pushed;
            }
            
            std::cout << "Phase " << phase << ": Added flow " << phase_flow 
                     << " (total: " << total_flow << ")" << std::endl;
        }
        
        std::cout << "Maximum flow: " << total_flow << std::endl;
        return total_flow;
    }
};

// Min-cut finder
class MinCut {
public:
    static std::pair<std::vector<int>, std::vector<int>> find_min_cut(const FlowNetwork& network, int source, int sink) {
        std::vector<bool> reachable(network.n, false);
        std::queue<int> q;
        q.push(source);
        reachable[source] = true;
        
        // BFS on residual graph
        while (!q.empty()) {
            int u = q.front();
            q.pop();
            
            for (const auto& edge : network.graph[u]) {
                if (!reachable[edge.to] && edge.residual_capacity() > 0) {
                    reachable[edge.to] = true;
                    q.push(edge.to);
                }
            }
        }
        
        std::vector<int> source_side, sink_side;
        for (int i = 0; i < network.n; ++i) {
            if (reachable[i]) {
                source_side.push_back(i);
            } else {
                sink_side.push_back(i);
            }
        }
        
        return {source_side, sink_side};
    }
    
    static void print_min_cut(const FlowNetwork& network, const std::vector<int>& source_side, 
                             const std::vector<int>& sink_side) {
        std::cout << "\nMinimum Cut:" << std::endl;
        std::cout << "Source side: {";
        for (size_t i = 0; i < source_side.size(); ++i) {
            std::cout << network.vertex_names[source_side[i]];
            if (i < source_side.size() - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
        
        std::cout << "Sink side: {";
        for (size_t i = 0; i < sink_side.size(); ++i) {
            std::cout << network.vertex_names[sink_side[i]];
            if (i < sink_side.size() - 1) std::cout << ", ";
        }
        std::cout << "}" << std::endl;
        
        std::cout << "Cut edges:" << std::endl;
        int cut_capacity = 0;
        for (int u : source_side) {
            for (const auto& edge : network.graph[u]) {
                if (std::find(sink_side.begin(), sink_side.end(), edge.to) != sink_side.end() && edge.capacity > 0) {
                    std::cout << "  " << network.vertex_names[u] << " -> " << network.vertex_names[edge.to] 
                             << " (capacity: " << edge.capacity << ")" << std::endl;
                    cut_capacity += edge.capacity;
                }
            }
        }
        std::cout << "Total cut capacity: " << cut_capacity << std::endl;
    }
};

// Example problems
void solve_simple_network() {
    std::cout << "Simple Max Flow Example" << std::endl;
    std::cout << "=======================" << std::endl;
    
    FlowNetwork network(4);
    network.set_vertex_name(0, "S");
    network.set_vertex_name(1, "A");
    network.set_vertex_name(2, "B");
    network.set_vertex_name(3, "T");
    
    network.add_edge(0, 1, 10); // S -> A
    network.add_edge(0, 2, 8);  // S -> B
    network.add_edge(1, 2, 5);  // A -> B
    network.add_edge(1, 3, 10); // A -> T
    network.add_edge(2, 3, 10); // B -> T
    
    network.print_network();
    std::cout << std::endl;
    
    // Test different algorithms
    FordFulkerson ff;
    int ff_result = ff.max_flow(network, 0, 3);
    std::cout << std::endl;
    
    EdmondsKarp ek;
    int ek_result = ek.max_flow(network, 0, 3);
    std::cout << std::endl;
    
    Dinic dinic;
    int dinic_result = dinic.max_flow(network, 0, 3);
    
    // Find and print min cut
    auto cut = MinCut::find_min_cut(network, 0, 3);
    MinCut::print_min_cut(network, cut.first, cut.second);
    
    std::cout << "\nResults comparison:" << std::endl;
    std::cout << "Ford-Fulkerson: " << ff_result << std::endl;
    std::cout << "Edmonds-Karp: " << ek_result << std::endl;
    std::cout << "Dinic: " << dinic_result << std::endl;
    std::cout << std::endl;
}

void solve_bipartite_matching() {
    std::cout << "Bipartite Matching Example" << std::endl;
    std::cout << "==========================" << std::endl;
    
    // Model as max flow: source -> left side -> right side -> sink
    FlowNetwork network(8);
    network.set_vertex_name(0, "Source");
    network.set_vertex_name(1, "L1");
    network.set_vertex_name(2, "L2");
    network.set_vertex_name(3, "L3");
    network.set_vertex_name(4, "R1");
    network.set_vertex_name(5, "R2");
    network.set_vertex_name(6, "R3");
    network.set_vertex_name(7, "Sink");
    
    // Source to left side (capacity 1 each)
    network.add_edge(0, 1, 1);
    network.add_edge(0, 2, 1);
    network.add_edge(0, 3, 1);
    
    // Left to right side (possible matchings)
    network.add_edge(1, 4, 1); // L1 -> R1
    network.add_edge(1, 5, 1); // L1 -> R2
    network.add_edge(2, 5, 1); // L2 -> R2
    network.add_edge(2, 6, 1); // L2 -> R3
    network.add_edge(3, 4, 1); // L3 -> R1
    network.add_edge(3, 6, 1); // L3 -> R3
    
    // Right side to sink (capacity 1 each)
    network.add_edge(4, 7, 1);
    network.add_edge(5, 7, 1);
    network.add_edge(6, 7, 1);
    
    network.print_network();
    std::cout << std::endl;
    
    EdmondsKarp ek;
    int max_matching = ek.max_flow(network, 0, 7);
    
    std::cout << "Maximum matching size: " << max_matching << std::endl;
    std::cout << "Matching edges:" << std::endl;
    
    for (int u = 1; u <= 3; ++u) { // Left side vertices
        for (const auto& edge : network.graph[u]) {
            if (edge.to >= 4 && edge.to <= 6 && edge.flow > 0) {
                std::cout << "  " << network.vertex_names[u] << " matched with " 
                         << network.vertex_names[edge.to] << std::endl;
            }
        }
    }
    std::cout << std::endl;
}

void solve_supply_demand_network() {
    std::cout << "Supply and Demand Network" << std::endl;
    std::cout << "=========================" << std::endl;
    
    // Model with supersource and supersink
    FlowNetwork network(8);
    network.set_vertex_name(0, "SuperSource");
    network.set_vertex_name(1, "Supply1");
    network.set_vertex_name(2, "Supply2");
    network.set_vertex_name(3, "Intermediate1");
    network.set_vertex_name(4, "Intermediate2");
    network.set_vertex_name(5, "Demand1");
    network.set_vertex_name(6, "Demand2");
    network.set_vertex_name(7, "SuperSink");
    
    // Supersource to supply nodes
    network.add_edge(0, 1, 15); // Supply1 can provide 15 units
    network.add_edge(0, 2, 20); // Supply2 can provide 20 units
    
    // Supply to intermediate nodes
    network.add_edge(1, 3, 10);
    network.add_edge(1, 4, 8);
    network.add_edge(2, 3, 12);
    network.add_edge(2, 4, 15);
    
    // Intermediate to demand nodes
    network.add_edge(3, 5, 8);
    network.add_edge(3, 6, 12);
    network.add_edge(4, 5, 10);
    network.add_edge(4, 6, 8);
    
    // Demand nodes to supersink
    network.add_edge(5, 7, 12); // Demand1 needs 12 units
    network.add_edge(6, 7, 18); // Demand2 needs 18 units
    
    network.print_network();
    std::cout << std::endl;
    
    Dinic dinic;
    int max_flow = dinic.max_flow(network, 0, 7);
    
    std::cout << "Total flow delivered: " << max_flow << std::endl;
    std::cout << "Total demand: " << 12 + 18 << std::endl;
    
    if (max_flow == 30) {
        std::cout << "All demand can be satisfied!" << std::endl;
    } else {
        std::cout << "Cannot satisfy all demand." << std::endl;
    }
    std::cout << std::endl;
}

int main() {
    std::cout << "Maximum Flow Algorithms" << std::endl;
    std::cout << "======================" << std::endl << std::endl;
    
    // Example 1: Simple network
    solve_simple_network();
    
    // Example 2: Bipartite matching
    solve_bipartite_matching();
    
    // Example 3: Supply and demand network
    solve_supply_demand_network();
    
    return 0;
}
