#include <iostream>
#include <vector>
#include <string>
#include <map>
#include <iomanip>
#include <algorithm>

/**
 * Supply Chain Optimization Example
 * 
 * This example demonstrates a real-world supply chain optimization problem
 * that combines multiple optimization techniques we've implemented:
 * - Linear programming for production planning
 * - Network flows for transportation
 * - Integer programming for facility location decisions
 */

// Supply chain components
struct Supplier {
    std::string name;
    int capacity;
    double unit_cost;
    std::vector<double> transport_costs; // to each warehouse
    
    Supplier(const std::string& n, int cap, double cost) 
        : name(n), capacity(cap), unit_cost(cost) {}
};

struct Warehouse {
    std::string name;
    int capacity;
    double fixed_cost;
    std::vector<double> transport_costs; // to each customer
    bool is_open;
    
    Warehouse(const std::string& n, int cap, double cost) 
        : name(n), capacity(cap), fixed_cost(cost), is_open(false) {}
};

struct Customer {
    std::string name;
    int demand;
    
    Customer(const std::string& n, int dem) : name(n), demand(dem) {}
};

// Supply chain network
class SupplyChainNetwork {
private:
    std::vector<Supplier> suppliers;
    std::vector<Warehouse> warehouses;
    std::vector<Customer> customers;
    
    // Decision variables (simplified representation)
    std::vector<std::vector<double>> supplier_to_warehouse; // flow amounts
    std::vector<std::vector<double>> warehouse_to_customer; // flow amounts
    std::vector<bool> warehouse_decisions; // open/close decisions
    
public:
    void add_supplier(const std::string& name, int capacity, double unit_cost) {
        suppliers.emplace_back(name, capacity, unit_cost);
    }
    
    void add_warehouse(const std::string& name, int capacity, double fixed_cost) {
        warehouses.emplace_back(name, capacity, fixed_cost);
    }
    
    void add_customer(const std::string& name, int demand) {
        customers.emplace_back(name, demand);
    }
    
    void set_transport_costs() {
        // Set up transport cost matrices (simplified with distance-based costs)
        for (auto& supplier : suppliers) {
            supplier.transport_costs.resize(warehouses.size());
            for (size_t w = 0; w < warehouses.size(); ++w) {
                // Simplified: cost based on supplier and warehouse indices
                supplier.transport_costs[w] = 2.0 + 0.5 * ((supplier.name[0] + warehouses[w].name[0]) % 10);
            }
        }
        
        for (auto& warehouse : warehouses) {
            warehouse.transport_costs.resize(customers.size());
            for (size_t c = 0; c < customers.size(); ++c) {
                // Simplified: cost based on warehouse and customer indices
                warehouse.transport_costs[c] = 1.0 + 0.3 * ((warehouse.name[0] + customers[c].name[0]) % 8);
            }
        }
    }
    
    void print_network_info() {
        std::cout << "Supply Chain Network Configuration" << std::endl;
        std::cout << "==================================" << std::endl;
        
        std::cout << "\nSuppliers:" << std::endl;
        for (const auto& supplier : suppliers) {
            std::cout << "  " << supplier.name << ": capacity=" << supplier.capacity 
                     << ", unit_cost=$" << std::fixed << std::setprecision(2) << supplier.unit_cost << std::endl;
        }
        
        std::cout << "\nWarehouses:" << std::endl;
        for (const auto& warehouse : warehouses) {
            std::cout << "  " << warehouse.name << ": capacity=" << warehouse.capacity 
                     << ", fixed_cost=$" << std::fixed << std::setprecision(2) << warehouse.fixed_cost << std::endl;
        }
        
        std::cout << "\nCustomers:" << std::endl;
        for (const auto& customer : customers) {
            std::cout << "  " << customer.name << ": demand=" << customer.demand << std::endl;
        }
        
        std::cout << "\nTransport Costs (Supplier -> Warehouse):" << std::endl;
        std::cout << std::setw(12) << "Supplier";
        for (const auto& warehouse : warehouses) {
            std::cout << std::setw(10) << warehouse.name;
        }
        std::cout << std::endl;
        
        for (const auto& supplier : suppliers) {
            std::cout << std::setw(12) << supplier.name;
            for (double cost : supplier.transport_costs) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(2) << cost;
            }
            std::cout << std::endl;
        }
        
        std::cout << "\nTransport Costs (Warehouse -> Customer):" << std::endl;
        std::cout << std::setw(12) << "Warehouse";
        for (const auto& customer : customers) {
            std::cout << std::setw(10) << customer.name;
        }
        std::cout << std::endl;
        
        for (const auto& warehouse : warehouses) {
            std::cout << std::setw(12) << warehouse.name;
            for (double cost : warehouse.transport_costs) {
                std::cout << std::setw(10) << std::fixed << std::setprecision(2) << cost;
            }
            std::cout << std::endl;
        }
        std::cout << std::endl;
    }
    
    // Simplified optimization using greedy heuristics
    // In practice, this would use LP/MIP solvers
    double optimize_supply_chain() {
        std::cout << "Optimizing Supply Chain" << std::endl;
        std::cout << "=======================" << std::endl;
        
        // Initialize decision variables
        supplier_to_warehouse.assign(suppliers.size(), std::vector<double>(warehouses.size(), 0.0));
        warehouse_to_customer.assign(warehouses.size(), std::vector<double>(customers.size(), 0.0));
        warehouse_decisions.assign(warehouses.size(), false);
        
        // Step 1: Facility location decisions (simplified greedy approach)
        std::cout << "Step 1: Facility Location Decisions" << std::endl;
        
        // Calculate total demand
        int total_demand = 0;
        for (const auto& customer : customers) {
            total_demand += customer.demand;
        }
        
        // Greedy warehouse selection based on cost-effectiveness
        std::vector<std::pair<double, int>> warehouse_scores;
        for (size_t w = 0; w < warehouses.size(); ++w) {
            double avg_transport_cost = 0.0;
            for (const auto& customer : customers) {
                avg_transport_cost += warehouses[w].transport_costs[customer.demand % customers.size()];
            }
            avg_transport_cost /= customers.size();
            
            double score = warehouses[w].fixed_cost + avg_transport_cost * total_demand;
            warehouse_scores.emplace_back(score, w);
        }
        
        std::sort(warehouse_scores.begin(), warehouse_scores.end());
        
        // Open warehouses until capacity is sufficient
        int total_warehouse_capacity = 0;
        for (const auto& score_pair : warehouse_scores) {
            int w = score_pair.second;
            warehouse_decisions[w] = true;
            warehouses[w].is_open = true;
            total_warehouse_capacity += warehouses[w].capacity;
            
            std::cout << "  Opening warehouse: " << warehouses[w].name << std::endl;
            
            if (total_warehouse_capacity >= total_demand) {
                break;
            }
        }
        
        // Step 2: Transportation optimization (simplified)
        std::cout << "\nStep 2: Transportation Optimization" << std::endl;
        
        // Satisfy customer demands greedily
        std::vector<int> remaining_demand(customers.size());
        for (size_t c = 0; c < customers.size(); ++c) {
            remaining_demand[c] = customers[c].demand;
        }
        
        for (size_t c = 0; c < customers.size(); ++c) {
            while (remaining_demand[c] > 0) {
                // Find best warehouse to serve this customer
                int best_warehouse = -1;
                double best_cost = 1e9;
                
                for (size_t w = 0; w < warehouses.size(); ++w) {
                    if (warehouses[w].is_open && warehouses[w].capacity > 0) {
                        if (warehouses[w].transport_costs[c] < best_cost) {
                            best_cost = warehouses[w].transport_costs[c];
                            best_warehouse = w;
                        }
                    }
                }
                
                if (best_warehouse == -1) break;
                
                // Allocate as much as possible
                int allocation = std::min(remaining_demand[c], warehouses[best_warehouse].capacity);
                warehouse_to_customer[best_warehouse][c] += allocation;
                remaining_demand[c] -= allocation;
                warehouses[best_warehouse].capacity -= allocation;
            }
        }
        
        // Step 3: Supply allocation
        std::cout << "\nStep 3: Supply Allocation" << std::endl;
        
        for (size_t w = 0; w < warehouses.size(); ++w) {
            if (!warehouses[w].is_open) continue;
            
            // Calculate total flow into this warehouse
            double total_outflow = 0.0;
            for (size_t c = 0; c < customers.size(); ++c) {
                total_outflow += warehouse_to_customer[w][c];
            }
            
            // Allocate from suppliers to meet this demand
            double remaining_need = total_outflow;
            for (size_t s = 0; s < suppliers.size() && remaining_need > 0; ++s) {
                double allocation = std::min(remaining_need, (double)suppliers[s].capacity);
                supplier_to_warehouse[s][w] = allocation;
                remaining_need -= allocation;
            }
        }
        
        // Calculate total cost
        double total_cost = calculate_total_cost();
        
        std::cout << "\nOptimization completed!" << std::endl;
        std::cout << "Total cost: $" << std::fixed << std::setprecision(2) << total_cost << std::endl;
        
        return total_cost;
    }
    
    double calculate_total_cost() {
        double total_cost = 0.0;
        
        // Fixed costs for open warehouses
        for (size_t w = 0; w < warehouses.size(); ++w) {
            if (warehouse_decisions[w]) {
                total_cost += warehouses[w].fixed_cost;
            }
        }
        
        // Production costs
        for (size_t s = 0; s < suppliers.size(); ++s) {
            for (size_t w = 0; w < warehouses.size(); ++w) {
                total_cost += supplier_to_warehouse[s][w] * suppliers[s].unit_cost;
            }
        }
        
        // Transportation costs (supplier to warehouse)
        for (size_t s = 0; s < suppliers.size(); ++s) {
            for (size_t w = 0; w < warehouses.size(); ++w) {
                total_cost += supplier_to_warehouse[s][w] * suppliers[s].transport_costs[w];
            }
        }
        
        // Transportation costs (warehouse to customer)
        for (size_t w = 0; w < warehouses.size(); ++w) {
            for (size_t c = 0; c < customers.size(); ++c) {
                total_cost += warehouse_to_customer[w][c] * warehouses[w].transport_costs[c];
            }
        }
        
        return total_cost;
    }
    
    void print_solution() {
        std::cout << "\nOptimal Supply Chain Solution" << std::endl;
        std::cout << "=============================" << std::endl;
        
        // Warehouse decisions
        std::cout << "\nWarehouse Decisions:" << std::endl;
        for (size_t w = 0; w < warehouses.size(); ++w) {
            std::cout << "  " << warehouses[w].name << ": " 
                     << (warehouse_decisions[w] ? "OPEN" : "CLOSED") << std::endl;
        }
        
        // Supply flows
        std::cout << "\nSupplier -> Warehouse Flows:" << std::endl;
        for (size_t s = 0; s < suppliers.size(); ++s) {
            for (size_t w = 0; w < warehouses.size(); ++w) {
                if (supplier_to_warehouse[s][w] > 0.1) {
                    std::cout << "  " << suppliers[s].name << " -> " << warehouses[w].name 
                             << ": " << std::fixed << std::setprecision(1) << supplier_to_warehouse[s][w] << " units" << std::endl;
                }
            }
        }
        
        // Distribution flows
        std::cout << "\nWarehouse -> Customer Flows:" << std::endl;
        for (size_t w = 0; w < warehouses.size(); ++w) {
            for (size_t c = 0; c < customers.size(); ++c) {
                if (warehouse_to_customer[w][c] > 0.1) {
                    std::cout << "  " << warehouses[w].name << " -> " << customers[c].name 
                             << ": " << std::fixed << std::setprecision(1) << warehouse_to_customer[w][c] << " units" << std::endl;
                }
            }
        }
        
        // Cost breakdown
        std::cout << "\nCost Breakdown:" << std::endl;
        double fixed_costs = 0.0;
        for (size_t w = 0; w < warehouses.size(); ++w) {
            if (warehouse_decisions[w]) {
                fixed_costs += warehouses[w].fixed_cost;
            }
        }
        std::cout << "  Fixed costs: $" << std::fixed << std::setprecision(2) << fixed_costs << std::endl;
        
        double production_costs = 0.0;
        for (size_t s = 0; s < suppliers.size(); ++s) {
            for (size_t w = 0; w < warehouses.size(); ++w) {
                production_costs += supplier_to_warehouse[s][w] * suppliers[s].unit_cost;
            }
        }
        std::cout << "  Production costs: $" << std::fixed << std::setprecision(2) << production_costs << std::endl;
        
        double transport_costs = calculate_total_cost() - fixed_costs - production_costs;
        std::cout << "  Transportation costs: $" << std::fixed << std::setprecision(2) << transport_costs << std::endl;
        std::cout << "  Total cost: $" << std::fixed << std::setprecision(2) << calculate_total_cost() << std::endl;
    }
};

int main() {
    std::cout << "Supply Chain Optimization Example" << std::endl;
    std::cout << "=================================" << std::endl << std::endl;
    
    // Create supply chain network
    SupplyChainNetwork network;
    
    // Add suppliers
    network.add_supplier("Supplier_A", 150, 5.0);
    network.add_supplier("Supplier_B", 200, 4.5);
    network.add_supplier("Supplier_C", 100, 6.0);
    
    // Add warehouses
    network.add_warehouse("Warehouse_1", 120, 1000.0);
    network.add_warehouse("Warehouse_2", 150, 1200.0);
    network.add_warehouse("Warehouse_3", 100, 800.0);
    
    // Add customers
    network.add_customer("Customer_X", 80);
    network.add_customer("Customer_Y", 60);
    network.add_customer("Customer_Z", 90);
    
    // Set up transport costs
    network.set_transport_costs();
    
    // Print network information
    network.print_network_info();
    
    // Optimize the supply chain
    double optimal_cost = network.optimize_supply_chain();
    
    // Print solution
    network.print_solution();
    
    std::cout << "\n\nOptimization Techniques Used:" << std::endl;
    std::cout << "1. Facility Location: Integer programming concepts (binary decisions)" << std::endl;
    std::cout << "2. Transportation: Network flow optimization" << std::endl;
    std::cout << "3. Production Planning: Linear programming principles" << std::endl;
    std::cout << "4. Heuristic Methods: Greedy algorithms for practical solutions" << std::endl;
    
    std::cout << "\nReal-World Extensions:" << std::endl;
    std::cout << "• Multi-period planning with inventory costs" << std::endl;
    std::cout << "• Stochastic demand and supply uncertainties" << std::endl;
    std::cout << "• Multiple product types and substitution" << std::endl;
    std::cout << "• Capacity expansion decisions over time" << std::endl;
    std::cout << "• Environmental and sustainability constraints" << std::endl;
    
    return 0;
}
