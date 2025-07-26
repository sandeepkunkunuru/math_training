#include <iostream>
#include <vector>
#include <algorithm>
#include <queue>
#include <unordered_map>
#include <string>
#include <iomanip>
#include <limits>
#include <functional>

/**
 * Job Shop Scheduling Problem Implementation
 * 
 * This file demonstrates various approaches to solving job shop scheduling problems
 * including priority rules, constraint-based scheduling, and local search methods.
 */

// Forward declarations
struct Job;
struct Operation;
struct Machine;
struct Schedule;

// Operation structure representing a single operation in a job
struct Operation {
    int job_id;           // Which job this operation belongs to
    int operation_id;     // Position in job sequence
    int machine_id;       // Required machine
    int processing_time;  // Duration
    int start_time;       // Scheduled start time (-1 if not scheduled)
    int end_time;         // Scheduled end time (-1 if not scheduled)
    
    Operation(int job, int op, int machine, int duration) 
        : job_id(job), operation_id(op), machine_id(machine), 
          processing_time(duration), start_time(-1), end_time(-1) {}
    
    bool is_scheduled() const { return start_time >= 0; }
    
    void schedule_at(int time) {
        start_time = time;
        end_time = time + processing_time;
    }
    
    void unschedule() {
        start_time = -1;
        end_time = -1;
    }
};

// Job structure representing a sequence of operations
struct Job {
    int job_id;
    std::vector<Operation> operations;
    int current_operation;  // Next operation to be scheduled
    
    Job(int id) : job_id(id), current_operation(0) {}
    
    void add_operation(int machine_id, int processing_time) {
        operations.emplace_back(job_id, operations.size(), machine_id, processing_time);
    }
    
    Operation* get_next_operation() {
        if (current_operation < operations.size()) {
            return &operations[current_operation];
        }
        return nullptr;
    }
    
    bool is_complete() const {
        return current_operation >= operations.size();
    }
    
    int completion_time() const {
        if (operations.empty()) return 0;
        return operations.back().end_time;
    }
    
    void advance_operation() {
        current_operation++;
    }
    
    void reset() {
        current_operation = 0;
        for (auto& op : operations) {
            op.unschedule();
        }
    }
};

// Machine structure tracking scheduled operations
struct Machine {
    int machine_id;
    std::vector<Operation*> scheduled_operations;
    
    Machine(int id) : machine_id(id) {}
    
    int earliest_available_time() const {
        if (scheduled_operations.empty()) return 0;
        
        int latest_end = 0;
        for (const auto* op : scheduled_operations) {
            latest_end = std::max(latest_end, op->end_time);
        }
        return latest_end;
    }
    
    bool can_schedule_at(int start_time, int duration) const {
        int end_time = start_time + duration;
        
        for (const auto* op : scheduled_operations) {
            // Check for overlap
            if (!(end_time <= op->start_time || start_time >= op->end_time)) {
                return false;
            }
        }
        return true;
    }
    
    void schedule_operation(Operation* op, int start_time) {
        op->schedule_at(start_time);
        scheduled_operations.push_back(op);
        
        // Keep operations sorted by start time
        std::sort(scheduled_operations.begin(), scheduled_operations.end(),
                 [](const Operation* a, const Operation* b) {
                     return a->start_time < b->start_time;
                 });
    }
    
    void remove_operation(Operation* op) {
        auto it = std::find(scheduled_operations.begin(), scheduled_operations.end(), op);
        if (it != scheduled_operations.end()) {
            scheduled_operations.erase(it);
            op->unschedule();
        }
    }
    
    void clear() {
        for (auto* op : scheduled_operations) {
            op->unschedule();
        }
        scheduled_operations.clear();
    }
    
    void print_schedule() const {
        std::cout << "Machine " << machine_id << ": ";
        if (scheduled_operations.empty()) {
            std::cout << "idle";
        } else {
            for (size_t i = 0; i < scheduled_operations.size(); ++i) {
                const auto* op = scheduled_operations[i];
                std::cout << "J" << op->job_id << "O" << op->operation_id 
                         << "[" << op->start_time << "-" << op->end_time << "]";
                if (i < scheduled_operations.size() - 1) std::cout << " -> ";
            }
        }
        std::cout << std::endl;
    }
};

// Complete schedule representation
struct Schedule {
    std::vector<Job> jobs;
    std::vector<Machine> machines;
    int makespan;
    
    Schedule(int num_jobs, int num_machines) : makespan(0) {
        for (int i = 0; i < num_jobs; ++i) {
            jobs.emplace_back(i);
        }
        for (int i = 0; i < num_machines; ++i) {
            machines.emplace_back(i);
        }
    }
    
    void calculate_makespan() {
        makespan = 0;
        for (const auto& job : jobs) {
            makespan = std::max(makespan, job.completion_time());
        }
    }
    
    bool is_valid() const {
        // Check that all operations are scheduled
        for (const auto& job : jobs) {
            for (const auto& op : job.operations) {
                if (!op.is_scheduled()) return false;
            }
        }
        
        // Check precedence constraints
        for (const auto& job : jobs) {
            for (size_t i = 1; i < job.operations.size(); ++i) {
                if (job.operations[i-1].end_time > job.operations[i].start_time) {
                    return false;
                }
            }
        }
        
        // Check machine conflicts
        for (const auto& machine : machines) {
            for (size_t i = 1; i < machine.scheduled_operations.size(); ++i) {
                const auto* prev = machine.scheduled_operations[i-1];
                const auto* curr = machine.scheduled_operations[i];
                if (prev->end_time > curr->start_time) {
                    return false;
                }
            }
        }
        
        return true;
    }
    
    void reset() {
        for (auto& job : jobs) {
            job.reset();
        }
        for (auto& machine : machines) {
            machine.clear();
        }
        makespan = 0;
    }
    
    void print_schedule() const {
        std::cout << "Job Shop Schedule (Makespan: " << makespan << ")" << std::endl;
        std::cout << "===========================================" << std::endl;
        
        for (const auto& machine : machines) {
            machine.print_schedule();
        }
        
        std::cout << "\nJob completion times:" << std::endl;
        for (const auto& job : jobs) {
            std::cout << "Job " << job.job_id << ": " << job.completion_time() << std::endl;
        }
        std::cout << std::endl;
    }
};

// Priority-based scheduling algorithms
class PriorityScheduler {
public:
    enum Priority {
        SHORTEST_PROCESSING_TIME,
        LONGEST_PROCESSING_TIME,
        EARLIEST_DUE_DATE,
        FIRST_COME_FIRST_SERVED
    };
    
    static Schedule solve(Schedule schedule, Priority priority) {
        std::cout << "Running Priority-Based Scheduler" << std::endl;
        std::cout << "================================" << std::endl;
        
        schedule.reset();
        
        // Create priority queue of available operations
        auto compare = get_comparator(priority);
        std::priority_queue<Operation*, std::vector<Operation*>, decltype(compare)> ready_queue(compare);
        
        // Initialize with first operation of each job
        for (auto& job : schedule.jobs) {
            if (auto* op = job.get_next_operation()) {
                ready_queue.push(op);
            }
        }
        
        int current_time = 0;
        while (!ready_queue.empty()) {
            auto* op = ready_queue.top();
            ready_queue.pop();
            
            Machine& machine = schedule.machines[op->machine_id];
            
            // Find earliest time this operation can start
            int earliest_start = std::max(current_time, machine.earliest_available_time());
            
            // Check precedence constraint
            if (op->operation_id > 0) {
                const auto& prev_op = schedule.jobs[op->job_id].operations[op->operation_id - 1];
                earliest_start = std::max(earliest_start, prev_op.end_time);
            }
            
            // Schedule the operation
            machine.schedule_operation(op, earliest_start);
            
            // Advance job and add next operation to queue
            schedule.jobs[op->job_id].advance_operation();
            if (auto* next_op = schedule.jobs[op->job_id].get_next_operation()) {
                ready_queue.push(next_op);
            }
        }
        
        schedule.calculate_makespan();
        return schedule;
    }
    
private:
    using Comparator = std::function<bool(Operation*, Operation*)>;
    
    static Comparator get_comparator(Priority priority) {
        switch (priority) {
            case SHORTEST_PROCESSING_TIME:
                return [](Operation* a, Operation* b) {
                    return a->processing_time > b->processing_time; // Min heap
                };
            case LONGEST_PROCESSING_TIME:
                return [](Operation* a, Operation* b) {
                    return a->processing_time < b->processing_time; // Max heap
                };
            case FIRST_COME_FIRST_SERVED:
            default:
                return [](Operation* a, Operation* b) {
                    if (a->job_id != b->job_id) return a->job_id > b->job_id;
                    return a->operation_id > b->operation_id;
                };
        }
    }
};

// Simple local search improvement
class LocalSearchScheduler {
public:
    static Schedule improve_by_swapping(Schedule schedule) {
        std::cout << "Running Local Search (Operation Swapping)" << std::endl;
        std::cout << "=========================================" << std::endl;
        
        int initial_makespan = schedule.makespan;
        bool improved = true;
        int iterations = 0;
        
        while (improved && iterations < 100) {
            improved = false;
            iterations++;
            
            // Try swapping adjacent operations on each machine
            for (auto& machine : schedule.machines) {
                if (machine.scheduled_operations.size() < 2) continue;
                
                for (size_t i = 0; i < machine.scheduled_operations.size() - 1; ++i) {
                    Operation* op1 = machine.scheduled_operations[i];
                    Operation* op2 = machine.scheduled_operations[i + 1];
                    
                    // Skip if operations are from same job (precedence constraint)
                    if (op1->job_id == op2->job_id) continue;
                    
                    // Try swapping
                    if (try_swap_operations(schedule, op1, op2)) {
                        schedule.calculate_makespan();
                        if (schedule.makespan < initial_makespan) {
                            improved = true;
                            initial_makespan = schedule.makespan;
                            break;
                        } else {
                            // Revert swap
                            try_swap_operations(schedule, op2, op1);
                        }
                    }
                }
                if (improved) break;
            }
        }
        
        std::cout << "Local search completed after " << iterations << " iterations" << std::endl;
        std::cout << "Makespan improvement: " << initial_makespan << " -> " << schedule.makespan << std::endl;
        
        return schedule;
    }
    
private:
    static bool try_swap_operations(Schedule& schedule, Operation* op1, Operation* op2) {
        // Check if swap is feasible (precedence constraints)
        Job& job1 = schedule.jobs[op1->job_id];
        Job& job2 = schedule.jobs[op2->job_id];
        
        // Simple swap of start times
        int temp_start = op1->start_time;
        op1->start_time = op2->start_time;
        op2->start_time = temp_start;
        
        op1->end_time = op1->start_time + op1->processing_time;
        op2->end_time = op2->start_time + op2->processing_time;
        
        // Check if schedule is still valid
        return schedule.is_valid();
    }
};

// Example problem instances
Schedule create_simple_example() {
    Schedule schedule(3, 3);
    
    // Job 0: M0(3) -> M1(2) -> M2(2)
    schedule.jobs[0].add_operation(0, 3);
    schedule.jobs[0].add_operation(1, 2);
    schedule.jobs[0].add_operation(2, 2);
    
    // Job 1: M1(2) -> M2(1) -> M0(4)
    schedule.jobs[1].add_operation(1, 2);
    schedule.jobs[1].add_operation(2, 1);
    schedule.jobs[1].add_operation(0, 4);
    
    // Job 2: M2(4) -> M0(3)
    schedule.jobs[2].add_operation(2, 4);
    schedule.jobs[2].add_operation(0, 3);
    
    return schedule;
}

Schedule create_larger_example() {
    Schedule schedule(4, 3);
    
    // Job 0: M0(2) -> M1(3) -> M2(1)
    schedule.jobs[0].add_operation(0, 2);
    schedule.jobs[0].add_operation(1, 3);
    schedule.jobs[0].add_operation(2, 1);
    
    // Job 1: M1(1) -> M0(4) -> M2(2)
    schedule.jobs[1].add_operation(1, 1);
    schedule.jobs[1].add_operation(0, 4);
    schedule.jobs[1].add_operation(2, 2);
    
    // Job 2: M2(3) -> M1(2) -> M0(1)
    schedule.jobs[2].add_operation(2, 3);
    schedule.jobs[2].add_operation(1, 2);
    schedule.jobs[2].add_operation(0, 1);
    
    // Job 3: M0(1) -> M2(2) -> M1(3)
    schedule.jobs[3].add_operation(0, 1);
    schedule.jobs[3].add_operation(2, 2);
    schedule.jobs[3].add_operation(1, 3);
    
    return schedule;
}

int main() {
    std::cout << "Job Shop Scheduling Problem" << std::endl;
    std::cout << "===========================" << std::endl << std::endl;
    
    // Example 1: Simple 3x3 problem
    std::cout << "=== SIMPLE 3x3 INSTANCE ===" << std::endl;
    auto simple_instance = create_simple_example();
    
    std::cout << "Problem instance:" << std::endl;
    std::cout << "Job 0: M0(3) -> M1(2) -> M2(2)" << std::endl;
    std::cout << "Job 1: M1(2) -> M2(1) -> M0(4)" << std::endl;
    std::cout << "Job 2: M2(4) -> M0(3)" << std::endl << std::endl;
    
    // Try different priority rules
    auto spt_schedule = PriorityScheduler::solve(simple_instance, PriorityScheduler::SHORTEST_PROCESSING_TIME);
    std::cout << "Shortest Processing Time (SPT) Rule:" << std::endl;
    spt_schedule.print_schedule();
    
    auto lpt_schedule = PriorityScheduler::solve(simple_instance, PriorityScheduler::LONGEST_PROCESSING_TIME);
    std::cout << "Longest Processing Time (LPT) Rule:" << std::endl;
    lpt_schedule.print_schedule();
    
    auto fcfs_schedule = PriorityScheduler::solve(simple_instance, PriorityScheduler::FIRST_COME_FIRST_SERVED);
    std::cout << "First Come First Served (FCFS) Rule:" << std::endl;
    fcfs_schedule.print_schedule();
    
    // Apply local search to best solution
    auto best_schedule = (spt_schedule.makespan <= lpt_schedule.makespan && 
                         spt_schedule.makespan <= fcfs_schedule.makespan) ? spt_schedule :
                        (lpt_schedule.makespan <= fcfs_schedule.makespan) ? lpt_schedule : fcfs_schedule;
    
    auto improved_schedule = LocalSearchScheduler::improve_by_swapping(best_schedule);
    improved_schedule.print_schedule();
    
    std::cout << std::endl;
    
    // Example 2: Larger 4x3 problem
    std::cout << "=== LARGER 4x3 INSTANCE ===" << std::endl;
    auto larger_instance = create_larger_example();
    
    std::cout << "Problem instance:" << std::endl;
    std::cout << "Job 0: M0(2) -> M1(3) -> M2(1)" << std::endl;
    std::cout << "Job 1: M1(1) -> M0(4) -> M2(2)" << std::endl;
    std::cout << "Job 2: M2(3) -> M1(2) -> M0(1)" << std::endl;
    std::cout << "Job 3: M0(1) -> M2(2) -> M1(3)" << std::endl << std::endl;
    
    // Compare priority rules
    auto larger_spt = PriorityScheduler::solve(larger_instance, PriorityScheduler::SHORTEST_PROCESSING_TIME);
    auto larger_lpt = PriorityScheduler::solve(larger_instance, PriorityScheduler::LONGEST_PROCESSING_TIME);
    auto larger_fcfs = PriorityScheduler::solve(larger_instance, PriorityScheduler::FIRST_COME_FIRST_SERVED);
    
    std::cout << "Priority Rule Comparison:" << std::endl;
    std::cout << "SPT Makespan: " << larger_spt.makespan << std::endl;
    std::cout << "LPT Makespan: " << larger_lpt.makespan << std::endl;
    std::cout << "FCFS Makespan: " << larger_fcfs.makespan << std::endl << std::endl;
    
    // Show best solution
    auto larger_best = (larger_spt.makespan <= larger_lpt.makespan && 
                       larger_spt.makespan <= larger_fcfs.makespan) ? larger_spt :
                      (larger_lpt.makespan <= larger_fcfs.makespan) ? larger_lpt : larger_fcfs;
    
    std::cout << "Best Priority Rule Solution:" << std::endl;
    larger_best.print_schedule();
    
    // Apply local search
    auto larger_improved = LocalSearchScheduler::improve_by_swapping(larger_best);
    larger_improved.print_schedule();
    
    return 0;
}
