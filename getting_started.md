# Getting Started with Optimization Mathematics and OR-Tools

This guide will help you set up your environment and start working with the examples in this repository.

## Setting Up Your Environment

1. **Install Python Dependencies**

   First, install the required Python packages:

   ```bash
   pip install -r requirements.txt
   ```

   This will install OR-Tools, NumPy, and Matplotlib.

2. **Verify Installation**

   Run a simple test to verify that OR-Tools is installed correctly:

   ```bash
   python -c "from ortools.linear_solver import pywraplp; print('OR-Tools version:', pywraplp.Solver.SolverVersion())"
   ```

## Working Through the Examples

The examples in this repository are organized by topic, with increasing complexity:

1. **Linear Programming**
   - Start with `linear_programming/01_introduction_to_lp.md` to understand the theory
   - Run `linear_programming/02_simple_lp_example.py` to see a basic example

2. **Constraint Programming**
   - Read `constraint_programming/01_introduction_to_cp.md` for the concepts
   - Run `constraint_programming/02_sudoku_solver.py` to see constraint programming in action

3. **Integer Programming**
   - Study `integer_programming/01_introduction_to_ip.md` for the theory
   - Run `integer_programming/02_knapsack_problem.py` to understand integer variables

4. **Vehicle Routing**
   - Read `routing/01_introduction_to_vrp.md` for an overview
   - Run `routing/02_capacitated_vrp.py` to see a more complex application

## Learning Path

Follow this sequence for a structured learning experience:

1. **Week 1-2**: Focus on linear programming fundamentals
   - Read the theory in `linear_programming/01_introduction_to_lp.md`
   - Experiment with `linear_programming/02_simple_lp_example.py`
   - Try modifying the constraints and objective function

2. **Week 3-4**: Move to constraint programming
   - Study `constraint_programming/01_introduction_to_cp.md`
   - Run and understand `constraint_programming/02_sudoku_solver.py`
   - Try creating your own constraint satisfaction problem

3. **Week 5-6**: Explore integer programming
   - Learn from `integer_programming/01_introduction_to_ip.md`
   - Analyze `integer_programming/02_knapsack_problem.py`
   - Experiment with different problem instances

4. **Week 7-8**: Dive into vehicle routing
   - Study `routing/01_introduction_to_vrp.md`
   - Run `routing/02_capacitated_vrp.py`
   - Try adding time windows or multiple depots

## Recommended Learning Resources

1. **Books**:
   - "Introduction to Linear Optimization" by Bertsimas and Tsitsiklis
   - "Constraint Processing" by Rina Dechter
   - "Integer Programming" by Laurence Wolsey

2. **Online Resources**:
   - [OR-Tools Documentation](https://developers.google.com/optimization)
   - [Operations Research Stack Exchange](https://or.stackexchange.com/)
   - [INFORMS Tutorials](https://www.informs.org/Resource-Center/INFORMS-Tutorials)

3. **Courses**:
   - Coursera: "Discrete Optimization" by Pascal Van Hentenryck
   - edX: "Optimization Methods for Business Analytics" by MIT

## Next Steps

After working through these examples, you can:

1. Create your own optimization models for real-world problems
2. Explore the OR-Tools codebase to understand its implementation
3. Contribute to OR-Tools by fixing bugs or adding features

Remember to refer to the `optimization_learning_plan.md` for a comprehensive learning roadmap.
