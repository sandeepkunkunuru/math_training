cmake_minimum_required(VERSION 3.14)
project(optimization_math_training)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# Include directories for OR-Tools
include_directories(${CMAKE_CURRENT_SOURCE_DIR}/include)

# Find OR-Tools package if available
find_package(ortools CONFIG QUIET)

# Add subdirectories for each topic
add_subdirectory(cpp_implementations/linear_algebra)
add_subdirectory(cpp_implementations/calculus)
add_subdirectory(cpp_implementations/probability)
add_subdirectory(cpp_implementations/optimization)
add_subdirectory(cpp_implementations/linear_programming)
add_subdirectory(cpp_implementations/integer_programming)
add_subdirectory(cpp_implementations/constraint_programming)
add_subdirectory(cpp_implementations/network_flows)
add_subdirectory(cpp_implementations/routing)
add_subdirectory(cpp_implementations/scheduling)
add_subdirectory(cpp_implementations/metaheuristics)
