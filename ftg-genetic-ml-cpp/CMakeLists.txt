cmake_minimum_required(VERSION 3.8)
project(ftg_genetic_ml)

# Default to C++14 or C++17
if(NOT CMAKE_CXX_STANDARD)
    set(CMAKE_CXX_STANDARD 17)
endif()

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
    add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# --- Include Directories ---
include_directories(include)

# --- Executable ---
add_executable(ftg_genetic_ml
        include/Parameters.h
        src/Parameters.cpp
        include/FollowTheGap.h
        src/FollowTheGap.cpp
        src/main.cpp
        include/GeneticML.h
        src/GeneticML.cpp
)