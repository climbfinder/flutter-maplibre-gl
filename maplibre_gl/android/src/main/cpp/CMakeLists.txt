cmake_minimum_required(VERSION 3.18.1)
project(climbwall CXX)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

add_library(climbwall SHARED climb_wall_layer.cpp)
target_include_directories(climbwall PRIVATE include)
target_link_libraries(climbwall GLESv3 log)
