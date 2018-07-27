
cmake_minimum_required(VERSION 3.8)

project(median2
  VERSION 1.0.0
  LANGUAGES CUDA CXX)

add_executable(median2 median2.cu)
target_link_libraries(median2)
