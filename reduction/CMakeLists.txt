cmake_minimum_required(VERSION 3.8)

project(reduction
  VERSION 1.0.0
  LANGUAGES CUDA CXX)

add_executable(reduction reduction.cu)
target_link_libraries(reduction)
