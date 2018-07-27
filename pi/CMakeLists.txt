
cmake_minimum_required(VERSION 3.8)

project(pi
  VERSION 1.0.0
  LANGUAGES CUDA CXX)

add_executable(pi pi.cu)
target_link_libraries(pi)
