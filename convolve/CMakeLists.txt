
cmake_minimum_required(VERSION 3.8)

project(convolve
  VERSION 1.0.0
  LANGUAGES CUDA CXX)

add_executable(convolve convolve.cu)
target_link_libraries(convolve)
