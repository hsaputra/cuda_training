
cmake_minimum_required(VERSION 3.8)

project(cuda_training
  VERSION 1.0.0
  LANGUAGES CUDA CXX)

add_executable(boxfilter boxfilter.cu)
target_link_libraries(boxfilter)
