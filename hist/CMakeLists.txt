
cmake_minimum_required(VERSION 3.8)

project(hist
  VERSION 1.0.0
  LANGUAGES CUDA CXX)

add_executable(hist hist.cu)
target_link_libraries(hist)
