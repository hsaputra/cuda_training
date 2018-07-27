
cmake_minimum_required(VERSION 3.8)

project(cuda_intro
  VERSION 1.0.0
  LANGUAGES CUDA CXX)

add_executable(helloworld helloworld.cu)
target_link_libraries(helloworld)

add_executable(increment increment.cu)
target_link_libraries(increment)

add_executable(vector_addition vector_addition.cu)
target_link_libraries(vector_addition)
