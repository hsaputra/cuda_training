
cmake_minimum_required(VERSION 3.8)

project(cudaDeviceProperties
  VERSION 1.0.0
  LANGUAGES CUDA CXX)

add_executable(cudaDeviceInfo cudaDeviceInfo.cu)
target_link_libraries(cudaDeviceInfo)
