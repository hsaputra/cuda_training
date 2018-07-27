
cmake_minimum_required(VERSION 3.8)

project(cuda_training
  VERSION 1.0.0
  LANGUAGES CUDA CXX)

find_package(CUDA)
cuda_select_nvcc_arch_flags(cuda_arch "Auto")
string(REPLACE ";" " " cuda_arch "${cuda_arch}")
set(CMAKE_CUDA_FLAGS "${cuda_arch}")

if(APPLE)
	set(CMAKE_BUILD_RPATH ${CMAKE_CUDA_IMPLICIT_LINK_DIRECTORIES})
endif()

set(CMAKE_CUDA_STANDARD 11)

add_subdirectory(boxfilter)
add_subdirectory(convolve)
add_subdirectory(cudaDeviceProperties)
add_subdirectory(hist)
add_subdirectory(intro)
add_subdirectory(median2)
add_subdirectory(pi)
add_subdirectory(reduction)
add_subdirectory(textures)
add_subdirectory(transpose)
