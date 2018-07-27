
cmake_minimum_required(VERSION 3.8)

project(texture
  VERSION 1.0.0
  LANGUAGES CUDA CXX)

add_executable(texture texture.cu)
target_link_libraries(texture)
