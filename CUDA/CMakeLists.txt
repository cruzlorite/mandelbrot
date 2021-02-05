# Minimun cmake version
cmake_minimum_required(VERSION 3.5)

# c++ standard
set(CMAKE_CXX_STANDARD 14)

# Executable and project names
set(TARGET mandelbrot_cuda)
project(mandelbrot_cuda)

## Find cuda
find_package(CUDA QUIET REQUIRED)
    
# Project source files 
set(SOURCES mandelbrot.cu util/lodepng.cpp)

# Output binary on /bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

cuda_add_executable(${TARGET} ${SOURCES})