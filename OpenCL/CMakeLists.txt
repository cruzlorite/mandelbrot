cmake_minimum_required(VERSION 3.5)

# c++ standard
set(CMAKE_CXX_STANDARD 14)

# Executable and project names
set(TARGET mandelbrot_opencl)
project(mandelbrot_opencl)

# Source files
set(SOURCES mandelbrot.cpp util/lodepng.cpp)

# Output binary on /bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Add an executable with the above sources
add_executable(${TARGET} ${SOURCES})

# Link libraries
target_link_libraries(${TARGET} OpenCL)
