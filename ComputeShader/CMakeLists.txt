# Set the minimum version of CMake that can be used
cmake_minimum_required(VERSION 3.5)

# c++ standard
set(CMAKE_CXX_STANDARD 14)

# Executable and project names
project(mandelbrot_ComputeShader)

# Include external headers
include_directories(external)

# Build glad
add_subdirectory(external/glad)

# Source files
set(SOURCES "mandelbrot.cpp" "util/lodepng.cpp")

# Set link libraries depending on OS
if(WIN32)
  set(BUILD_OS    WIN32)
  set(GL_NAMES    opengl32)
  set(GLFW_NAMES  glfw3)

elseif(UNIX AND APPLE)
  message(FATAL_ERROR "Build for Apple not supported")

elseif(UNIX AND NOT APPLE)
  set(BUILD_OS    GNU-Linux)
  set(GL_NAMES    GL)
  set(GLFW_NAMES  glfw)
endif()

########### FIND LIBRARIES #############

  # Where to find GLFW
  set(GLFW_PATH lib/${BUILD_OS}/GLFW)

  # Find GL and GLFW
  find_library(GL   NAMES ${GL_NAMES})
  find_library(GLFW NAMES ${GLFW_NAMES} PATHS ${GLFW_PATH})

########################################

# Output binary on /bin
set(CMAKE_RUNTIME_OUTPUT_DIRECTORY ${CMAKE_SOURCE_DIR}/bin)

# Add an executable with the above sources
add_executable(${PROJECT_NAME} ${SOURCES})

# Link libraries
target_link_libraries(${PROJECT_NAME} ${GL} ${GLFW} glad)