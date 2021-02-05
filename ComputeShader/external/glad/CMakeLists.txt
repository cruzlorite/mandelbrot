cmake_minimum_required(VERSION 3.5)
add_library(
        glad STATIC
        glad.c
        )

if(UNIX AND APPLE)
  message(FATAL_ERROR "Build for Apple not supported")
elseif(UNIX AND NOT APPLE)
  # On linux we need to link dl
  target_link_libraries(glad dl)
endif()
