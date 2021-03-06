cmake_minimum_required(VERSION 2.8 FATAL_ERROR)
project(OverFeat)

set(PROJECTS_DIR ${PROJECT_SOURCE_DIR})

set(SOURCE_FILES
		src/tools/ppm.cpp
		bin/linux_64/overfeat
		src/overfeat.cpp
		src/overfeatcmd.cpp
		)

set(INCLUDE_DIRS
		src/
		src/tools/
		data/default/
		bin/linux_64/
		bin/linux_32/
		)
include(${PROJECTS_DIR}/src/TH/CMakeLists.txt)

###############################################################################
# Torch
set(TORCH_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/TH)
find_library(TORCH_LIBRARIES
		NAMES th TH
		PATHS ${INCLUDE_DIRS})
list(APPEND LIBRARIES ${TORCH_LIBRARIES})
list(APPEND LIBRARY_DIRS ${TORCH_LIBRARY_DIR})
message("Torch libs: ${TORCH_LIBRARIES}")

###############################################################################
# OpenMP
find_package(OpenMP REQUIRED)
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} ${OpenMP_CXX_FLAGS}")
set(CMAKE_C_FLAGS "${CMAKE_C_FLAGS} ${OpenMP_C_FLAGS}")

###############################################################################
include_directories(${INCLUDE_DIRS})
link_directories(${LIBRARY_DIRS})

add_library(OverFeat ${SOURCE_FILES})
target_link_libraries(OverFeat ${LIBRARIES})

###############################################################################
# core executable
add_executable(overfeatcmd ${SOURCE_FILES} src/overfeatcmd.cpp)
target_link_libraries(overfeatcmd ${LIBRARIES})
