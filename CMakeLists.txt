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

###############################################################################
# Torch
set(TORCH_LIBRARY_DIR ${CMAKE_CURRENT_SOURCE_DIR}/src/TH)
find_library(TORCH_LIBRARY
		NAMES th TH
		PATHS ${INCLUDE_DIRS})
list(APPEND LIBRARIES ${TORCH_LIBRARY})
list(APPEND LIBRARY_DIRS ${TORCH_LIBRARY_DIR})

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
# test executable
add_executable(OverFeatTest ${SOURCE_FILES} bin/linux_64/overfeat)
target_link_libraries(OverFeatTest ${LIBRARIES})
