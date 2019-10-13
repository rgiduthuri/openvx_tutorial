#
# Global setup file for OpenVX CMake

#
cmake_minimum_required(VERSION 2.8.9)

project(openvx-book-samples)
find_package(OpenCV REQUIRED)
if(${OpenCV_VERSION} VERSION_GREATER 4 OR ${OpenCV_VERSION} VERSION_EQUAL 4)
  message("OpenCV version 4.0 or higher detected")
  SET(CMAKE_CXX_FLAGS  "${CMAKE_CXX_FLAGS} -std=c++11")
  message("${OpenCV_LIBS}")
endif()

find_package(LAPACK)
if(${LAPACK_FOUND})
  message("LAPACK found:")
  message(${LAPACK_LIBRARIES})
else()
  message("LAPACK not found")
endif()

find_library(OPENVX openvx)
message("OpenVX found: ${OPENVX}")
find_library(VXU vxu)
find_package(VXA QUIET)
if(NOT ${VXA_FOUND})
  message(FATAL_ERROR "The required VXA library is not found, please download it from https://github.com/relrotciv/vxa.")
endif()


include_directories($ENV{C_INCLUDE_PATH})
include_directories(ppm-io)
link_directories(/usr/local/lib)

add_subdirectory(example1)
add_subdirectory(example2)
add_subdirectory(example3)
add_subdirectory(example4)
add_subdirectory(stitch)
add_subdirectory(hough)
add_subdirectory(birds-eye)
add_subdirectory(undistort)
add_subdirectory(tracking)
