# Copyright (c) 2016 The Khronos Group Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and/or associated documentation files (the
# "Materials"), to deal in the Materials without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Materials, and to
# permit persons to whom the Materials are furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Materials.
#
# THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

cmake_minimum_required  ( VERSION 2.8                                             )
get_filename_component  ( project ${CMAKE_CURRENT_LIST_DIR} NAME                  )
project                 ( ${project}                                              )
find_package            ( OpenCV REQUIRED                                         )
include_directories     ( ${OpenCV_INCLUDE_DIRS}                                  )
include_directories     ( ${CMAKE_SOURCE_DIR}/include                             )
aux_source_directory    ( . SRC_LIST                                              )
link_directories        ( ${OpenVX_LIBPATH}                                       )
add_executable          ( ${PROJECT_NAME} ${SRC_LIST}                             )
target_link_libraries   ( ${PROJECT_NAME} ${OpenVX_LIBRARIES} ${OpenCV_LIBRARIES} )
