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

cmake_minimum_required  ( VERSION 2.8                 )
project                 ( easy_exercise1              )
aux_source_directory    ( . SRC_LIST                  )

option(USE_KHRONOS_SAMPLE "Use Khronos Sample Implementation" ON)

if( USE_KHRONOS_SAMPLE )
  set                   ( VX_PATH /home/openvx/openvx_sample/install/Linux/x64 )
  if( CMAKE_BUILD_TYPE MATCHES ".*Deb.*" )
    link_directories    ( ${VX_PATH}/Debug/bin   )
  else( CMAKE_BUILD_TYPE MATCHES ".*Deb.*" )
    link_directories    ( ${VX_PATH}/Release/bin )
  endif( CMAKE_BUILD_TYPE MATCHES ".*Deb.*" )
  include_directories   ( ${VX_PATH}/Release/include  )
  add_executable          ( ${PROJECT_NAME} ${SRC_LIST} )
  target_link_libraries ( ${PROJECT_NAME} openvx vxu opencv_imgproc opencv_highgui opencv_core )
else( USE_KHRONOS_SAMPLE )
  include_directories   ( ${CMAKE_SOURCE_DIR}/openvx/include  )
  add_executable          ( ${PROJECT_NAME} ${SRC_LIST} )
  target_link_libraries ( ${PROJECT_NAME} openvx opencv_imgproc opencv_highgui opencv_core )
endif( USE_KHRONOS_SAMPLE )
