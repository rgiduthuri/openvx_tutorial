# 
#  Permission is hereby granted, free of charge, to any person obtaining a
#  copy of this software and/or associated documentation files (the
#  "Materials"), to deal in the Materials without restriction, including
#  without limitation the rights to use, copy, modify, merge, publish,
#  distribute, sublicense, and/or sell copies of the Materials, and to
#  permit persons to whom the Materials are furnished to do so, subject to
#  the following conditions:
# 
#  The above copyright notice and this permission notice shall be included
#  in all copies or substantial portions of the Materials.
# 
#  THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
#  EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
#  MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
#  IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
#  CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
#  TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
#  MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
#

cmake_minimum_required(VERSION 2.8.9)

find_package(OpenCL REQUIRED)

set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -std=c++11 -DCL_USE_DEPRECATED_OPENCL_1_2_APIS")

# opencl_example
add_executable(opencl_kernel_example opencl_kernel_example.cpp)
target_link_libraries(opencl_kernel_example ${OpenCL_LIBRARIES})

# interop_example
add_executable(opencl_interop_example opencl_interop_example.cpp my_vx_tensor_map_impl.cpp)
target_link_libraries(opencl_interop_example openvx ${OpenCL_LIBRARIES})
