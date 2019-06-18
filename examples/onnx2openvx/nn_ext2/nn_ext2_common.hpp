/*
* Copyright (c) 2019 <copyright holders>
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#ifndef nn_ext2_common_hpp__
#define nn_ext2_common_hpp__

#include <VX/vx.h>
#include <iostream>

////////
//! library public APIs for export
#define VX_API_EXPORT extern "C" __attribute__ ((visibility ("default")))

////////
// Useful macros for OpenVX error checking:
//   ERROR_CHECK_STATUS     - check whether the status is VX_SUCCESS
//   ERROR_CHECK_OBJECT     - check whether the object creation is successful
#define ERROR_CHECK_STATUS(status) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            std::cout << "ERROR: failed with status = (" << status_ << ") at " __FILE__ "#" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }
#define ERROR_CHECK_OBJECT(obj) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            std::cout << "ERROR: failed with status = (" << status_ << ") at " __FILE__ "#" << __LINE__ << std::endl; \
            exit(1); \
        } \
    }

////////////////////////////////////////////////////////////////////////////
//! \brief Utility function to create a node from kernel enum
vx_node createNodeFromKernelEnum(
    vx_graph graph,
    vx_enum kernelEnum,
    vx_reference params[],
    vx_uint32 num);

#endif // nn_ext2_common_hpp__
