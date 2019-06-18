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

#ifndef nn_ext2_h__
#define nn_ext2_h__

#include <VX/vx.h>

//////////////////////////////////////////////////////////////////////
//! OpenVX kernels in this module
enum nn_ext2_library {
    NN_EXT2_LIBRARY = 0x80,
};
enum nn_ext2_kernel_e {
    VX_KERNEL_CONCAT_LAYER  = VX_KERNEL_BASE(VX_ID_KHRONOS, NN_EXT2_LIBRARY) + 0x001,
    VX_KERNEL_RESHAPE_LAYER = VX_KERNEL_BASE(VX_ID_KHRONOS, NN_EXT2_LIBRARY) + 0x002,
};

extern "C" {

//////////////////////////////////////////////////////////////////////
//! \brief Concat kernel
VX_API_ENTRY vx_node VX_API_CALL vxConcatLayer(
    vx_graph graph,
    vx_tensor input[],
    vx_size num_inputs,
    vx_size axis,
    vx_tensor output);

//////////////////////////////////////////////////////////////////////
//! \brief Reshape kernel
VX_API_ENTRY vx_node VX_API_CALL vxReshapeLayer(
    vx_graph graph,
    vx_tensor input,
    vx_tensor output);

//////////////////////////////////////////////////////////////////////
//! \brief Load kernels of this module into a context
VX_API_ENTRY vx_status VX_API_CALL vxLoadKernelsNnExt2(vx_context context);

};

#endif // nn_ext2_h__
