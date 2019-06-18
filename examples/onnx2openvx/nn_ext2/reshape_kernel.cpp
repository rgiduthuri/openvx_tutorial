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

#include "nn_ext2.h"
#include "nn_ext2_common.hpp"
#include "reshape_kernel.hpp"

////////
// user kernel for reshape operation
//
vx_status VX_CALLBACK reshape_validator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // get input tensor meta data
    vx_size num_of_idims;
    vx_size idims[8];
    vx_enum data_type;
    vx_uint8 fixed_point_pos;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_idims, sizeof(num_of_idims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &idims, num_of_idims * sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_FIXED_POINT_POSITION, &fixed_point_pos, sizeof(fixed_point_pos)));

    // get output tensor dimensions
    vx_size num_of_odims;
    vx_size odims[8];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_odims, sizeof(num_of_odims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &odims, num_of_odims * sizeof(vx_size)));

    // make sure tensor type is INT16
    if(data_type != VX_TYPE_INT16) {
        return VX_ERROR_INVALID_FORMAT;
    }

    // make sure input and output tensors have same number of elements
    vx_size icount = 1, ocount = 1;
    for(vx_size i = 0; i < num_of_idims; i++) {
        icount *= idims[i];
    }
    for(vx_size i = 0; i < num_of_odims; i++) {
        ocount *= odims[i];
    }
    if(icount != ocount) {
        return VX_ERROR_INVALID_DIMENSION;
    }

    // set required output tensor meta data
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_odims, sizeof(num_of_odims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DIMS, &odims, num_of_odims * sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[1], VX_TENSOR_FIXED_POINT_POSITION, &fixed_point_pos, sizeof(fixed_point_pos)));

    return VX_SUCCESS;
}

vx_status VX_CALLBACK reshape_initializer(vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    // get input tensor meta data
    vx_size num_of_dims;
    vx_size dims[8];
    vx_enum data_type;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &dims, num_of_dims * sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type)));

    // allocate and set local buffer to the node
    vx_size buffer_size = sizeof(vx_int16);
    for(vx_size i = 0; i < num_of_dims; i++) {
        buffer_size *= dims[i];
    }
    vx_uint8 * local_buffer = new vx_uint8[buffer_size];
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_SIZE, &buffer_size, sizeof(buffer_size)));
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &local_buffer, sizeof(local_buffer)));

    return VX_SUCCESS;
}

vx_status VX_CALLBACK reshape_uninitializer(vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    // free node's local buffer
    vx_uint8 * local_buffer = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &local_buffer, sizeof(local_buffer)));
    if(local_buffer) {
        delete[] local_buffer;
    }

    return VX_SUCCESS;
}

vx_status VX_CALLBACK reshape_host_compute(vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    // get node's local buffer
    vx_size buffer_size;
    vx_uint8 * local_buffer;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_SIZE, &buffer_size, sizeof(buffer_size)));
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &local_buffer, sizeof(local_buffer)));

    // get dimensions of input and output tensors and calculate strides
    vx_tensor input = (vx_tensor)parameters[0];
    vx_tensor output = (vx_tensor)parameters[1];
    vx_size num_of_idims, num_of_odims;
    vx_size idims[8], istride[8];
    vx_size odims[8], ostride[8];
    ERROR_CHECK_STATUS(vxQueryTensor(input, VX_TENSOR_NUMBER_OF_DIMS, &num_of_idims, sizeof(num_of_idims)));
    ERROR_CHECK_STATUS(vxQueryTensor(input, VX_TENSOR_DIMS, &idims, num_of_idims * sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor(output, VX_TENSOR_NUMBER_OF_DIMS, &num_of_odims, sizeof(num_of_odims)));
    ERROR_CHECK_STATUS(vxQueryTensor(output, VX_TENSOR_DIMS, &odims, num_of_odims * sizeof(vx_size)));
    vx_size ibuffer_size = sizeof(vx_int16);
    for(vx_size i = 0; i < num_of_idims; i++) {
        istride[i] = ibuffer_size;
        ibuffer_size *= idims[i];
    }
    vx_size obuffer_size = sizeof(vx_int16);
    for(vx_size i = 0; i < num_of_odims; i++) {
        ostride[i] = obuffer_size;
        obuffer_size *= odims[i];
    }

    // perform reshape by reading input into local buffer and writing local buffer into output
    vx_size zeros[8] = { 0 };
    ERROR_CHECK_STATUS(vxCopyTensorPatch(input, num_of_idims, zeros, idims,
            istride, local_buffer, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    ERROR_CHECK_STATUS(vxCopyTensorPatch(output, num_of_odims, zeros, odims,
            ostride, local_buffer, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

    return VX_SUCCESS;
}

////////
//! \brief register reshape_kernel
vx_status registerReshapeKernel(vx_context context)
{
    // register a custom kernel for reshape
    vx_kernel reshape_kernel = vxAddUserKernel(context,
            "openvx_tutorial.nn_ext2.reshape",
            VX_KERNEL_RESHAPE_LAYER,
            reshape_host_compute,
            2,
            reshape_validator,
            reshape_initializer,
            reshape_uninitializer);
    ERROR_CHECK_OBJECT(reshape_kernel);
    ERROR_CHECK_STATUS(vxAddParameterToKernel(reshape_kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(reshape_kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxFinalizeKernel(reshape_kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&reshape_kernel));

    return VX_SUCCESS;
}

////////
//! \brief reshape kernel node API
VX_API_EXPORT vx_node VX_API_CALL vxReshapeLayer(vx_graph graph, vx_tensor input, vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[] = {
            (vx_reference)input,
            (vx_reference)output,
        };
        node = createNodeFromKernelEnum(graph, VX_KERNEL_RESHAPE_LAYER, params, sizeof(params) / sizeof(params[0]));
    }
    return node;
}
