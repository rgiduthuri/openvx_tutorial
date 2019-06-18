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
#include "concat_kernel.hpp"

////////
//! \brief maximum number of concat inputs
#define MAX_CONCAT_LAYER_INPUTS   8

////////
// user kernel for concat operation
//
vx_status VX_CALLBACK concat_validator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // get first input tensor meta data
    vx_size num_of_dims;
    vx_size dims[8];
    vx_enum data_type;
    vx_uint8 fixed_point_pos;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DIMS, &dims, num_of_dims * sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[0], VX_TENSOR_FIXED_POINT_POSITION, &fixed_point_pos, sizeof(fixed_point_pos)));

    // make sure tensor type is INT16
    if(data_type != VX_TYPE_INT16) {
        return VX_ERROR_INVALID_FORMAT;
    }

    // get & check axis
    vx_size axis = 0;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[MAX_CONCAT_LAYER_INPUTS],
            &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
    if(axis >= num_of_dims) {
        return VX_ERROR_INVALID_DIMENSION;
    }

    // get output dimension
    for(size_t i = 1; i < MAX_CONCAT_LAYER_INPUTS; i++) {
        if(parameters[i]) {
            vx_size i_num_of_dims;
            vx_size i_dims[8];
            vx_enum i_data_type;
            vx_uint8 i_fixed_point_pos;
            ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_NUMBER_OF_DIMS, &i_num_of_dims, sizeof(i_num_of_dims)));
            ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_DIMS, &i_dims, i_num_of_dims * sizeof(vx_size)));
            ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_DATA_TYPE, &i_data_type, sizeof(i_data_type)));
            ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_FIXED_POINT_POSITION, &i_fixed_point_pos, sizeof(i_fixed_point_pos)));
            if(i_data_type != data_type || i_fixed_point_pos != fixed_point_pos) {
                return VX_ERROR_INVALID_VALUE;
            }
            else if(i_num_of_dims != num_of_dims) {
                return VX_ERROR_INVALID_DIMENSION;
            }
            else {
                for(size_t j = 0; j < num_of_dims; j++) {
                    if(j != axis && i_dims[j] != dims[j]) {
                        return VX_ERROR_INVALID_DIMENSION;
                    }
                }
            }
            dims[axis] += i_dims[axis];
        }
    }

    // set required output tensor meta data
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[MAX_CONCAT_LAYER_INPUTS+1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[MAX_CONCAT_LAYER_INPUTS+1], VX_TENSOR_DIMS, &dims, num_of_dims * sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[MAX_CONCAT_LAYER_INPUTS+1], VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(metas[MAX_CONCAT_LAYER_INPUTS+1], VX_TENSOR_FIXED_POINT_POSITION, &fixed_point_pos, sizeof(fixed_point_pos)));

    return VX_SUCCESS;
}

vx_status VX_CALLBACK concat_initializer(vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    // get axis
    vx_size axis = 0;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[MAX_CONCAT_LAYER_INPUTS],
            &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    // get output tensor dimensions
    vx_size num_of_dims;
    vx_size dims[8];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[MAX_CONCAT_LAYER_INPUTS+1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[MAX_CONCAT_LAYER_INPUTS+1], VX_TENSOR_DIMS, &dims, num_of_dims * sizeof(vx_size)));

    // get max axis_dimension
    dims[axis] = 0;
    for(size_t i = 0; i < MAX_CONCAT_LAYER_INPUTS; i++) {
        if(parameters[i]) {
            vx_size i_dims[8];
            ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_DIMS, &i_dims, num_of_dims * sizeof(vx_size)));
            if(i_dims[axis] > dims[axis]) {
                dims[axis] = i_dims[axis];
            }
        }
    }

    // allocate and set local buffer to the node (enough to hold max input tensor size)
    vx_size buffer_size = sizeof(vx_int16);
    for(vx_size i = 0; i < num_of_dims; i++) {
        buffer_size *= dims[i];
    }
    vx_uint8 * local_buffer = new vx_uint8[buffer_size];
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_SIZE, &buffer_size, sizeof(buffer_size)));
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &local_buffer, sizeof(local_buffer)));

    return VX_SUCCESS;
}

vx_status VX_CALLBACK concat_uninitializer(vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    // free node's local buffer
    vx_uint8 * local_buffer = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &local_buffer, sizeof(local_buffer)));
    if(local_buffer) {
        delete[] local_buffer;
    }

    return VX_SUCCESS;
}

vx_status VX_CALLBACK concat_host_compute(vx_node node, const vx_reference parameters[], vx_uint32 num)
{
    // get node's local buffer
    vx_size buffer_size;
    vx_uint8 * local_buffer;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_SIZE, &buffer_size, sizeof(buffer_size)));
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &local_buffer, sizeof(local_buffer)));

    // get axis
    vx_size axis = 0;
    ERROR_CHECK_STATUS(vxCopyScalar((vx_scalar)parameters[MAX_CONCAT_LAYER_INPUTS],
            &axis, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

    // get output tensor dimensions
    vx_size num_of_dims;
    vx_size dims[8];
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[MAX_CONCAT_LAYER_INPUTS+1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[MAX_CONCAT_LAYER_INPUTS+1], VX_TENSOR_DIMS, &dims, num_of_dims * sizeof(vx_size)));
    vx_size begin_dims[8] = { 0 };
    vx_size end_dims[8] = { 0 };
    vx_size stride[8] = { 0 };
    buffer_size = sizeof(vx_int16);
    for(vx_size i = 0; i < num_of_dims; i++) {
        end_dims[i] = dims[i];
        stride[i] = buffer_size;
        buffer_size *= dims[i];
    }

    // get max axis_dimension
    end_dims[axis] = 0;
    for(size_t i = 0; i < MAX_CONCAT_LAYER_INPUTS; i++) {
        if(parameters[i]) {
            vx_size i_dims[8];
            ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[i], VX_TENSOR_DIMS, &i_dims, num_of_dims * sizeof(vx_size)));
            vx_size i_stride[8] = { 0 }, i_buffer_size = sizeof(vx_int16);
            for(vx_size i = 0; i < num_of_dims; i++) {
                i_stride[i] = i_buffer_size;
                i_buffer_size *= i_dims[i];
            }
            end_dims[axis] += i_dims[axis];
            // copy region
            vx_size zeros[8] = { 0 };
            ERROR_CHECK_STATUS(vxCopyTensorPatch((vx_tensor)parameters[i],
                    num_of_dims, zeros, i_dims, i_stride,
                    local_buffer, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
            ERROR_CHECK_STATUS(vxCopyTensorPatch((vx_tensor)parameters[MAX_CONCAT_LAYER_INPUTS+1],
                    num_of_dims, begin_dims, end_dims, stride,
                    local_buffer, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
            begin_dims[axis] = end_dims[axis];
        }
    }

    return VX_SUCCESS;
}

////////
//! \brief register concat kernel
vx_status registerConcatKernel(vx_context context)
{
    // register a custom kernel for concat
    vx_kernel concat_kernel = vxAddUserKernel(context,
            "openvx_tutorial.nn_ext2.concat",
            VX_KERNEL_CONCAT_LAYER,
            concat_host_compute,
            MAX_CONCAT_LAYER_INPUTS+2,
            concat_validator,
            concat_initializer,
            concat_uninitializer);
    ERROR_CHECK_OBJECT(concat_kernel);
    ERROR_CHECK_STATUS(vxAddParameterToKernel(concat_kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    for(size_t i = 1; i < MAX_CONCAT_LAYER_INPUTS; i++) {
        ERROR_CHECK_STATUS(vxAddParameterToKernel(concat_kernel, i, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_OPTIONAL));
    }
    ERROR_CHECK_STATUS(vxAddParameterToKernel(concat_kernel, MAX_CONCAT_LAYER_INPUTS, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(concat_kernel, MAX_CONCAT_LAYER_INPUTS+1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxFinalizeKernel(concat_kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&concat_kernel));

    return VX_SUCCESS;
}

////////
//! \brief concat kernel node API
VX_API_EXPORT vx_node VX_API_CALL vxConcatLayer(
    vx_graph graph,
    vx_tensor input[],
    size_t num_inputs,
    vx_size axis,
    vx_tensor output)
{
    vx_node node = NULL;
    vx_context context = vxGetContext((vx_reference)graph);
    if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {
        vx_reference params[MAX_CONCAT_LAYER_INPUTS+2] = { 0 };
        vx_scalar axis_obj = vxCreateScalar(context, VX_TYPE_SIZE, &axis);
        ERROR_CHECK_OBJECT(axis_obj);
        for(size_t i = 0; i < num_inputs; i++) {
            params[i] = (vx_reference)input[i];
        }
        params[MAX_CONCAT_LAYER_INPUTS] = (vx_reference)axis_obj;
        params[MAX_CONCAT_LAYER_INPUTS+1] = (vx_reference)output;
        node = createNodeFromKernelEnum(graph, VX_KERNEL_CONCAT_LAYER, params, sizeof(params) / sizeof(params[0]));
        ERROR_CHECK_STATUS(vxReleaseScalar(&axis_obj));
    }
    return node;
}
