/*
 * Permission is hereby granted, free of charge, to any person obtaining a
 * copy of this software and/or associated documentation files (the
 * "Materials"), to deal in the Materials without restriction, including
 * without limitation the rights to use, copy, modify, merge, publish,
 * distribute, sublicense, and/or sell copies of the Materials, and to
 * permit persons to whom the Materials are furnished to do so, subject to
 * the following conditions:
 *
 * The above copyright notice and this permission notice shall be included
 * in all copies or substantial portions of the Materials.
 *
 * THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
 * EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
 * MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
 * IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
 * CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
 * TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
 * MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.
 */

/*!
 * \file    my_vx_tensor_map_impl.c
 * \brief   This is a simple reference implementation (not optimized)
 *          to demonstrate vxMapTensorPatch functionality, since it is
 *          not yet available in KhronosGroup/OpenVX-sample-impl repo
 * \author  Radhakrishna Giduthuri <radhakrishna.giduthuri@ieee.org>
 */

#include <VX/vx_khr_opencl_interop.h>
#include "my_vx_tensor_map_impl.h"
#include "common.h"

////
// local data structure to pass data between vxMapTensorPatch & vxUnmapTensorPatch
//
struct my_vx_tensor_map_id {
    vx_size number_of_dims;
    vx_size view_start[VX_CONTEXT_MAX_TENSOR_DIMS];
    vx_size view_end[VX_CONTEXT_MAX_TENSOR_DIMS];
    vx_size stride[VX_CONTEXT_MAX_TENSOR_DIMS];
    void * ptr;
    vx_enum usage;
    vx_enum mem_type;
};

////
// simple reference implementation (not optimized) for vxMapTensorPatch
//   this creates a temporary buffer and uses a copy, instead of returning
//   OpenVX internal buffer allocated for the tensor (so the copy overheads)
//
vx_status myVxMapTensorPatch(
    vx_tensor                                   tensor,
    vx_size                                     number_of_dims,
    const vx_size*                              view_start,
    const vx_size*                              view_end,
    vx_map_id*                                  map_id,
    vx_size*                                    stride,
    void**                                      ptr,
    vx_enum                                     usage,
    vx_enum                                     mem_type)
{
    ////
    // calculate output stride values
    //
    vx_enum data_type;
    vx_size dims[VX_CONTEXT_MAX_TENSOR_DIMS];
    ERROR_CHECK_STATUS( vxQueryTensor(tensor, VX_TENSOR_DATA_TYPE, &data_type, sizeof(vx_enum)) );
    ERROR_CHECK_STATUS( vxQueryTensor(tensor, VX_TENSOR_DIMS, &dims, sizeof(vx_size)*number_of_dims) );
    switch(data_type) {
        case VX_TYPE_INT16: stride[0] = sizeof(vx_int16); break;
        case VX_TYPE_UINT8: stride[0] = sizeof(vx_uint8); break;
        default: return VX_ERROR_NOT_SUPPORTED;
    }
    for(size_t dim = 1; dim < number_of_dims; dim++) {
        stride[dim] = dims[dim-1] * stride[dim-1];
    }
    vx_size size = dims[number_of_dims-1] * stride[number_of_dims-1];

    ////
    // create map_id and allocate temporary buffers to return
    //
    my_vx_tensor_map_id * id = new my_vx_tensor_map_id;
    ERROR_CHECK_NOT_NULL( id );
    id->number_of_dims = number_of_dims;
    id->usage = usage;
    id->mem_type = mem_type;
    if(id->mem_type == VX_MEMORY_TYPE_OPENCL_BUFFER)
    {
        vx_context context = vxGetContext((vx_reference)tensor);
        cl_context opencl_ctx;
        ERROR_CHECK_STATUS( vxQueryContext(context, VX_CONTEXT_CL_CONTEXT, &opencl_ctx, sizeof(cl_context)) );
        cl_int err;
        id->ptr = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE, size, NULL, &err);
        ERROR_CHECK_STATUS( err );
    }
    else
    {
        id->ptr = new vx_uint8[size];
        ERROR_CHECK_NOT_NULL( id->ptr );
    }
    for(size_t dim = 0; dim < number_of_dims; dim++) {
        id->view_start[dim] = view_start ? view_start[dim] : 0;
        id->view_end[dim] = view_end ? view_end[dim] : dims[dim];
        id->stride[dim] = stride[dim];
    }
    *map_id = (vx_map_id)id;
    *ptr = id->ptr;

    ////
    // copy tensor patch into temporarily allocated map buffer if read requested
    //
    if(id->usage == VX_READ_ONLY || id->usage == VX_READ_AND_WRITE)
    {
        vx_status status = vxCopyTensorPatch(tensor, id->number_of_dims,
                    id->view_start, id->view_end, id->stride, id->ptr,
                    VX_READ_ONLY, id->mem_type);
        if(status != VX_SUCCESS)
        {
            // release resources in case or error
            if(id->mem_type == VX_MEMORY_TYPE_OPENCL_BUFFER) {
                ERROR_CHECK_STATUS( clReleaseMemObject((cl_mem)id->ptr) );
            }
            else {
                delete (vx_uint8 *)id->ptr;
            }
            delete id;
            return status;
        }
    }

    return VX_SUCCESS;
}

////
// simple reference implementation (not optimized) for vxUnmapTensorPatch
//   this uses the temporary buffer used for mapping in myVxMapTensorPatch
//
vx_status myVxUnmapTensorPatch(
    vx_tensor                                   tensor,
    const vx_map_id                             map_id)
{
    ////
    // access mapped data
    //
    my_vx_tensor_map_id * id = (my_vx_tensor_map_id *)map_id;

    ////
    // copy tensor patch into temporarily allocated map buffer if read requested
    //
    if(id->usage == VX_WRITE_ONLY || id->usage == VX_READ_AND_WRITE)
    {
        vx_status status = vxCopyTensorPatch(tensor, id->number_of_dims,
                    id->view_start, id->view_end, id->stride, id->ptr,
                    VX_READ_ONLY, id->mem_type);
        if(status != VX_SUCCESS) {
            // release resources in case or error
            if(id->mem_type == VX_MEMORY_TYPE_OPENCL_BUFFER) {
                ERROR_CHECK_STATUS( clReleaseMemObject((cl_mem)id->ptr) );
            }
            else {
                delete (vx_uint8 *)id->ptr;
            }
            delete id;
            return status;
        }
    }

    ////
    // release temporary buffers allocated for mapping
    //
    if(id->mem_type == VX_MEMORY_TYPE_OPENCL_BUFFER) {
        ERROR_CHECK_STATUS( clReleaseMemObject((cl_mem)id->ptr) );
    }
    else {
        delete (vx_uint8 *)id->ptr;
    }
    delete id;

    return VX_SUCCESS;
}
