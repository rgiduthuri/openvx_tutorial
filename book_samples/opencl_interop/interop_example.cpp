/*
 * Copyright (c) 2019 Radhakrishna Giduthuri
 *
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
 * \file    interop_example.c
 * \example interop_example
 * \brief   This example gives a brief overview of OpenVX OpenCL interop
 *          using custom user kernel implementation. This example makes
 *          use of my_vx_tensor_map_impl.cpp/h for vxMapTensorPatch and
 *          vxUnmapTensorPatch APIs since they're currently implemented
 *          in the OpenVX-Sample-Impl repo
 * \author  Radhakrishna Giduthuri <radhakrishna.giduthuri@ieee.org>
 */

#include <VX/vx.h>
#include <VX/vx_khr_opencl_interop.h>
#include "my_vx_tensor_map_impl.h"
#include "common.h"

struct hard_sigmoid_local_data {
    cl_command_queue opencl_cmdq;
    cl_kernel opencl_kernel;
    size_t global_work_size;
    vx_size number_of_dims;
};

////////
// Reference C implementation of Hard Sigmoid
//
float hard_sigmoid_c_ref(float x, float alpha, float beta)
{
    float y = alpha * x + beta;
    if (y < 0) y = 0;
    if (y > 1) y = 1;
    return y;
}

////////
// OpenVX user kernel function for "hard_sigmoid"
//
vx_status VX_CALLBACK hard_sigmoid_opencl_function(vx_node node,
                const vx_reference arg[], vx_uint32 num_args)
{
    ////
    // get node parameters
    //
    vx_tensor tensor_x_obj = (vx_tensor)arg[2];
    vx_tensor tensor_y_obj = (vx_tensor)arg[3];

    ////
    // get node local data from VX_NODE_LOCAL_DATA_PTR
    //
    hard_sigmoid_local_data * data;
    ERROR_CHECK_STATUS( vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)) );

    ////
    // get OpenCL buffer from OpenVX tensor object as the input
    //   of "hard_sigmoid" OpenCL kernel ("X" parameter)
    //
    cl_mem x_mem;
    vx_map_id x_map_id;
    vx_size x_stride[VX_CONTEXT_MAX_TENSOR_DIMS];
    ERROR_CHECK_STATUS( vxMapTensorPatch(tensor_x_obj, data->number_of_dims, NULL, NULL,
                            &x_map_id, x_stride, (void **)&x_mem,
                            VX_READ_ONLY, VX_MEMORY_TYPE_OPENCL_BUFFER) );

    ////
    // get OpenCL buffer from OpenVX tensor object as the output
    //   of "hard_sigmoid" OpenCL kernel ("Y" parameter)
    //
    cl_mem y_mem;
    vx_map_id y_map_id;
    vx_size y_stride[VX_CONTEXT_MAX_TENSOR_DIMS];
    ERROR_CHECK_STATUS( vxMapTensorPatch(tensor_y_obj, data->number_of_dims, NULL, NULL,
                            &y_map_id, y_stride, (void **)&y_mem,
                            VX_WRITE_ONLY, VX_MEMORY_TYPE_OPENCL_BUFFER) );

    ////
    // set OpenCL buffers as input and output of "hard_sigmoid" kernel
    //
    ERROR_CHECK_STATUS( clSetKernelArg(data->opencl_kernel, 2, sizeof(cl_mem), (void *)&x_mem) );
    ERROR_CHECK_STATUS( clSetKernelArg(data->opencl_kernel, 3, sizeof(cl_mem), (void *)&y_mem) );

    ////
    // queue the "hard_sigmoid" kernel for execution in the OpenVX internal command-queue
    //   for optimal performance the OpenVX will queue up other OpenCL kernel in the graph
    //   so that the device can execute several OpenCL kernels until there is a data dependency
    //   for processing/data-access outside the device (like host).
    //
    ERROR_CHECK_STATUS( clEnqueueNDRangeKernel(data->opencl_cmdq, data->opencl_kernel,
            1, NULL, &data->global_work_size, NULL, 0, NULL, NULL) );

    ////
    // give the ownership of the OpenCL buffers back to the OpenVX
    //
    ERROR_CHECK_STATUS( vxUnmapTensorPatch(tensor_x_obj, x_map_id) );
    ERROR_CHECK_STATUS( vxUnmapTensorPatch(tensor_y_obj, y_map_id) );

    return VX_SUCCESS;
}

////////
// validate "hard_sigmoid" user kernel:
//   - scalar data type is VX_TYPE_FLOAT32
//   - tensor data types is VX_TYPE_INT16 for 16-bit fixed-point Q7.8
//   - input and output tensor dimensions are same
//
vx_status VX_CALLBACK hard_sigmoid_validator(vx_node node,
                const vx_reference arg[], vx_uint32 num_args, vx_meta_format metas[])
{
    ////
    // check data types
    //
    vx_enum data_type;
    ERROR_CHECK_STATUS( vxQueryScalar((vx_scalar)arg[0], VX_SCALAR_TYPE, &data_type, sizeof(vx_enum)) );
    if(data_type != VX_TYPE_FLOAT32) {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    ERROR_CHECK_STATUS( vxQueryScalar((vx_scalar)arg[1], VX_SCALAR_TYPE, &data_type, sizeof(vx_enum)) );
    if(data_type != VX_TYPE_FLOAT32) {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    ERROR_CHECK_STATUS( vxQueryTensor((vx_tensor)arg[2], VX_TENSOR_DATA_TYPE, &data_type, sizeof(vx_enum)) );
    if(data_type != VX_TYPE_FLOAT32) {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    ERROR_CHECK_STATUS( vxQueryTensor((vx_tensor)arg[3], VX_TENSOR_DATA_TYPE, &data_type, sizeof(vx_enum)) );
    if(data_type != VX_TYPE_FLOAT32) {
        return VX_ERROR_INVALID_PARAMETERS;
    }

    ////
    // check tensor dimensions
    //
    vx_size num_dims_x, num_dims_y;
    ERROR_CHECK_STATUS( vxQueryTensor((vx_tensor)arg[2], VX_TENSOR_NUMBER_OF_DIMS, &num_dims_x, sizeof(vx_size)) );
    ERROR_CHECK_STATUS( vxQueryTensor((vx_tensor)arg[3], VX_TENSOR_NUMBER_OF_DIMS, &num_dims_y, sizeof(vx_size)) );
    if(num_dims_x != num_dims_y) {
        return VX_ERROR_INVALID_PARAMETERS;
    }
    vx_size dims_x[VX_CONTEXT_MAX_TENSOR_DIMS], dims_y[VX_CONTEXT_MAX_TENSOR_DIMS];
    ERROR_CHECK_STATUS( vxQueryTensor((vx_tensor)arg[2], VX_TENSOR_DIMS, dims_x, sizeof(vx_size)*num_dims_x) );
    ERROR_CHECK_STATUS( vxQueryTensor((vx_tensor)arg[3], VX_TENSOR_DIMS, dims_y, sizeof(vx_size)*num_dims_y) );
    for(size_t dim = 0; dim < num_dims_x; dim++) {
        if(dims_x[dim] != dims_y[dim]) {
            return VX_ERROR_INVALID_PARAMETERS;
        }
    }

    return VX_SUCCESS;
}

////////
// initialize "hard_sigmoid" user node:
//   - build the OpenCL kernel for hard_sigmoid
//   - initialize alpha & beta arguments
//   - calculate global_work_size for OpenCL kernel execution
//   - save the initialized resources in VX_NODE_LOCAL_DATA_PTR
//
vx_status VX_CALLBACK hard_sigmoid_init(vx_node node,
                const vx_reference arg[], vx_uint32 num_args)
{
    ////
    // allocate and initialize node's local data
    //
    hard_sigmoid_local_data * data = new hard_sigmoid_local_data;
    ERROR_CHECK_NOT_NULL( data );

    ////
    // calculate global work for the "hard_sigmoid" kernel
    //   in this example, each thread is working on a single element,
    //   so total number of work items is number of elements in the tensor
    //
    vx_size number_of_dims, dims[VX_CONTEXT_MAX_TENSOR_DIMS];
    vx_tensor tensor_y_obj = (vx_tensor)arg[3];
    ERROR_CHECK_STATUS( vxQueryTensor(tensor_y_obj, VX_TENSOR_NUMBER_OF_DIMS, &number_of_dims, sizeof(vx_size)) );
    ERROR_CHECK_STATUS( vxQueryTensor(tensor_y_obj, VX_TENSOR_DIMS, dims, sizeof(vx_size)*number_of_dims) );
    size_t num_tensor_elements = 1;
    for(auto dim : dims)
        num_tensor_elements *= dim;
    data->global_work_size = num_tensor_elements;
    data->number_of_dims = number_of_dims;

    ////
    // get OpenCL command-queue from the node and corresponding OpenCL device
    // to build OpenCL executable
    //
    cl_context opencl_ctx;
    cl_device_id opencl_device;
    ERROR_CHECK_STATUS( vxQueryNode(node, VX_NODE_CL_COMMAND_QUEUE,
                            &data->opencl_cmdq, sizeof(cl_command_queue)) );
    ERROR_CHECK_STATUS( clGetCommandQueueInfo(data->opencl_cmdq, CL_QUEUE_CONTEXT,
                                        sizeof(cl_context), &opencl_ctx, NULL) );
    ERROR_CHECK_STATUS( clGetCommandQueueInfo(data->opencl_cmdq, CL_QUEUE_DEVICE,
                                        sizeof(cl_device_id), &opencl_device, NULL) );

    ////
    // compile OpenCL C program for "hard_sigmoid" ang get OpenCL kernel
    //   alpha : float constant
    //   beta  : float constant
    //   X     : input 16-bit fixed-point Q7.8 buffer
    //   Y     : output 16-bit fixed-point Q7.8 buffer
    //
    static const char hard_sigmoid_program_source[] =
      "  // OpenCL kernel to compute hard sigmoid activation     \n"
      "  __kernel void hard_sigmoid(float alpha, float beta,     \n"
      "        __global const short * X, __global short * Y)     \n"
      "  {                                                       \n"
      "    // get the index of current data element              \n"
      "    size_t i = get_global_id(0);                          \n"
      "                                                          \n"
      "    // read and convert input into float from Q7.8        \n"
      "    float x = X[i]/256.0;                                 \n"
      "                                                          \n"
      "    // compute hard sigmoid for the current data element  \n"
      "    float y = fmin(fmax(alpha * x + beta, 0), 1);         \n"
      "                                                          \n"
      "    // convert the output to Q7.8 and write               \n"
      "    Y[i] = (short)(y * 256.0);                            \n"
      "  }                                                       \n";
    const char * program_strings[] = {
        hard_sigmoid_program_source
    };
    size_t program_sizes[] = {
        sizeof(hard_sigmoid_program_source)
    };
    cl_int err;
    cl_program hard_sigmoid_program = clCreateProgramWithSource(opencl_ctx,
            1, program_strings, program_sizes, &err);
    ERROR_CHECK_STATUS( err );
    ERROR_CHECK_STATUS( clBuildProgram(hard_sigmoid_program, 1, &opencl_device, NULL, NULL, NULL) );
    data->opencl_kernel = clCreateKernel(hard_sigmoid_program, "hard_sigmoid", &err);
    ERROR_CHECK_STATUS( err );
    ERROR_CHECK_STATUS( clReleaseProgram(hard_sigmoid_program) );
    ERROR_CHECK_STATUS( clReleaseDevice(opencl_device) );
    ERROR_CHECK_STATUS( clReleaseContext(opencl_ctx) );

    ////
    // set the constant alpha and beta arguments to OpenCL kernel
    //
    float alpha, beta;
    ERROR_CHECK_STATUS( vxCopyScalar((vx_scalar)arg[0], &alpha, VX_READ_ONLY, VX_MEMORY_TYPE_HOST) );
    ERROR_CHECK_STATUS( vxCopyScalar((vx_scalar)arg[1], &beta, VX_READ_ONLY, VX_MEMORY_TYPE_HOST) );
    ERROR_CHECK_STATUS( clSetKernelArg(data->opencl_kernel, 0, sizeof(float), (void *)&alpha) );
    ERROR_CHECK_STATUS( clSetKernelArg(data->opencl_kernel, 1, sizeof(float), (void *)&beta) );

    ////
    // save node local data as VX_NODE_LOCAL_DATA_PTR
    //
    ERROR_CHECK_STATUS( vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)) );

    return VX_SUCCESS;
}

////////
// uninitialize "hard_sigmoid" user node:
//   - release the OpenCL kernel of hard_sigmoid
//   - release local data memory
//
vx_status VX_CALLBACK hard_sigmoid_uninit(vx_node node,
                const vx_reference arg[], vx_uint32 num_args)
{
    ////
    // get node local data from VX_NODE_LOCAL_DATA_PTR
    //
    hard_sigmoid_local_data * data;
    ERROR_CHECK_STATUS( vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &data, sizeof(data)) );

    ////
    // release all resources
    //
    ERROR_CHECK_STATUS( clReleaseKernel(data->opencl_kernel) );
    delete data;

    return VX_SUCCESS;
}

////////
// register user kernel
//
vx_kernel register_hard_sigmoid_kernel(vx_context openvx_ctx)
{
    ////
    // register user kernel for "hard_sigmoid"
    //   1. allocate and register a user kernel enum in the OpenVX context
    //   2. register hard_sigmoid user kernel with callback functions (above)
    //
    vx_enum hard_sigmoid_kernel_id;
    ERROR_CHECK_STATUS( vxAllocateUserKernelId(openvx_ctx, &hard_sigmoid_kernel_id) );
    vx_kernel user_kernel = vxAddUserKernel(openvx_ctx,
            "app.userkernels.hard_sigmoid", hard_sigmoid_kernel_id,
            hard_sigmoid_opencl_function, 4,
            hard_sigmoid_validator,
            hard_sigmoid_init,
            hard_sigmoid_uninit);
    ERROR_CHECK_STATUS( vxGetStatus((vx_reference)user_kernel) );

    ////
    // set user kernel arguments
    //
    ERROR_CHECK_STATUS( vxAddParameterToKernel(user_kernel, 0, VX_INPUT, 
                            VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED) );
    ERROR_CHECK_STATUS( vxAddParameterToKernel(user_kernel, 1, VX_INPUT, 
                            VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED) );
    ERROR_CHECK_STATUS( vxAddParameterToKernel(user_kernel, 2, VX_INPUT, 
                            VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED) );
    ERROR_CHECK_STATUS( vxAddParameterToKernel(user_kernel, 3, VX_OUTPUT, 
                            VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED) );

    ////
    // specify that the user kernel is using OpenCL interop
    //
    vx_bool use_opencl_interop = vx_true_e;
    ERROR_CHECK_STATUS( vxSetKernelAttribute(user_kernel, VX_KERNEL_USE_OPENCL,
                            &use_opencl_interop, sizeof(vx_bool)) );

    ////
    // finalize the user kernel after setting VX_KERNEL_USE_OPENCL attribute
    //
    ERROR_CHECK_STATUS( vxFinalizeKernel(user_kernel) );

    return user_kernel;
}

int main()
{
    ////
    // hard_sigmoind example configuration
    //   1. hard_sigmoid constants: alpha, beta
    //   2. input buffer size
    //   3. create test input values and output reference values in
    //      16-bit fixed-point Q7.8 representation
    //
    float alpha = 0.9f;
    float beta = 0.1f;
    size_t num_tensor_elements = 1000;
    short * x_input = new short[num_tensor_elements];
    short * y_output_ref = new short[num_tensor_elements];
    ERROR_CHECK_NOT_NULL( x_input );
    ERROR_CHECK_NOT_NULL( y_output_ref );
    float bias = (float)num_tensor_elements/2;
    float norm = (float)num_tensor_elements;
    for(size_t i = 0; i < num_tensor_elements; i++) {
        // generate test input and calculate reference output
        float x =  5.0f * (i - bias)/norm;
        float y =  hard_sigmoid_c_ref(x, alpha, beta);
        // convert x & y from float to Q7.8
        x_input[i] = (short)(x * 256.0f);
        y_output_ref[i] = (short)(y * 256.0f);

    }

    ////
    // select an OpenCL device
    //
    cl_platform_id platform_id;
    cl_device_id device_id;
    cl_int err;
    ERROR_CHECK_STATUS( clGetPlatformIDs(1, &platform_id, NULL) );
    ERROR_CHECK_STATUS( clGetDeviceIDs(platform_id, CL_DEVICE_TYPE_DEFAULT, 1, &device_id, NULL) );

    ////
    // create OpenCL context
    //   device can be returned back to OpenCL once context is created
    //
    cl_context opencl_ctx;
    cl_context_properties ctxprop[] = { CL_CONTEXT_PLATFORM,
        (cl_context_properties)platform_id, 0, 0 };
    opencl_ctx = clCreateContext(ctxprop, 1, &device_id, NULL, NULL, &err);
    ERROR_CHECK_STATUS( err );
    ERROR_CHECK_STATUS( clReleaseDevice(device_id) );
    printf("OK: created OpenCL context\n");

    ////
    // create OpenCL command-queue for the device
    //
    cl_command_queue opencl_cmdq;
    opencl_cmdq = clCreateCommandQueue(opencl_ctx, device_id, 0, &err);
    ERROR_CHECK_STATUS( err );
    printf("OK: created OpenCL command-queue\n");

    ////
    // create OpenVX context with OpenCL interoperability
    //
    vx_context openvx_ctx = vxCreateContextFromCL(opencl_ctx, opencl_cmdq);
    ERROR_CHECK_STATUS( vxGetStatus((vx_reference)openvx_ctx) );
    printf("OK: created OpenVX context with OpenCL interoperability\n");

    ////
    // register "hard_sigmoid" OpenVX user kernel
    //
    vx_kernel openvx_hard_sigmoid_kernel = register_hard_sigmoid_kernel(openvx_ctx);
    printf("OK: registered OpenVX user kernel for hard_sigmoid\n");

    ////
    // create OpenVX buffers for hard_sigmoid inputs and outputs
    //
    vx_scalar scalar_alpha = vxCreateScalar(openvx_ctx, VX_TYPE_FLOAT32, &alpha);
    vx_scalar scalar_beta = vxCreateScalar(openvx_ctx, VX_TYPE_FLOAT32, &beta);
    vx_tensor tensor_x = vxCreateTensor(openvx_ctx, 1, &num_tensor_elements, VX_TYPE_INT16, 8);
    vx_tensor tensor_y = vxCreateTensor(openvx_ctx, 1, &num_tensor_elements, VX_TYPE_INT16, 8);
    ERROR_CHECK_STATUS( vxGetStatus((vx_reference)scalar_alpha) );
    ERROR_CHECK_STATUS( vxGetStatus((vx_reference)scalar_beta) );
    ERROR_CHECK_STATUS( vxGetStatus((vx_reference)tensor_x) );
    ERROR_CHECK_STATUS( vxGetStatus((vx_reference)tensor_y) );
    printf("OK: created OpenVX data objects for hard_sigmoid test\n");

    ////
    // create OpenVX graph
    //
    vx_graph graph = vxCreateGraph(openvx_ctx);
    ERROR_CHECK_STATUS( vxGetStatus((vx_reference)graph) );
    printf("OK: created OpenVX graph objects\n");

    ////
    // add a node of hard_sigmoid kernel into OpenVX graph and set it's arguments
    //   the node object can be released after initializing the parameters
    //
    vx_node hard_sigmoid_node = vxCreateGenericNode(graph, openvx_hard_sigmoid_kernel);
    ERROR_CHECK_STATUS( vxGetStatus((vx_reference)hard_sigmoid_node) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex(hard_sigmoid_node, 0, (vx_reference) scalar_alpha) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex(hard_sigmoid_node, 1, (vx_reference) scalar_beta) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex(hard_sigmoid_node, 2, (vx_reference) tensor_x) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex(hard_sigmoid_node, 3, (vx_reference) tensor_y) );
    ERROR_CHECK_STATUS( vxReleaseNode(&hard_sigmoid_node) );
    printf("OK: inserted hard_sigmoid node into the graph\n");

    ////
    // verify the OpenVX graph
    //
    ERROR_CHECK_STATUS( vxVerifyGraph(graph) );
    printf("OK: verified the graph\n");

    ////
    // initialize input tensor
    //
    vx_size zero = 0, stride = sizeof(vx_int16);
    ERROR_CHECK_STATUS( vxCopyTensorPatch(tensor_x, 1, &zero, &num_tensor_elements,
                            &stride, x_input, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST) );
    printf("OK: initialized input tensor for hard_sigmoid\n");

    ////
    // execute the OpenVX graph
    //
    ERROR_CHECK_STATUS( vxProcessGraph(graph) );
    printf("OK: processed the graph with hard_sigmoid\n");

    ////
    // read the graph output and compare with reference output
    //
    short * y_output;
    vx_map_id map_id;
    ERROR_CHECK_STATUS( vxMapTensorPatch(tensor_y, 1, NULL, NULL, &map_id,
                            &stride, (void **)&y_output, VX_READ_ONLY, VX_MEMORY_TYPE_HOST) );
    printf("OK: mapped OpenVX output buffer to host address space\n");
    float err_square = 0;
    for (size_t i = 0; i < num_tensor_elements; i++) {
        short err_q78 = (y_output[i] - y_output_ref[i]);
        float err = err_q78 / 256.0f;
        err_square += err * err;
    }
    float mse = err_square/num_tensor_elements;
    ERROR_CHECK_STATUS( vxUnmapTensorPatch(tensor_y, map_id) );
    if(mse > 1e-4f) {
        printf("ERROR: something is wrong: MSE is too high: MSE = %.6f\n", mse);
        exit(1);
    }
    printf("OK: computed MSE against reference: MSE = %.6g (expected)\n", mse);

    ////
    // release all resources
    //
    delete[] x_input;
    delete[] y_output_ref;
    ERROR_CHECK_STATUS( vxReleaseContext(&openvx_ctx) );
    printf("OK: release all OpenVX resources\n");

    return 0;
}
