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
 * \file    opencl_example.c
 * \example opencl_example
 * \brief   This example gives a brief overview of OpenCL using
 *          16-bit fixed-point hard_sigmoid activation function
 * \author  Radhakrishna Giduthuri <radhakrishna.giduthuri@ieee.org>
 */

#if __APPLE__
#include <OpenCL/cl.h>
#else
#include <CL/cl.h>
#endif
#include "common.h"

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
    // keep the OpenCL C program source into a char array
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

    ////
    // compile OpenCL C program from source
    //
    const char * program_strings[] = {
        hard_sigmoid_program_source
    };
    size_t program_sizes[] = {
        sizeof(hard_sigmoid_program_source)
    };
    cl_program hard_sigmoid_program = clCreateProgramWithSource(opencl_ctx,
            1, program_strings, program_sizes, &err);
    ERROR_CHECK_STATUS( err );
    ERROR_CHECK_STATUS( clBuildProgram(hard_sigmoid_program, 1, &device_id, NULL, NULL, NULL) );
    printf("OK: compiled OpenCL program for hard_sigmoid kernel\n");

    ////
    // get kernel object for the "hard_sigmoid" kernel function in program
    //   once kernel object objects are created, the OpenCL program objects
    //   are not needed anymore
    //
    cl_kernel hard_sigmoid_kernel = clCreateKernel(hard_sigmoid_program, "hard_sigmoid", &err);
    ERROR_CHECK_STATUS( err );
    printf("OK: created hard_sigmoid OpenCL kernel object\n");
    ERROR_CHECK_STATUS( clReleaseProgram(hard_sigmoid_program) );
    printf("OK: released hard_sigmoid OpenCL program object (not needed anymore)\n");

    ////
    // create memory buffers for hard_sigmoid input and output
    //   16-bit fixed-point Q7.8 buffers
    //
    cl_mem x_mem = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE,
                        num_tensor_elements * sizeof(short), NULL, &err);
    ERROR_CHECK_STATUS( err );
    cl_mem y_mem = clCreateBuffer(opencl_ctx, CL_MEM_READ_WRITE,
                        num_tensor_elements * sizeof(short), NULL, &err);
    ERROR_CHECK_STATUS( err );
    printf("OK: created OpenCL buffers for hard_sigmoid input and output\n");

    ////
    // set "hard_sigmoid" kernel arguments:
    //    argument #0: "float alpha"
    //    argument #1: "float beta"
    //    argument #2: "__globla const short * X"
    //    argument #3: "__globla short * Y"
    //
    ERROR_CHECK_STATUS( clSetKernelArg(hard_sigmoid_kernel, 0, sizeof(float), (void *)&alpha) );
    ERROR_CHECK_STATUS( clSetKernelArg(hard_sigmoid_kernel, 1, sizeof(float), (void *)&beta) );
    ERROR_CHECK_STATUS( clSetKernelArg(hard_sigmoid_kernel, 2, sizeof(cl_mem), (void *)&x_mem) );
    ERROR_CHECK_STATUS( clSetKernelArg(hard_sigmoid_kernel, 3, sizeof(cl_mem), (void *)&y_mem) );
    printf("OK: set hard_sigmoid OpenCL kernel arguments\n");

    ////
    // initialize input buffer:
    //   1. map OpenCL buffer to host address space for writing (no copy from device to host)
    //   2. initialize input values
    //   3. unmap host address space so that kernel can access OpenCL buffer on device
    //
    short * x_buf = (short *)clEnqueueMapBuffer(opencl_cmdq, x_mem,
                                CL_TRUE, CL_MAP_WRITE, 0,
                                num_tensor_elements * sizeof(short),
                                0, NULL, NULL, &err);
    ERROR_CHECK_STATUS( err );
    for (size_t i = 0; i < num_tensor_elements; i++) {
        x_buf[i] = x_input[i];
    }
    ERROR_CHECK_STATUS( clEnqueueUnmapMemObject(opencl_cmdq, x_mem, x_buf, 0, NULL, NULL) );
    printf("OK: queue test data write to OpenCL input buffer on device\n");

    ////
    // run hard_sigmoid kernel parallely across "num_tensor_elements" work-items
    //   just queue up the job to execute after input buffer write is completed
    //
    size_t global_item_size = num_tensor_elements;
    ERROR_CHECK_STATUS( clEnqueueNDRangeKernel(opencl_cmdq, hard_sigmoid_kernel,
                            1, NULL, &global_item_size,
                            NULL, 0, NULL, NULL) );
    printf("OK: queued OpenCL kernel for execution\n");

    ////
    // read output from "hard_sigmoid" and compare with reference output
    //   the clEnqueueMapBuffer will return after the kernel execution as well as
    //   the read of output data from device to host address space is completed
    //
    short * y_buf = (short *)clEnqueueMapBuffer(opencl_cmdq, y_mem,
                                CL_TRUE, CL_MAP_READ, 0,
                                num_tensor_elements * sizeof(short),
                                0, NULL, NULL, &err);
    ERROR_CHECK_STATUS( err );
    printf("OK: mapped OpenCL output buffer to host address space\n");
    float err_square = 0;
    for (size_t i = 0; i < num_tensor_elements; i++) {
        short err_q78 = (y_buf[i] - y_output_ref[i]);
        float err = err_q78 / 256.0f;
        err_square += err * err;
    }
    float mse = err_square/num_tensor_elements;
    ERROR_CHECK_STATUS( clEnqueueUnmapMemObject(opencl_cmdq, y_mem, y_buf, 0, NULL, NULL) );
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
    ERROR_CHECK_STATUS( clReleaseKernel(hard_sigmoid_kernel) );
    ERROR_CHECK_STATUS( clReleaseCommandQueue(opencl_cmdq) );
    ERROR_CHECK_STATUS( clReleaseMemObject(x_mem) );
    ERROR_CHECK_STATUS( clReleaseMemObject(y_mem) );
    ERROR_CHECK_STATUS( clReleaseContext(opencl_ctx) );
    printf("OK: release all OpenCL resources\n");

    return 0;
}
