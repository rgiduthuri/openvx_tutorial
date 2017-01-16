/*
 * Copyright (c) 2016 The Khronos Group Inc.
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
 * \file    solution_exercise4.cpp
 * \example solution_exercise4
 * \brief   OpenVX user kernel to implement cosine activation function on tensor objects.
 *          Look for TODO keyword in comments for the code snippets that you need to write.
 * \author  Radhakrishna Giduthuri <radha.giduthuri@ieee.org>
 */

////////
// Include OpenCV wrapper for image capture and display.
#include "opencv_camera_display.h"

////////
// The top-level OpenVX header file is "VX/vx.h".
// TODO: ****
// For tensors, we need extensions header file "vx_ext_amd.h".
#include <VX/vx.h>
#include <vx_ext_amd.h>


////////
// Useful macros for OpenVX error checking:
//   ERROR_CHECK_STATUS     - check status is VX_SUCCESS
//   ERROR_CHECK_OBJECT     - check if the object creation is successful
#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }

#define ERROR_CHECK_OBJECT( obj ) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }

////////
// User kernel should have a unique enumerations and name for user kernel:
//   USER_LIBRARY_EXAMPLE      - library ID for user kernels in this example
//   USER_KERNEL_TENSOR_COS    - enumeration for "app.userkernels.tensor_cos" kernel
//
// TODO:********
//   1. Define USER_LIBRARY_EXAMPLE
//   2. Define USER_KERNEL_TENSOR_COS using VX_KERNEL_BASE() macro
enum user_library_e
{
    USER_LIBRARY_EXAMPLE        = 1,
};
enum user_kernel_e
{
    USER_KERNEL_TENSOR_COS     = VX_KERNEL_BASE( VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE ) + 0x001,
};

////////
// The node creation interface for the "app.userkernels.tensor_cos" kernel.
// This user kernel example expects parameters in the following order:
//   parameter #0  --  input tensor  of format VX_TYPE_INT16
//   parameter #1  --  output tensor of format VX_TYPE_INT16
//
// TODO:********
//   1. Use vxGetKernelByEnum API to get a kernel object from USER_KERNEL_TENSOR_COS.
//      Note that you need to use vxGetContext API to get the context from a graph object.
//   2. Use vxCreateGenericNode API to create a node from the kernel object.
//   3. Use vxSetParameterByIndex API to set node arguments.
//   4. Release the kernel object that are not needed any more.
//   5. Use ERROR_CHECK_OBJECT and ERROR_CHECK_STATUS macros for error detection.
vx_node userTensorCosNode( vx_graph graph,
                           vx_tensor input,
                           vx_tensor output )
{
    vx_context context = vxGetContext( ( vx_reference ) graph );
    vx_kernel kernel   = vxGetKernelByEnum( context, USER_KERNEL_TENSOR_COS );
    ERROR_CHECK_OBJECT( kernel );
    vx_node node       = vxCreateGenericNode( graph, kernel );
    ERROR_CHECK_OBJECT( node );

    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 0, ( vx_reference ) input ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 1, ( vx_reference ) output ) );

    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );

    return node;
}

////////
// The user kernel validator callback should check to make sure that all the input
// parameters have correct data types and set meta format for the output parameters.
// The input parameters to be validated are:
//   parameter #0  --  input tensor of format VX_TYPE_INT16
// The output parameters that requires setting meta format is:
//   parameter #1  --  output tebsor of format VX_TYPE_INT16 with the same dimensions as input
// TODO:********
//   1. Query the input tensor for the dimensions and format.
//   2. Check to make sure that the input tensor format is VX_TYPE_INT16.
//   3. Set the required output tensor meta data as following:
//      - output tensor dimensions should be same as input tensor
//      - output tensor format should be VX_TYPE_INT16
//      - output tensor fixed-point position can be whatever the user requested
//          * query the output tensor for the fixed-point position value
//          * set the same value in output tensor meta data
vx_status VX_CALLBACK tensor_cos_validator( vx_node node,
                                             const vx_reference parameters[], vx_uint32 num,
                                             vx_meta_format metas[] )
{
    // parameter #0 -- query dimensions and format
    vx_size num_of_dims;
    ERROR_CHECK_STATUS( vxQueryTensor( ( vx_tensor )parameters[0], VX_TENSOR_NUM_OF_DIMS, &num_of_dims, sizeof( num_of_dims ) ) );
    if( num_of_dims > 4 ) // sanity check to avoid stack corruption with querying VX_TENSOR_DIMS below
    {
        return VX_ERROR_INVALID_DIMENSION;
    }
    vx_size dims[4];
    ERROR_CHECK_STATUS( vxQueryTensor( ( vx_tensor )parameters[0], VX_TENSOR_DIMS, &dims, num_of_dims * sizeof(vx_size) ) );
    vx_enum data_type;
    ERROR_CHECK_STATUS( vxQueryTensor( ( vx_tensor )parameters[0], VX_TENSOR_DATA_TYPE, &data_type, sizeof( data_type ) ) );

    // parameter #0 -- check input tensor format to be VX_TYPE_INF16
    if( data_type != VX_TYPE_INT16 )
    {
        return VX_ERROR_INVALID_FORMAT;
    }

    // parameter #1 -- query fixed-point position
    vx_uint8 fixed_point_pos;
    ERROR_CHECK_STATUS( vxQueryTensor( ( vx_tensor )parameters[1], VX_TENSOR_FIXED_POINT_POS, &fixed_point_pos, sizeof( fixed_point_pos ) ) );

    // parameter #1 -- set required output tensor meta data
    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( metas[1], VX_TENSOR_NUM_OF_DIMS,  &num_of_dims,  sizeof( num_of_dims ) ) );
    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( metas[1], VX_TENSOR_DIMS, &dims, sizeof( dims ) ) );
    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( metas[1], VX_TENSOR_DATA_TYPE, &data_type, sizeof( data_type ) ) );
    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( metas[1], VX_TENSOR_FIXED_POINT_POS, &fixed_point_pos, sizeof( fixed_point_pos ) ) );

    return VX_SUCCESS;
}

////////
// User kernel host side function gets called to execute the user kernel node on CPU.
// Perform element-wise consine function on input tensor to produce output tensor.
//
// TODO:********
//   1. Get fixed-point position and dimensions of input and output tensors.
//      Note that both input and output tensors have same dimensions.
//   2. Access input and output tensor object data using vxMapTensorPatch API.
//   3. Perform element-wise cosine function using fixed-point position.
//   4. Use vxUnmapTensorPatch API to give the data buffers control back to OpenVX framework.
vx_status VX_CALLBACK tensor_cos_host_side_function( vx_node node, const vx_reference * refs, vx_uint32 num )
{
    // Get fixed-point position and dimensions of input and output tensors.
    // Note that both input and output tensors have same dimensions.
    vx_tensor input   = ( vx_tensor ) refs[0];
    vx_tensor output  = ( vx_tensor ) refs[1];
    vx_size num_of_dims;
    vx_size dims[4] = { 1, 1, 1, 1 };
    vx_uint8 input_fixed_point_pos;
    vx_uint8 output_fixed_point_pos;
    ERROR_CHECK_STATUS( vxQueryTensor( input,  VX_TENSOR_NUM_OF_DIMS, &num_of_dims, sizeof( num_of_dims ) ) );
    ERROR_CHECK_STATUS( vxQueryTensor( input,  VX_TENSOR_DIMS, &dims, num_of_dims * sizeof(vx_size) ) );
    ERROR_CHECK_STATUS( vxQueryTensor( input,  VX_TENSOR_FIXED_POINT_POS, &input_fixed_point_pos, sizeof( input_fixed_point_pos ) ) );
    ERROR_CHECK_STATUS( vxQueryTensor( output, VX_TENSOR_FIXED_POINT_POS, &output_fixed_point_pos, sizeof( output_fixed_point_pos ) ) );

    // Access input and output tensor object data using vxMapTensorPatch API.
    vx_size zeros[4] = { 0 };
    vx_map_id map_input, map_output;
    vx_uint8 * buf_input, * buf_output;
    vx_size stride_input[4] = { 0 };
    vx_size stride_output[4] = { 0 };
    ERROR_CHECK_STATUS( vxMapTensorPatch( input,
                                          num_of_dims, zeros, dims,
                                          &map_input, stride_input,
                                          (void **)&buf_input, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0 ) );
    ERROR_CHECK_STATUS( vxMapTensorPatch( output,
                                          num_of_dims, zeros, dims,
                                          &map_output, stride_output,
                                          (void **)&buf_output, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0 ) );

    // Perform element-wise cosine function using fixed-point position.
    vx_float32 input_to_float_multiplier = 1.0f / (vx_float32)(1 << input_fixed_point_pos);
    vx_float32 output_to_int16_multiplier = (vx_float32)(1 << output_fixed_point_pos);
    for( vx_size dim3 = 0; dim3 < dims[3]; dim3++)
    {
        for( vx_size dim2 = 0; dim2 < dims[2]; dim2++)
        {
            for( vx_size dim1 = 0; dim1 < dims[1]; dim1++)
            {
                const vx_int16 * ibuf = (const vx_int16 *) (buf_input +
                                                            dim3 * stride_input[3] +
                                                            dim2 * stride_input[2] +
                                                            dim1 * stride_input[1] );
                vx_int16 * obuf = (vx_int16 *) (buf_output +
                                                dim3 * stride_output[3] +
                                                dim2 * stride_output[2] +
                                                dim1 * stride_output[1] );
                for( vx_size dim0 = 0; dim0 < dims[0]; dim0++)
                {
                    // no saturation done here
                    vx_int16 ivalue = ibuf[dim0];
                    vx_int16 ovalue = (vx_int16)(cosf((vx_float32)ivalue * input_to_float_multiplier) * output_to_int16_multiplier + 0.5f);
                    obuf[dim0] = ovalue;
                }
            }
        }
    }

    // Use vxUnmapTensorPatch API to give the data buffers control back to OpenVX framework.
    ERROR_CHECK_STATUS( vxUnmapTensorPatch( input,  map_input ) );
    ERROR_CHECK_STATUS( vxUnmapTensorPatch( output, map_output ) );

    return VX_SUCCESS;
}

////////
// The user kernel target support callback is responsible for informing the
// OpenVX framework whether the implementation is supported CPU and/or GPU.
// TODO:********
//   1. Set the supported_target_affinity to "AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU"
//! \brief The kernel target support callback.
vx_status VX_CALLBACK tensor_cos_query_target_support(vx_graph graph, vx_node node,
    vx_bool use_opencl_1_2,              // [input]  false: OpenCL driver is 2.0+; true: OpenCL driver is 1.2
    vx_uint32& supported_target_affinity // [output] must be set to AGO_TARGET_AFFINITY_CPU or AGO_TARGET_AFFINITY_GPU or (AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU)
    )
{
    supported_target_affinity = AGO_TARGET_AFFINITY_CPU | AGO_TARGET_AFFINITY_GPU;
    return VX_SUCCESS;
}

////////
// The user kernel OpenCL code generator callback is responsible for generation
// OpenCL code specific for the configuration of the node. It can make use of
// parameter meta data, such as, number of tensor object dimensions, data format,
// etc. to tune the code for best performance.
// TODO:********
//   1. Query the input/output tensor meta data.
//   2. Generate the OpenCL code and set kernel function name
//   3. Set the work_dim and global_work required by clEnqueueNDRangeKernel
vx_status VX_CALLBACK tensor_cos_opencl_codegen(
    vx_node node,                                  // [input] node
    const vx_reference parameters[],               // [input] parameters
    vx_uint32 num,                                 // [input] number of parameters
    bool opencl_load_function,                     // [input]  false: normal OpenCL kernel; true: reserved
    char opencl_kernel_function_name[64],          // [output] kernel_name for clCreateKernel()
    std::string& opencl_kernel_code,               // [output] string for clCreateProgramWithSource()
    std::string& opencl_build_options,             // [output] options for clBuildProgram()
    vx_uint32& opencl_work_dim,                    // [output] work_dim for clEnqueueNDRangeKernel()
    vx_size opencl_global_work[],                  // [output] global_work[] for clEnqueueNDRangeKernel()
    vx_size opencl_local_work[],                   // [output] local_work[] for clEnqueueNDRangeKernel()
    vx_uint32& opencl_local_buffer_usage_mask,     // [output] reserved: must be ZERO
    vx_uint32& opencl_local_buffer_size_in_bytes   // [output] reserved: must be ZERO
    )
{
    // Get fixed-point position and dimensions of input and output tensors.
    // Note that both input and output tensors have same dimensions.
    vx_tensor input   = ( vx_tensor ) parameters[0];
    vx_tensor output  = ( vx_tensor ) parameters[1];
    vx_size num_of_dims;
    vx_size dims[4] = { 1, 1, 1, 1 };
    vx_uint8 input_fixed_point_pos;
    vx_uint8 output_fixed_point_pos;
    ERROR_CHECK_STATUS( vxQueryTensor( input,  VX_TENSOR_NUM_OF_DIMS, &num_of_dims, sizeof( num_of_dims ) ) );
    ERROR_CHECK_STATUS( vxQueryTensor( input,  VX_TENSOR_DIMS, &dims, num_of_dims * sizeof(vx_size) ) );
    ERROR_CHECK_STATUS( vxQueryTensor( input,  VX_TENSOR_FIXED_POINT_POS, &input_fixed_point_pos, sizeof( input_fixed_point_pos ) ) );
    ERROR_CHECK_STATUS( vxQueryTensor( output, VX_TENSOR_FIXED_POINT_POS, &output_fixed_point_pos, sizeof( output_fixed_point_pos ) ) );

    // Get input/output stride values for use in OpenCL code generation
    vx_size input_stride[4], output_stride[4];
    ERROR_CHECK_STATUS( vxQueryTensor( input,  VX_TENSOR_STRIDE_OPENCL, &input_stride, num_of_dims * sizeof( vx_size ) ) );
    ERROR_CHECK_STATUS( vxQueryTensor( output, VX_TENSOR_STRIDE_OPENCL, &output_stride, num_of_dims * sizeof( vx_size ) ) );

    if( num_of_dims == 3 )
    {
        // The OpenCL kernel below uses 3-dimensional global_work and
        // each work item will process 2 consecutive elements in the tensor
        char item[8192];
        sprintf(item,
            "__kernel void tensor_cos(__global uchar * t0_buf, uint t0_offset,\n"
            "                         __global uchar * t1_buf, uint t1_offset)\n"
            "{\n"
            "  uint i0 = get_global_id(0);\n"
            "  uint i1 = get_global_id(1);\n"
            "  uint i2 = get_global_id(2);\n"
            "  t0_buf += t0_offset + i0 * 4 + i1 * %d + i2 * %d;\n" // input_stride[1], input_stride[2]
            "  t1_buf += t1_offset + i0 * 4 + i1 * %d + i2 * %d;\n" // output_stride[1], output_stride[2]
            "  short2 ivalue = *(__global short2 *)t0_buf;\n"
            "  float2 fvalue;\n"
            "  fvalue.s0 = cos(%16.10f * ivalue.s0) * %16.10f;\n" // 1/(1 << input_fixed_point_pos), 1 << output_fixed_point_pos
            "  fvalue.s1 = cos(%16.10f * ivalue.s1) * %16.10f;\n" // 1/(1 << input_fixed_point_pos), 1 << output_fixed_point_pos
            "  short2 ovalue = convert_short2_rte(fvalue);\n"
            "  *(__global short2 *)t1_buf = ovalue;\n"
            "}\n"
            , (vx_uint32)input_stride[1], (vx_uint32)input_stride[2]
            , (vx_uint32)output_stride[1], (vx_uint32)output_stride[2]
            , 1.0f/(float)(1 << input_fixed_point_pos), (float)(1 << output_fixed_point_pos)
            , 1.0f/(float)(1 << input_fixed_point_pos), (float)(1 << output_fixed_point_pos)
            );
        opencl_kernel_code = item;

        // set kernel function name
        strcpy(opencl_kernel_function_name, "tensor_cos");

        // set global work: note that global_work[0] will two elements per work-item
        opencl_work_dim = 3;
        opencl_global_work[0] = (dims[0] + 1) >> 1;
        opencl_global_work[1] =  dims[1];
        opencl_global_work[2] =  dims[2];
    }
    else
    {
        // not supported
        return VX_ERROR_NOT_SUPPORTED;
    }

    return VX_SUCCESS;
}

////////
// User kernels needs to be registered with every OpenVX context before use in a graph.
//
// TODO:********
//   1. Use vxAddUserKernel API to register "app.userkernels.tensor_cos" with
//      kernel enumeration = USER_KERNEL_TENSOR_COS, numParams = 2, and
//      all of the user kernel callback functions you implemented above.
//   2. Use vxSetKernelAttribute API to set the OpenCL target support and
//      OpenCL code generation callbacks using the following attributes:
//        - VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT
//        - VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK
//   3. Use vxAddParameterToKernel API to specify direction, data_type, and
//      state of all 2 parameters to the kernel. Look into the comments of
//      userTensorCosNode function (above) to details about the order of
//      kernel parameters and their types.
//   4. Use vxFinalizeKernel API to make the kernel ready to use in a graph.
//      Note that the kernel object is still valid after this call.
//      So you need to call vxReleaseKernel before returning from this function.
vx_status registerUserKernel( vx_context context )
{
    vx_kernel kernel = vxAddUserKernel( context,
                                    "app.userkernels.tensor_cos",
                                    USER_KERNEL_TENSOR_COS,
                                    tensor_cos_host_side_function,
                                    2,   // numParams
                                    tensor_cos_validator,
                                    NULL,
                                    NULL );
    ERROR_CHECK_OBJECT( kernel );

    // register the extension callbacks for OpenCL
    amd_kernel_query_target_support_f query_target_support_f = tensor_cos_query_target_support;
    amd_kernel_opencl_codegen_callback_f opencl_codegen_callback_f = tensor_cos_opencl_codegen;
    ERROR_CHECK_STATUS( vxSetKernelAttribute( kernel, VX_KERNEL_ATTRIBUTE_AMD_QUERY_TARGET_SUPPORT,
                                              &query_target_support_f, sizeof( query_target_support_f ) ) );
    ERROR_CHECK_STATUS( vxSetKernelAttribute( kernel, VX_KERNEL_ATTRIBUTE_AMD_OPENCL_CODEGEN_CALLBACK,
                                              &opencl_codegen_callback_f, sizeof( opencl_codegen_callback_f ) ) );

    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 0, VX_INPUT,  VX_TYPE_TENSOR,  VX_PARAMETER_STATE_REQUIRED ) ); // input
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR,  VX_PARAMETER_STATE_REQUIRED ) ); // output
    ERROR_CHECK_STATUS( vxFinalizeKernel( kernel ) );
    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );

    vxAddLogEntry( ( vx_reference ) context, VX_SUCCESS, "OK: registered user kernel app.userkernels.tensor_cos\n" );
    return VX_SUCCESS;
}

////////
// log_callback function implements a mechanism to print log messages
// from the OpenVX framework onto console.
void VX_CALLBACK log_callback( vx_context    context,
                   vx_reference  ref,
                   vx_status     status,
                   const vx_char string[] )
{
    printf( "LOG: [ status = %d ] %s\n", status, string );
    fflush( stdout );
}

////////
// main() has all the OpenVX application code for this exercise.
// Command-line usage:
//   % solution_exercise4 [<video-sequence>|<camera-device-number>]
// When neither video sequence nor camera device number is specified,
// it defaults to the video sequence in "PETS09-S1-L1-View001.avi".
int main( int argc, char * argv[] )
{
    // Get default video sequence when nothing is specified on command-line and
    // instantiate OpenCV GUI module for reading input RGB images and displaying
    // the image with OpenVX results
    const char * video_sequence = argv[1];
    CGuiModule gui( video_sequence );

    // Try grab first video frame from the sequence using cv::VideoCapture
    // and check if video frame is available
    if( !gui.Grab() )
    {
        printf( "ERROR: input has no video\n" );
        return 1;
    }

    ////////
    // Set the application configuration parameters. Note that input video
    // sequence is an 8-bit RGB image with dimensions given by gui.GetWidth()
    // and gui.GetHeight(). The parameters for the tensors are:
    //   tensor_dims                    - 3 dimensions of tensor [3 x <width> x <height>]
    //   tensor_input_fixed_point_pos   - fixed-point position for input tensor
    //   tensor_output_fixed_point_pos  - fixed-point position for output tensor
    vx_uint32  width                         = gui.GetWidth();
    vx_uint32  height                        = gui.GetHeight();
    vx_size    tensor_dims[3]                = { width, height, 3 }; // 3 channels (RGB)
    vx_uint8   tensor_input_fixed_point_pos  = 5; // input[-128..127] will be mapped to -4..3.96875
    vx_uint8   tensor_output_fixed_point_pos = 7; // output[-1..1] will be mapped to -128 to 128

    ////////
    // Create the OpenVX context and make sure returned context is valid and
    // register the log_callback to receive messages from OpenVX framework.
    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT( context );
    vxRegisterLogCallback( context, log_callback, vx_false_e );

    ////////
    // Register user kernels with the context.
    //
    // TODO:********
    //   1. Register user kernel with context by calling your implementation of "registerUserKernel()".
    ERROR_CHECK_STATUS( registerUserKernel( context ) );

    ////////
    // Create OpenVX tensor objects for input and output
    //
    // TODO:********
    //   1. Create tensor objects using tensor_dims, tensor_input_fixed_point_pos, and
    //      tensor_output_fixed_point_pos
    vx_tensor input_tensor   = vxCreateTensor( context, 3, tensor_dims, VX_TYPE_INT16, tensor_input_fixed_point_pos );
    vx_tensor output_tensor  = vxCreateTensor( context, 3, tensor_dims, VX_TYPE_INT16, tensor_output_fixed_point_pos );
    ERROR_CHECK_OBJECT( input_tensor );
    ERROR_CHECK_OBJECT( output_tensor );

    ////////
    // Create, build, and verify the graph with user kernel node.
    //
    // TODO:********
    //   1. Build a graph with just one node created using userTensorCosNode()
    vx_graph graph = vxCreateGraph( context );
    ERROR_CHECK_OBJECT( graph );
    vx_node cos_node = userTensorCosNode( graph, input_tensor, output_tensor );
    ERROR_CHECK_OBJECT( cos_node );
    ERROR_CHECK_STATUS( vxReleaseNode( &cos_node ) );
    ERROR_CHECK_STATUS( vxVerifyGraph( graph ) );

    ////////
    // Process the video sequence frame by frame until the end of sequence or aborted.
    cv::Mat bgrMatForOutputDisplay( height, width, CV_8UC3 );
    for( int frame_index = 0; !gui.AbortRequested(); frame_index++ )
    {
        ////////
        // Copy input RGB frame from OpenCV into input_tensor with UINT8 to Q10.5 (INT16) conversion.
        // In order to do this, vxMapTensorPatch API (see "vx_ext_amd.h").
        //
         // TODO:********
         //   1. Use vxMapTensorPatch API for access to input tensor object for writing
         //   2. Copy UINT8 data from OpenCV RGB image to tensor object
         //   3. Use vxUnmapTensorPatch API to return control of buffer back to framework
        vx_uint8 * cv_rgb_image_buffer = gui.GetBuffer();
        vx_size rgb_stride             = gui.GetStride();
        vx_size zeros[3]               = { 0 };
        vx_size tensor_stride[3];
        vx_map_id map_id;
        vx_uint8 * buf;
        ERROR_CHECK_STATUS( vxMapTensorPatch( input_tensor,
                                              3, zeros, tensor_dims,
                                              &map_id, tensor_stride,
                                              (void **)&buf, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0 ) );
        for( vx_size c = 0; c < 3; c++ )
        {
            for( vx_size y = 0; y < height; y++ )
            {
                const vx_uint8 * img = cv_rgb_image_buffer + y * rgb_stride + c;
                vx_int16 * inp = (vx_int16 *)(buf + y * tensor_stride[1] + c * tensor_stride[2]);
                for( vx_size x = 0; x < width; x++ )
                {
                    // convert 0..255 to Q10.5 [-4..3.96875 range] fixed-point format
                    inp[x] = (vx_int16)img[x * 3] - 128;
                }
            }
        }
        ERROR_CHECK_STATUS( vxUnmapTensorPatch( input_tensor, map_id ) );


        ////////
        // Now that input tensor is ready, just run the graph.
        //
        // TODO:********
        //   1. Call vxProcessGraph to execute the tensor_cos kernel in graph
        ERROR_CHECK_STATUS( vxProcessGraph( graph ) );

        ////////
        // Display the output tensor object as RGB image
        //
        // TODO:********
        //   1. Use vxMapTensorPatch API for access to output tensor object for reading
        //   2. Copy tensor object data into OpenCV RGB image
        //   3. Use vxUnmapTensorPatch API to return control of buffer back to framework
        ERROR_CHECK_STATUS( vxMapTensorPatch( output_tensor,
                                              3, zeros, tensor_dims,
                                              &map_id, tensor_stride,
                                              (void **)&buf, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, 0 ) );
        vx_uint8 * cv_bgr_image_buffer = bgrMatForOutputDisplay.data;
        vx_size bgr_stride             = bgrMatForOutputDisplay.step;
        for( vx_size c = 0; c < 3; c++ )
        {
            for( vx_size y = 0; y < height; y++ )
            {
                const vx_int16 * out = (const vx_int16 *)(buf + y * tensor_stride[1] + c * tensor_stride[2]);
                vx_uint8 * img = cv_bgr_image_buffer + y * bgr_stride + (2 - c); // (2 - c) for RGB to BGR conversion
                for( vx_size x = 0; x < width; x++ )
                {
                    // scale convert Q8.7 [-1..1 range] fixed-point format to 0..255 with saturation
                    vx_int16 value = out[x] + 128;
                    value = value > 255 ? 255 : value; // saturation needed
                    img[x * 3] = (vx_uint8)value;
                }
            }
        }
        cv::imshow( "Cosine", bgrMatForOutputDisplay );
        ERROR_CHECK_STATUS( vxUnmapTensorPatch( output_tensor, map_id ) );

        ////////
        // Display the results and grab the next input RGB frame for the next iteration.
        char text[128];
        sprintf( text, "Keyboard ESC/Q-Quit SPACE-Pause [FRAME %d] [fixed_point_pos input:%d output:%d]", frame_index, tensor_input_fixed_point_pos, tensor_output_fixed_point_pos );
        gui.DrawText( 0, 16, text );
        gui.Show();
        if( !gui.Grab() )
        {
            // Terminate the processing loop if the end of sequence is detected.
            gui.WaitForKey();
            break;
        }
    }

    ////////
    // To release an OpenVX object, you need to call vxRelease<Object> API which takes a pointer to the object.
    // If the release operation is successful, the OpenVX framework will reset the object to NULL.
    //
    // TODO:****
    //   1. Release graph and tensor objects
    ERROR_CHECK_STATUS( vxReleaseGraph( &graph ) );
    ERROR_CHECK_STATUS( vxReleaseTensor( &input_tensor ) );
    ERROR_CHECK_STATUS( vxReleaseTensor( &output_tensor ) );
    ERROR_CHECK_STATUS( vxReleaseContext( &context ) );

    return 0;
}
