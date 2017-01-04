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
 * \file    exercise2.cpp
 * \example exercise2
 * \brief   Access to OpenCV medianBlur function as an OpenVX user kernel.
 *          Look for TODO STEP keyword in comments for the code snippets that you need to write.
 * \author  Radhakrishna Giduthuri <radha.giduthuri@ieee.org>
 */

////////
// Include OpenCV wrapper for image capture and display.
#include "opencv_camera_display.h"

////////
// The most important top-level OpenVX header files are "VX/vx.h" and "VX/vxu.h".
// The "VX/vx.h" includes all headers needed to support functionality of the
// OpenVX specification, except for immediate mode functions, and it includes:
//    VX/vx_types.h     -- type definitions required by the OpenVX standard
//    VX/vx_api.h       -- all framework API definitions
//    VX/vx_kernels.h   -- list of supported kernels in the OpenVX standard
//    VX/vx_nodes.h     -- easier-to-use functions for the kernels
//    VX/vx_vendors.h
// The "VX/vxu.h" defines the immediate mode utility functions (not needed here).
#include <VX/vx.h>


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
//   USER_KERNEL_MEDIAN_BLUR   - enumeration for "app.userkernels.median_blur" kernel
//
// TODO STEP 01:********
//   1. Define USER_LIBRARY_EXAMPLE. We gave the code in comments.
//   2. Define USER_KERNEL_MEDIAN_BLUR using VX_KERNEL_BASE() macro
//      We gave the most of the code in comments; fill in the missing parts.
enum user_library_e
{
//    USER_LIBRARY_EXAMPLE        = 1,
};
enum user_kernel_e
{
//    USER_KERNEL_MEDIAN_BLUR     = VX_KERNEL_BASE( VX_ID_DEFAULT, /* Fill in user library ID */ ) + 0x001,
};

////////
// The node creation interface for the "app.userkernels.median_blur" kernel.
// This user kernel example expects parameters in the following order:
//   parameter #0  --  input image  of format VX_DF_IMAGE_U8
//   parameter #1  --  output image of format VX_DF_IMAGE_U8
//   parameter #2  --  scalar ksize of type   VX_TYPE_INT32
//
// TODO STEP 02:********
//   1. Use vxGetKernelByEnum API to get a kernel object from USER_KERNEL_MEDIAN_BLUR.
//      Note that you need to use vxGetContext API to get the context from a graph object.
//      We gave most of the code in comments; fill in missing pieces.
//   2. Use vxCreateGenericNode API to create a node from the kernel object.
//      We gave code skeleton in comments; fill in missing pieces.
//   3. Create scalar objects for ksize parameter.
//      We gave code skeleton in comments; fill in missing pieces.
//   4. Use vxSetParameterByIndex API to set node arguments.
//      We gave code for one parameter; do similar for others.
//   5. Release the kernel and scalar objects that are not needed any more.
//      We gave code for one; do similar for the other.
//   6. Use ERROR_CHECK_OBJECT and ERROR_CHECK_STATUS macros for error detection.
//      We gave code for 3 error checks; do similar for remaining.
vx_node userMedianBlurNode( vx_graph graph,
                            vx_image input,
                            vx_image output,
                            vx_int32 ksize )
{
//    vx_context context = vxGetContext( ( vx_reference ) /* Fill in parameter */ );
//    vx_kernel kernel   = vxGetKernelByEnum( /* Fill in parameters */ );
//    ERROR_CHECK_OBJECT( kernel );

    vx_node node       = NULL; // vxCreateGenericNode( /* Fill in parameters */ );
//    ERROR_CHECK_OBJECT( node );

//    vx_scalar s_ksize  = vxCreateScalar( /* Fill in parameters */ );

//    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 0, ( vx_reference ) input ) );
//    ...

//    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );


    return node;
}

////////
// The user kernel validator callback should check to make sure that all the input
// parameters have correct data types and set meta format for the output parameters.
// The input parameters to be validated are:
//   parameter #0  --  input image  of format VX_DF_IMAGE_U8
//   parameter #2  --  scalar ksize of type   VX_TYPE_INT32
// The output parameters that requires setting meta format is:
//   parameter #1  --  output image of format VX_DF_IMAGE_U8 with the same dimensions as input
// TODO STEP 03:********
//   1. Check to make sure that the input image format is VX_DF_FORMAT_U8.
//   2. Check to make sure that the scalar type is VX_TYPE_INT32.
//   3. Query the input image for the image dimensions and set the output image
//      meta data to have the same dimensions as input and VX_DF_FORMAT_U8.
vx_status VX_CALLBACK median_blur_validator( vx_node node,
                                             const vx_reference parameters[], vx_uint32 num,
                                             vx_meta_format metas[] )
{
    // parameter #0 -- input image of format VX_DF_IMAGE_U8
//    vx_df_image format = VX_DF_IMAGE_VIRT;
//    ERROR_CHECK_STATUS( vxQueryImage( ( vx_image )parameters[0], /* Fill in parameters */, &format, sizeof( format ) ) );
//    if( format != /* Fill in value */ )
//    {
//        return VX_ERROR_INVALID_FORMAT;
//    }

    // parameter #2 -- scalar of type VX_TYPE_INT32
//    vx_enum type = VX_TYPE_INVALID;
//    ERROR_CHECK_STATUS( vxQueryScalar( /* Fill in parameters */ ) );
//    if( /* Fill in the condition */ )
//    {
//        return VX_ERROR_INVALID_TYPE;
//    }

    // parameter #1 -- output image should be VX_DF_FORMAT_U8 and should have same dimensions as input image
//    vx_uint32 width = 0, height = 0;
//    ERROR_CHECK_STATUS( vxQueryImage( ( vx_image )parameters[0], /* Fill in parameter */,  &width,  sizeof( width ) ) );
//    ERROR_CHECK_STATUS( vxQueryImage( /* Fill in parameter */ ) );
//    format = VX_DF_IMAGE_U8;
//    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( metas[1], VX_IMAGE_FORMAT, &format, sizeof( format ) ) );
//    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( /* Fill in parameters */,  &width,  sizeof( width ) ) );
//    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( /* Fill in parameters */ ) );

    return VX_SUCCESS;
}

////////
// User kernel host side function gets called to execute the user kernel node.
// You need to wrap the OpenVX objects into OpenCV Mat objects and call cv::medianBlur.
//
// TODO STEP 04:********
//   1. Get ksize value from scalar object in refs[2].
//      We gave code skeleton; fill in parameters.
//   2. Get the image dimensions from input image.
//      We gave code skeleton; fill in parameters.
//   3. Access input and output image patches and wrap it in a cv::Mat object.
//      Use the cv::Mat mat(width, height, CV_U8, ptr, addr.stride_y) to wrap
//      an OpenVX image plane buffer in an OpenCV mat object. Note that
//      you need to access input image with VX_READ_ONLY and output image
//      with VX_WRITE_ONLY using vxMapImagePatch API.
//      We gave code skeleton; fill in parameters.
//   4. Just call cv::medianBlur(input, output, ksize)
//      We gave code in comments.
//   5. Use vxUnmapImagePatch API to give the image buffers control back to OpenVX framework
//      We gave code for one; do similar for the other.
vx_status VX_CALLBACK median_blur_host_side_function( vx_node node, const vx_reference * refs, vx_uint32 num )
{
    vx_image  input   = ( vx_image ) refs[0];
    vx_image  output  = ( vx_image ) refs[1];
    vx_scalar s_ksize = ( vx_scalar ) refs[2];

//    vx_int32  ksize   = 0;
//    ERROR_CHECK_STATUS( vxCopyScalar( /* Fill in parameters */ ) );

//    vx_uint32 width = 0, height = 0;
//    ERROR_CHECK_STATUS( vxQueryImage( /* Fill in parameters */ ) );
//    ERROR_CHECK_STATUS( vxQueryImage( /* Fill in parameters */ ) );

//    vx_rectangle_t rect = { 0, 0, width, height };
//    vx_map_id map_input, map_output;
//    vx_imagepatch_addressing_t addr_input, addr_output;
//    void * ptr_input, * ptr_output;
//    ERROR_CHECK_STATUS( vxMapImagePatch( input,  /* Fill in parameters */,  VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X ) );
//    ERROR_CHECK_STATUS( vxMapImagePatch( output, /* Fill in parameters */, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X ) );

//    cv::Mat mat_input(  height, width, CV_8U, ptr_input,  addr_input .stride_y );
//    cv::Mat mat_output( /* Fill in parameters */ );
//    cv::medianBlur( mat_input, mat_output, ksize );

//    ERROR_CHECK_STATUS( vxUnmapImagePatch( input,  map_input ) );
//    ERROR_CHECK_STATUS( vxUnmapImagePatch( output, /* Fill in parameters */ ) );

    return VX_SUCCESS;
}

////////
// User kernels needs to be registered with every OpenVX context before use in a graph.
//
// TODO STEP 05:********
//   1. Use vxAddUserKernel API to register "app.userkernels.median_blur" with
//      kernel enumeration = USER_KERNEL_MEDIAN_BLUR, numParams = 3, and
//      all of the user kernel callback functions you implemented above.
//      We gave the code skeleton; fill in parameters.
//   2. Use vxAddParameterToKernel API to specify direction, data_type, and
//      state of all 3 parameters to the kernel. Look into the comments of
//      userMedianBlurNode function (above) to details about the order of
//      kernel parameters and their types.
//      We gave code for one parameter; do similar for other two.
//   3. Use vxFinalizeKernel API to make the kernel ready to use in a graph.
//      Note that the kernel object is still valid after this call.
//      So you need to call vxReleaseKernel before returning from this function.
//      We gave the code skeleton; fill in parameters.
vx_status registerUserKernel( vx_context context )
{
//    vx_kernel kernel = vxAddUserKernel( context,
//                                    "app.userkernels.median_blur",
//                                    USER_KERNEL_MEDIAN_BLUR,
//                                    /* Fill in parameters */ );
//    ERROR_CHECK_OBJECT( kernel );

//    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 0, VX_INPUT,  VX_TYPE_IMAGE,  VX_PARAMETER_STATE_REQUIRED ) ); // input
//    ...

//    ERROR_CHECK_STATUS( vxFinalizeKernel( /* Fill in parameter */ ) );
//    ERROR_CHECK_STATUS( vxReleaseKernel( /* Fill in parameter */ ) );


    vxAddLogEntry( ( vx_reference ) context, VX_SUCCESS, "OK: registered user kernel app.userkernels.median_blur\n" );
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
//   % exercise2 [<video-sequence>|<camera-device-number>]
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
    // and gui.GetHeight(). The parameters for the median filter are:
    //   ksize                   - median filter kernel size
    vx_uint32  width     = gui.GetWidth();
    vx_uint32  height    = gui.GetHeight();
    vx_int32   ksize     = 7;

    ////////
    // Create the OpenVX context and make sure returned context is valid and
    // register the log_callback to receive messages from OpenVX framework.
    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT( context );
    vxRegisterLogCallback( context, log_callback, vx_false_e );

    ////////
    // Register user kernels with the context.
    //
    // TODO STEP 06:********
    //   1. Register user kernel with context by calling your implementation of "registerUserKernel()".



    ////////
    // Create OpenVX image object for input RGB image and median filter output image.
    vx_image input_rgb_image       = vxCreateImage( context, width, height, VX_DF_IMAGE_RGB );
    vx_image output_filtered_image = vxCreateImage( context, width, height, VX_DF_IMAGE_U8 );
    ERROR_CHECK_OBJECT( input_rgb_image );
    ERROR_CHECK_OBJECT( output_filtered_image );

    ////////********
    // Create graph object and intermediate image objects.
    // Given that input is an RGB image, it is best to extract a gray image
    // from RGB image, which requires two steps:
    //   - perform RGB to IYUV color conversion
    //   - extract Y channel from IYUV image
    // This requires two intermediate OpenVX image objects. Since you don't
    // need to access these objects from the application, they can be virtual
    // objects that can be created using the vxCreateVirtualImage API.
    vx_graph graph      = vxCreateGraph( context );
    ERROR_CHECK_OBJECT( graph );
    vx_image yuv_image  = vxCreateVirtualImage( graph, width, height, VX_DF_IMAGE_IYUV );
    vx_image luma_image = vxCreateVirtualImage( graph, width, height, VX_DF_IMAGE_U8 );
    ERROR_CHECK_OBJECT( yuv_image );
    ERROR_CHECK_OBJECT( luma_image );

    ////////********
    // Now all the objects have been created for building the graph
    // that converts RGB into luma image and then performs median blur
    // using user kernel USER_KERNEL_MEDIAN_BLUR.
    // Then use vxVerifyGraph API for initialization.
    //
    // TODO STEP 07:********
    //   1. Use userMedianBlurNode function to add "median_blur" node.
    //      We gave most of the code in comments; fill in parameters for user kernel.
//    vx_node nodes[] =
//    {
//        vxColorConvertNode(   graph, input_rgb_image, yuv_image ),
//        vxChannelExtractNode( graph, yuv_image, VX_CHANNEL_Y, luma_image ),
//        userMedianBlurNode(   graph, /* Fill in parameters */ )
//    };
//    for( vx_size i = 0; i < sizeof( nodes ) / sizeof( nodes[0] ); i++ )
//    {
//        ERROR_CHECK_OBJECT( nodes[i] );
//        ERROR_CHECK_STATUS( vxReleaseNode( &nodes[i] ) );
//    }
//    ERROR_CHECK_STATUS( vxReleaseImage( &yuv_image ) );
//    ERROR_CHECK_STATUS( vxReleaseImage( &luma_image ) );
//    ERROR_CHECK_STATUS( vxVerifyGraph( graph ) );


    ////////
    // Process the video sequence frame by frame until the end of sequence or aborted.
    for( int frame_index = 0; !gui.AbortRequested(); frame_index++ )
    {
        ////////
        // Copy input RGB frame from OpenCV to OpenVX. In order to do this,
        // you need to use vxCopyImagePatch API.
        // See "VX/vx_api.h" for the description of this APIs.
        vx_rectangle_t cv_rgb_image_region;
        cv_rgb_image_region.start_x    = 0;
        cv_rgb_image_region.start_y    = 0;
        cv_rgb_image_region.end_x      = width;
        cv_rgb_image_region.end_y      = height;
        vx_imagepatch_addressing_t cv_rgb_image_layout;
        cv_rgb_image_layout.stride_x   = 3;
        cv_rgb_image_layout.stride_y   = gui.GetStride();
        vx_uint8 * cv_rgb_image_buffer = gui.GetBuffer();
        ERROR_CHECK_STATUS( vxCopyImagePatch( input_rgb_image, &cv_rgb_image_region, 0,
                                              &cv_rgb_image_layout, cv_rgb_image_buffer,
                                              VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST ) );

        ////////********
        // Now that input RGB image is ready, just run the graph.
        // TODO STEP 08:********
        //   1. Use vxProcessGraph to run the graph.
        //      We gave the code in comments; just uncomment.
//        ERROR_CHECK_STATUS( vxProcessGraph( graph ) );


        ////////
        // Display the output filtered image.
        vx_rectangle_t rect = { 0, 0, width, height };
        vx_map_id map_id;
        vx_imagepatch_addressing_t addr;
        void * ptr;
        ERROR_CHECK_STATUS( vxMapImagePatch( output_filtered_image, &rect, 0, &map_id, &addr, &ptr,
                                             VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X ) );
        cv::Mat mat( height, width, CV_8U, ptr, addr.stride_y );
        cv::imshow( "MedianBlur", mat );
        ERROR_CHECK_STATUS( vxUnmapImagePatch( output_filtered_image, map_id ) );

        ////////
        // Display the results and grab the next input RGB frame for the next iteration.
        char text[128];
        sprintf( text, "Keyboard ESC/Q-Quit SPACE-Pause [FRAME %d] [ksize %d]", frame_index, ksize );
        gui.DrawText( 0, 16, text );
        gui.Show();
        if( !gui.Grab() )
        {
            // Terminate the processing loop if the end of sequence is detected.
            gui.WaitForKey();
            break;
        }
    }

    ////////********
    // Query graph performance using VX_GRAPH_PERFORMANCE and print timing
    // in milliseconds. Note that time units of vx_perf_t fields are nanoseconds.
    vx_perf_t perf = { 0 };
    ERROR_CHECK_STATUS( vxQueryGraph( graph, VX_GRAPH_PERFORMANCE, &perf, sizeof( perf ) ) );
    printf( "GraphName NumFrames Avg(ms) Min(ms)\n"
            "Median    %9d %7.3f %7.3f\n",
            ( int )perf.num, ( float )perf.avg * 1e-6f, ( float )perf.min * 1e-6f );

    ////////********
    // To release an OpenVX object, you need to call vxRelease<Object> API which takes a pointer to the object.
    // If the release operation is successful, the OpenVX framework will reset the object to NULL.
    ERROR_CHECK_STATUS( vxReleaseGraph( &graph ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &input_rgb_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &output_filtered_image ) );
    ERROR_CHECK_STATUS( vxReleaseContext( &context ) );

    return 0;
}
