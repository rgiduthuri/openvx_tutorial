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
 * \file main.cpp
 * \example solution_exercise3
 * \brief Access to OpenCV medianBlur function as an OpenVX user kernel.
 * Look for TODO keyword in comments to code snipets that you need write.
 * \author Radhakrishna Giduthuri <radha.giduthuri@ieee.org>
 */

////////
// include OpenCV wrapper for image capture and display
#include "opencv_camera_display.h"

////////
// Most important top-level OpenVX header files are "VX/vx.h" and "VX/vxu.h".
// The "VX/vx.h" includes all headers needed to support functionality of the
// OpenVX specification, except for immediate mode functions, and it includes:
//    VX/vx_types.h     -- type definitions required by the OpenVX standard
//    VX/vx_api.h       -- All framework API definitions
//    VX/vx_kernels.h   -- list of supported kernels in the OpenVX standard 
//    VX/vx_nodes.h     -- 
//    VX/vx_vendors.h
// The "VX/vxu.h" defines immediate mode utility functions (not needed here).
#include <VX/vx.h>

////////
// Useful macros for OpenVX error checking:
//   ERROR_CHECK_STATUS      - check status is VX_SUCCESS
//   ERROR_CHECK_OBJECT      - check if the object creation is successful
#define ERROR_CHECK_STATUS(status) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }
#define ERROR_CHECK_OBJECT(obj) { \
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
// TODO:********
//   1. Define USER_LIBRARY_EXAMPLE
//   2. Define USER_KERNEL_MEDIAN_BLUR using VX_KERNEL_BASE() macro
enum user_library_e {
    USER_LIBRARY_EXAMPLE        = 1,
};
enum user_kernel_e {
    USER_KERNEL_MEDIAN_BLUR     = VX_KERNEL_BASE(VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE) + 0x001,
};

////////
// The node creation interface for the "app.userkernels.median_blur" kernel.
// This user kernel example expects parameters in the following order:
//   parameter#0 -- input image of format VX_DF_IMAGE_U8
//   parameter#1 -- output image of format VX_DF_IMAGE_U8
//   parameter#2 -- scalar ksize of type VX_TYPE_INT32
// TODO:********
//   1. Use vxGetKernelByEnum API to get kernel object from USER_KERNEL_MEDIAN_BLUR.
//      Note that you need to use vxGetContext API to get context from a graph object.
//   2. Use vxCreateGenericNode API to create a node from kernel object.
//   3. Create scalar objects for ksize parameter
//   4. Use vxSetParameterByIndex API to set node arguments
//   5. Release the kernel and scalar objects that are not needed anymore
//   6. Make sure to ERROR_CHECK_OBJECT and ERROR_CHECK_STATUS macros for error detection.
vx_node userMedianBlurNode( vx_graph graph, vx_image input, vx_image output, vx_int32 ksize)
{
    vx_context context = vxGetContext((vx_reference) graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_MEDIAN_BLUR);
    ERROR_CHECK_OBJECT(kernel);
    vx_node node = vxCreateGenericNode(graph, kernel);
    ERROR_CHECK_OBJECT(node);

    vx_scalar s_ksize = vxCreateScalar( context, VX_TYPE_INT32, &ksize );
    ERROR_CHECK_OBJECT(s_ksize);

    ERROR_CHECK_STATUS(vxSetParameterByIndex( node, 0, (vx_reference) input ));
    ERROR_CHECK_STATUS(vxSetParameterByIndex( node, 1, (vx_reference) output ));
    ERROR_CHECK_STATUS(vxSetParameterByIndex( node, 2, (vx_reference) s_ksize ));

    ERROR_CHECK_STATUS(vxReleaseScalar(&s_ksize));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));

    return node;
}

////////
// User kernel input validator callback should check to make sure that all the input
// parameters have correct data types. This user kernel example expects the inputs
// to be valid in the following order:
//   parameter#0 -- input image of format VX_DF_IMAGE_U8
//   parameter#2 -- scalar ksize of type VX_TYPE_INT32
// TODO:********
//   1. Use vxGetParameterByIndex API to get access to requested parameter
//   2. Use vxQueryParameter API with VX_PARAMETER_ATTRIBUTE_REF to access the input object
//   3. If the index is 0, check to make sure that image format is VX_DF_FORMAT_U8
//   4. If the index is 2, check to make sure that the scalar type is VX_TYPE_INT32
vx_status VX_CALLBACK median_blur_input_validator( vx_node node, vx_uint32 index )
{
    vx_reference ref = NULL;
    vx_parameter parameter = vxGetParameterByIndex( node, index );
    ERROR_CHECK_STATUS(vxQueryParameter( parameter, VX_PARAMETER_ATTRIBUTE_REF, &ref, sizeof( ref )));
    ERROR_CHECK_STATUS(vxReleaseParameter(&parameter));
    ERROR_CHECK_OBJECT(ref);

    if(index == 0)
    { // parameter#0 -- input image of format VX_DF_IMAGE_U8
        vx_df_image format = VX_DF_IMAGE_VIRT;
        ERROR_CHECK_STATUS(vxQueryImage((vx_image)ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof( format )));
        ERROR_CHECK_STATUS(vxReleaseImage((vx_image *)&ref));
        if( format != VX_DF_IMAGE_U8 )
            return VX_ERROR_INVALID_FORMAT;
    }
    else if(index == 2)
    { // parameters#2 -- scalar of type VX_TYPE_INT32
        vx_enum type = VX_TYPE_INVALID;
        ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)ref, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof( type )));
        ERROR_CHECK_STATUS(vxReleaseScalar((vx_scalar *)&ref));
        if(type != VX_TYPE_INT32)
            return VX_ERROR_INVALID_TYPE;
    }
    else
    { // invalid input parameter
        return VX_ERROR_INVALID_PARAMETERS;
    }

    return VX_SUCCESS;
}

////////
// User kernel output validator callback should set the output parameter meta data
// This user kernel example has only one output parameter with same dimensions as parameter#0.
//   parameter#1 -- output image of format VX_DF_IMAGE_U8 with same dimensions as input
// TODO:********
//   1. Get the input image dimensions from parameter#0
//   2. Set the output image meta data to have same dimensions as input and VX_DF_FORMAT_U8
vx_status VX_CALLBACK median_blur_output_validator( vx_node node, vx_uint32 index, vx_meta_format meta )
{
    if(index == 1)
    {
        vx_uint32 width = 0, height = 0;
        vx_image input = NULL;
        vx_parameter parameter = vxGetParameterByIndex( node, 0 );
        ERROR_CHECK_STATUS(vxQueryParameter( parameter, VX_PARAMETER_ATTRIBUTE_REF, &input, sizeof( input )));
        ERROR_CHECK_STATUS(vxReleaseParameter(&parameter));
        ERROR_CHECK_OBJECT(input);
        ERROR_CHECK_STATUS(vxQueryImage(input, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
        ERROR_CHECK_STATUS(vxQueryImage(input, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
        ERROR_CHECK_STATUS(vxReleaseImage(&input));

        vx_df_image format = VX_DF_IMAGE_U8;
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof(format)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
        ERROR_CHECK_STATUS(vxSetMetaFormatAttribute(meta, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));
    }
    else
    { // invalid input parameter
        return VX_ERROR_INVALID_PARAMETERS;
    }

    return VX_SUCCESS;
}

////////
// User kernel host side function gets called to execute the user kernel node.
// You need to wrap OpenVX object into OpenCV Mat objects and call cv::medianBlur.
// TODO:********
//   1. Get ksize value from scalar object in refs[2].
//   2. Access input and output image patches and wrap it in a cv::Mat object.
//      Use the cv::Mat mat(width, height, CV_U8, ptr, addr.stride_y) to wrap
//      an OpenVX image plane buffer in an OpenCV mat object. Note that
//      you need to access input image with VX_READ_ONLY and output image
//      with VX_WRITE_ONLY.
//   3. Just call cv::medianBlur(input, output, ksize)
//   4. Use vxCommitImagePatch API to give the image buffers control back to OpenVX framework
vx_status VX_CALLBACK median_blur_host_side_function( vx_node node, const vx_reference * refs, vx_uint32 num )
{
    vx_image input = (vx_image)refs[0];
    vx_image output = (vx_image)refs[1];
    vx_scalar s_ksize = (vx_scalar)refs[2];

    vx_int32 ksize = 0;
    ERROR_CHECK_STATUS(vxReadScalarValue(s_ksize, &ksize));

    vx_uint32 width = 0, height = 0;
    ERROR_CHECK_STATUS(vxQueryImage(input, VX_IMAGE_ATTRIBUTE_WIDTH, &width, sizeof(width)));
    ERROR_CHECK_STATUS(vxQueryImage(input, VX_IMAGE_ATTRIBUTE_HEIGHT, &height, sizeof(height)));

    vx_rectangle_t rect = { 0, 0, width, height };
    vx_imagepatch_addressing_t addr_input = { 0 }, addr_output = { 0 };
    void * ptr_input = NULL, * ptr_output = NULL;
    ERROR_CHECK_STATUS(vxAccessImagePatch(input, &rect, 0, &addr_input, &ptr_input, VX_READ_ONLY));
    ERROR_CHECK_STATUS(vxAccessImagePatch(output, &rect, 0, &addr_output, &ptr_output, VX_WRITE_ONLY));

    cv::Mat mat_input(height, width, CV_8U, ptr_input, addr_input.stride_y);
    cv::Mat mat_output(height, width, CV_8U, ptr_output, addr_output.stride_y);
    cv::medianBlur(mat_input, mat_output, ksize);

    ERROR_CHECK_STATUS(vxCommitImagePatch(input, &rect, 0, &addr_input, ptr_input));
    ERROR_CHECK_STATUS(vxCommitImagePatch(output, &rect, 0, &addr_output, ptr_output));

    return VX_SUCCESS;
}

////////
// User kernels needs to be registered with every OpenVX context before use in a graph.
// TODO:********
//   1. Use vxAddKernel API to register "app.userkernels.median_blur" with
//      USER_KERNEL_MEDIAN_BLUR as kernel enumeration, numParams as 3, and
//      all of the user kernel callback functions you implemented above.
//   2. Use vxAddParameterToKernel API to specify direction, data_type, and
//      state of all 8 parameters to the kernel. Look into the comments of
//      userMedianBlurNode function (above) to details about the order of
//      kernel parameters and their types.
//   3. Use vxFinalizeKernel API to make the kernel ready to use in a graph.
//      Note that the kernel object is still valid after this call. So you
//      need to call vxReleaseKernel before returning from this function.
vx_status registerUserKernel( vx_context context )
{
    vx_kernel kernel = vxAddKernel( context,
            "app.userkernels.median_blur",
            USER_KERNEL_MEDIAN_BLUR,
            median_blur_host_side_function,
            3,   // numParams
            median_blur_input_validator,
            median_blur_output_validator,
            NULL,
            NULL
        );
    ERROR_CHECK_OBJECT(kernel);

    ERROR_CHECK_STATUS(vxAddParameterToKernel( kernel, 0, VX_INPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED )); // input
    ERROR_CHECK_STATUS(vxAddParameterToKernel( kernel, 1, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED )); // output
    ERROR_CHECK_STATUS(vxAddParameterToKernel( kernel, 2, VX_INPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED )); // ksize
    ERROR_CHECK_STATUS(vxFinalizeKernel( kernel ));
    ERROR_CHECK_STATUS(vxReleaseKernel( &kernel ));

    vxAddLogEntry((vx_reference) context, VX_SUCCESS, "OK: registered user kernel app.userkernels.median_blur\n" );
    return VX_SUCCESS;
}

////////
// log_callback function should implements a mechanism to print log messages 
// from OpenVX framework onto console.
void log_callback( vx_context context, vx_reference ref, 
    vx_status status, const vx_char string[] )
{
    printf( "LOG: [ %3d ] %s", status, string );
    fflush(stdout);
}

////////
// main() has all the OpenVX application code for this exercise.
// Command-line usage:
//   % solution_exercise2 [<video-sequence>|<camera-device-number>]
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
    if( !gui.Grab())
    {
        printf( "ERROR: input has no video\n" );
        return 1;
    }

    ////////
    // Set the application configuration parameters. Note that input video
    // sequence is an 8-bit RGB image with dimensions given by gui.GetWidth()
    // and gui.GetHeight(). Harris corners algorithm specific parameters are:
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
    // Register user kernels with the context
    // TODO:********
    //   1. Register user kernel with context by calling your implementation of "registerUserKernel()"
    ERROR_CHECK_STATUS(registerUserKernel( context ));

    ////////
    // Create OpenVX image object for input RGB image and median filter output image
    vx_image input_rgb_image = vxCreateImage( context, width, height, VX_DF_IMAGE_RGB );
    vx_image output_filtered_image = vxCreateImage( context, width, height, VX_DF_IMAGE_U8 );
    ERROR_CHECK_OBJECT(input_rgb_image);
    ERROR_CHECK_OBJECT(output_filtered_image);

    ////////********
    // Create graph object and intermediate image objects.
    // Given that input is an RGB image, it is best to extract a gray scale
    // from RGB image, which requires two steps:
    //   - perform RGB to IYUV color conversion
    //   - extract Y channel from IYUV image
    // This requires two intermediate OpenVX image objects. Since you don't
    // need to access these objects from the application, they can be virtual
    // objects that can be created using the vxCreateVirtualImage API.
    vx_graph graph = vxCreateGraph( context );
    ERROR_CHECK_OBJECT( graph );
    vx_image yuv_image = vxCreateVirtualImage( graph, width, height, VX_DF_IMAGE_IYUV );
    vx_image luma_image = vxCreateVirtualImage( graph, width, height, VX_DF_IMAGE_U8 );
    ERROR_CHECK_OBJECT( yuv_image );
    ERROR_CHECK_OBJECT( luma_image );

    ////////********
    // Now all the objects have been created to be able to build the graphs.
    // Build a graph that converts RGB into luma image and then performs median blur
    // using user kernel USER_KERNEL_MEDIAN_BLUR. And then use vxVerifyGraph API for
    // initialization.
    // TODO:********
    //   1. Use userMedianBlurNode function to add "median_blur" node.
    vx_node nodes[] = {
        vxColorConvertNode( graph, input_rgb_image, yuv_image ),
        vxChannelExtractNode( graph, yuv_image, VX_CHANNEL_Y, luma_image ),
        userMedianBlurNode( graph, luma_image, output_filtered_image, ksize ),
    };
    for(vx_size i = 0; i < sizeof(nodes)/sizeof(nodes[0]); i++) {
        ERROR_CHECK_OBJECT( nodes[i] );
        ERROR_CHECK_STATUS( vxReleaseNode( &nodes[i] ));
    }
    ERROR_CHECK_STATUS( vxReleaseImage( &yuv_image ));
    ERROR_CHECK_STATUS( vxReleaseImage( &luma_image ));
    ERROR_CHECK_STATUS( vxVerifyGraph( graph ));

    ////////
    // process video sequence frame by frame until end of sequence or aborted.
    for( int frame_index = 0; !gui.AbortRequested(); frame_index++ )
    {
        ////////
        // local variable for checking status of OpenVX API calls
        vx_status status;

        ////////
        // Copy input RGB frame from OpenCV to OpenVX. In order to do this,
        // you need to use vxAccessImagePatch and vxCommitImagePatch APIs.
        // See "VX/vx_api.h" for the description of these APIs.
        vx_rectangle_t cv_rgb_image_region;
        cv_rgb_image_region.start_x = 0;
        cv_rgb_image_region.start_y = 0;
        cv_rgb_image_region.end_x = width;
        cv_rgb_image_region.end_y = height;
        vx_imagepatch_addressing_t cv_rgb_image_layout;
        cv_rgb_image_layout.stride_x = 3;
        cv_rgb_image_layout.stride_y = gui.GetStride();
        vx_uint8 * cv_rgb_image_buffer = gui.GetBuffer();
        status = vxAccessImagePatch( input_rgb_image, &cv_rgb_image_region, 0, 
            &cv_rgb_image_layout, (void **)&cv_rgb_image_buffer, VX_WRITE_ONLY );
        ERROR_CHECK_STATUS( status );
        status = vxCommitImagePatch( input_rgb_image, &cv_rgb_image_region, 0,
            &cv_rgb_image_layout, cv_rgb_image_buffer );
        ERROR_CHECK_STATUS( status );

        ////////********
        // Now that input RGB image is ready, just run a graph.
        ERROR_CHECK_STATUS(vxProcessGraph( graph ));

        ////////
        // Display the output filtered image
        vx_rectangle_t rect = { 0, 0, width, height };
        vx_imagepatch_addressing_t addr = { 0 };
        void * ptr = NULL;
        ERROR_CHECK_STATUS(vxAccessImagePatch(output_filtered_image, &rect, 0, &addr, &ptr, VX_READ_ONLY));
        cv::Mat mat(height, width, CV_8U, ptr, addr.stride_y);
        cv::imshow("MedianBlur", mat);
        ERROR_CHECK_STATUS(vxCommitImagePatch(output_filtered_image, &rect, 0, &addr, ptr));

        ////////
        // Display the input RGB and grab next input RGB frame for next iteration
        char text[128];
        sprintf( text, "Keyboard ESC/Q-Quit SPACE-Pause [FRAME %d] [ksize %d]", frame_index, ksize );
        gui.DrawText( 0, 16, text );
        gui.Show();
        if( !gui.Grab())
        {
            // terminate the processing loop if end of sequence is detected
            gui.WaitForKey();
            break;
        }
    }

    ////////********
    // Release all the OpenVX objects created in this exercise and make sure
    // to release the context at the end. To release an OpenVX object, you
    // need to call vxRelease<Object> API which takes a pointer to the object.
    // If the release operation is successful, the OpenVX framework will
    // reset the object to NULL.
    ERROR_CHECK_STATUS( vxReleaseGraph( &graph ));
    ERROR_CHECK_STATUS( vxReleaseImage( &input_rgb_image ));
    ERROR_CHECK_STATUS( vxReleaseImage( &output_filtered_image ));
    ERROR_CHECK_STATUS( vxReleaseContext( &context ));

    return 0;
}
