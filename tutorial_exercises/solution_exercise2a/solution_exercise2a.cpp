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
 * \file    solution_exercise2a.cpp
 * \example solution_exercise2a
 * \brief   Feature tracker example with two graphs and a user kernel
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
// log_callback function implements a mechanism to print log messages
// from OpenVX framework onto console. The log_callback function can be
// activated by calling vxRegisterLogCallback API in STEP 02.
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
//   % solution_exercise2 [<video-sequence>|<camera-device-number>]
// When neither video sequence nor camera device number is specified,
// it defaults to the video sequence in "PETS09-S1-L1-View001.avi".
int main( int argc, char * argv[] )
{
    // Get default video sequence when nothing is specified on command-line and
    // instantiate OpenCV GUI module for reading input RGB images and displaying
    // the image with OpenVX results.
    const char * video_sequence = argv[1];
    CGuiModule gui( video_sequence );

    // Try to grab the first video frame from the sequence using cv::VideoCapture
    // and check if a video frame is available.
    if( !gui.Grab() )
    {
        printf( "ERROR: input has no video\n" );
        return 1;
    }

    ////////
    // Set the application configuration parameters. Note that input video
    // sequence is an 8-bit RGB image with dimensions given by gui.GetWidth()
    // and gui.GetHeight(). The parameters for the Harris corners algorithm are:
    //   max_keypoint_count      - maximum number of keypoints to track
    //   harris_strength_thresh  - minimum threshold score to keep a corner
    //                             (computed using the normalized Sobel kernel)
    //   harris_min_distance     - radial L2 distance for non-max suppression
    //   harris_sensitivity      - sensitivity threshold k from the Harris-Stephens
    //   harris_gradient_size    - window size for gradient computation
    //   harris_block_size       - block window size used to compute the
    //                             Harris corner score
    //   lk_pyramid_levels       - number of pyramid levels for LK optical flow
    //   lk_termination          - can be VX_TERM_CRITERIA_ITERATIONS or
    //                               VX_TERM_CRITERIA_EPSILON or
    //                               VX_TERM_CRITERIA_BOTH
    //   lk_epsilon              - error for terminating the algorithm
    //   lk_num_iterations       - number of iterations
    //   lk_use_initial_estimate - turn on/off use of initial estimates
    //   lk_window_dimension     - size of window on which to perform the algorithm
    //   trackable_kp_ratio_thr  - threshold for ratio of keypoints being tracked and total keypoints
    vx_uint32  width                   = gui.GetWidth();
    vx_uint32  height                  = gui.GetHeight();
    vx_size    max_keypoint_count      = 10000;
    vx_float32 harris_strength_thresh  = 0.0005f;
    vx_float32 harris_min_distance     = 5.0f;
    vx_float32 harris_sensitivity      = 0.04f;
    vx_int32   harris_gradient_size    = 3;
    vx_int32   harris_block_size       = 3;
    vx_uint32  lk_pyramid_levels       = 6;
    vx_float32 lk_pyramid_scale        = VX_SCALE_PYRAMID_HALF;
    vx_enum    lk_termination          = VX_TERM_CRITERIA_BOTH;
    vx_float32 lk_epsilon              = 0.01f;
    vx_uint32  lk_num_iterations       = 5;
    vx_bool    lk_use_initial_estimate = vx_false_e;
    vx_uint32  lk_window_dimension     = 6;
    vx_float32 trackable_kp_ratio_thr  = 0.8f;

    ////////********
    // Create the OpenVX context and make sure the returned context is valid.
    //
    // TODO STEP 01:********
    //   1. Create an OpenVX context using vxCreateContext API.
    //   2. The vxGetStatus API is designed to check for object creation errors.
    //      Look at the definition of ERROR_CHECK_OBJECT macro to see how
    //      the vxGetStatus API used for error checking.
    //   3. Use ERROR_CHECK_OBJECT macro to check if context creation was successful.
    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT( context );


    ////////********
    // Register the log_callback that was implemented above main() and receive all
    // log messages from the OpenVX framework. All calls to vxAddLogEntry API from
    // framework, extensions, or user kernels will result in a call to log_callback.
    //
    // TODO STEP 02:********
    //   1. Uncomment the lines below to register log_callback using vxRegisterLogCallback API
    //      and test log message receiption using vxAddLogEntry API
    vxRegisterLogCallback( context, log_callback, vx_false_e );
    vxAddLogEntry( ( vx_reference ) context, VX_FAILURE, "Hello there!\n" );


    ////////
    // Create OpenVX image object for input RGB image.
    //
    // TODO STEP 03:********
    //   1. Create an image object for use as input. For image dimensions, use the
    //      width & height configuration parameters defined above. For the image
    //      format, use VX_DF_IMAGE_RGB enum.
    //   2. Use ERROR_CHECK_OBJECT to check proper creation of objects.
    vx_image input_rgb_image = vxCreateImage( context, width, height, VX_DF_IMAGE_RGB );
    ERROR_CHECK_OBJECT( input_rgb_image );


    ////////********
    // OpenVX optical flow functionality requires pyramids of the current input
    // image and the previous image. It also requires keypoints that correspond
    // to the previous pyramid and will output updated keypoints into
    // another keypoint array. To be able to toggle between the current and
    // the previous buffers, you need to use OpenVX delay objects and vxAgeDelay().
    // Create OpenVX pyramid and array object exemplars and create OpenVX delay
    // objects for both to hold two of each. Note that the exemplar objects are not
    // needed once the delay objects are created.
    //
    // TODO STEP 04:********
    //   1. Use vxCreatePyramid API to create a pyramid exemplar with the
    //      same dimensions as the input image, VX_DF_IMAGE_U8 as image format,
    //      lk_pyramid_levels as levels, and lk_pyramid_scale as scale.
    //      We gave code for this in comments.
    //   2. Use vxCreateArray API to create an array exemplar with
    //      keypoint data type with num_keypoint_count as capacity.
    //      You need to add missing parameters to code in comments.
    //   3. Use vxCreateDelay API to create delay objects for pyramid and
    //      keypoint array using the exemplars created using the two steps above.
    //      Use 2 delay slots for both of the delay objects.
    //      We gave code for one in comments; do similar for the other.
    //   4. Release the pyramid and keypoint array exemplar objects.
    //      We gave code for one in comments; do similar for the other.
    //   5. Use ERROR_CHECK_OBJECT/STATUS macros for proper error checking.
    //      We gave few error checks; do similar for the others.
    vx_pyramid pyramidExemplar = vxCreatePyramid( context, lk_pyramid_levels,
                                                  lk_pyramid_scale, width, height, VX_DF_IMAGE_U8 );
    ERROR_CHECK_OBJECT( pyramidExemplar );
    vx_delay pyramidDelay   = vxCreateDelay( context, ( vx_reference )pyramidExemplar, 2 );
    ERROR_CHECK_OBJECT( pyramidDelay );
    ERROR_CHECK_STATUS( vxReleasePyramid( &pyramidExemplar ) );
    vx_array keypointsExemplar = vxCreateArray( context, VX_TYPE_KEYPOINT, max_keypoint_count );
    ERROR_CHECK_OBJECT( keypointsExemplar );
    vx_delay keypointsDelay = vxCreateDelay( context, ( vx_reference )keypointsExemplar, 2 );
    ERROR_CHECK_STATUS( vxReleaseArray( &keypointsExemplar ) );


    ////////********
    // An object from a delay slot can be accessed using vxGetReferenceFromDelay API.
    // You need to use index = 0 for the current object and index = -1 for the previous object.
    //
    // TODO STEP 05:********
    //   1. Use vxGetReferenceFromDelay API to get the current and previous
    //      pyramid objects from pyramid delay object. Note that you need
    //      to typecast the vx_reference object to vx_pyramid.
    //      We gave code for one in comments; do similar for the other.
    //   2. Similarly, get the current and previous keypoint array objects from
    //      the keypoint delay object.
    //      We gave code for one in comments; do similar for the other.
    //   3. Use ERROR_CHECK_OBJECT for proper error checking.
    //      We gave one error check; do similar for the others.
    vx_pyramid currentPyramid  = ( vx_pyramid ) vxGetReferenceFromDelay( pyramidDelay, 0 );
    vx_pyramid previousPyramid = ( vx_pyramid ) vxGetReferenceFromDelay( pyramidDelay, -1 );
    vx_array currentKeypoints  = ( vx_array )   vxGetReferenceFromDelay( keypointsDelay, 0 );
    vx_array previousKeypoints = ( vx_array )   vxGetReferenceFromDelay( keypointsDelay, -1 );
    ERROR_CHECK_OBJECT( currentPyramid );
    ERROR_CHECK_OBJECT( previousPyramid );
    ERROR_CHECK_OBJECT( currentKeypoints );
    ERROR_CHECK_OBJECT( previousKeypoints );


    ////////********
    // Harris and optical flow algorithms require their own graph objects.
    // The Harris graph needs to extract gray scale image out of input RGB,
    // compute an initial set of keypoints, and compute an initial pyramid for use
    // by the optical flow graph.
    //
    // TODO STEP 06:********
    //   1. Create two graph objects: one for the Harris corner detector and
    //      the other for feature tracking using optical flow using the
    //      vxCreateGraph API.
    //      We gave code for one graph; do similar for the other.
    //   2. Use ERROR_CHECK_OBJECT to check the objects.
    //      We gave one error check; do similar for the other.
    vx_graph graphHarris = vxCreateGraph( context );
    vx_graph graphTrack  = vxCreateGraph( context );
    ERROR_CHECK_OBJECT( graphHarris );
    ERROR_CHECK_OBJECT( graphTrack );


    ////////********
    // Harris and pyramid computation expect input to be an 8-bit image.
    // Given that input is an RGB image, it is best to extract a gray image
    // from RGB image, which requires two steps:
    //   - perform RGB to IYUV color conversion
    //   - extract Y channel from IYUV image
    // This requires two intermediate OpenVX image objects. Since you don't
    // need to access these objects from the application, they can be virtual
    // objects that can be created using the vxCreateVirtualImage API.
    //
    // TODO STEP 07:********
    //   1. Create an IYUV image and a U8 image (for Y channel) with the same
    //      dimensions as the input RGB image. Note that the image formats for
    //      IYUV and U8 images are VX_DF_IMAGE_IYUV and VX_DF_IMAGE_U8.
    //      Note that virtual objects are specific to a graph, so you
    //      need to create two sets, one for each graph.
    //      We gave one fully in comments and you need to fill in missing
    //      parameters for the others.
    //   2. Use ERROR_CHECK_OBJECT to check the objects.
    //      We gave one error check in comments; do similar for others.
    vx_image harris_yuv_image       = vxCreateVirtualImage( graphHarris, width, height, VX_DF_IMAGE_IYUV );
    vx_image harris_gray_image      = vxCreateVirtualImage( graphHarris, width, height, VX_DF_IMAGE_U8 );
    vx_image opticalflow_yuv_image  = vxCreateVirtualImage( graphTrack,  width, height, VX_DF_IMAGE_IYUV );
    vx_image opticalflow_gray_image = vxCreateVirtualImage( graphTrack,  width, height, VX_DF_IMAGE_U8 );
    ERROR_CHECK_OBJECT( harris_yuv_image );
    ERROR_CHECK_OBJECT( harris_gray_image );
    ERROR_CHECK_OBJECT( opticalflow_yuv_image );
    ERROR_CHECK_OBJECT( opticalflow_gray_image );


    ////////********
    // The Harris corner detector and optical flow nodes (see "VX/vx_nodes.h")
    // take strength_thresh, min_distance, sensitivity, epsilon,
    // num_iterations, and use_initial_estimate parameters as scalar
    // data objects. So, you need to create scalar objects with the corresponding
    // configuration parameters.
    //
    // TODO STEP 08:********
    //   1. Create scalar data objects of VX_TYPE_FLOAT32 for strength_thresh,
    //      min_distance, sensitivity, and epsilon. Set their
    //      initial values to harris_strength_thresh, harris_min_distance,
    //      harris_sensitivity, and lk_epsilon.
    //      We gave code full code for one scalar in comments; fill in
    //      missing arguments for other ones.
    //   2. Similarly, create scalar objects for num_iterations and
    //      use_initial_estimate with initial values: lk_num_iterations and
    //      lk_use_initial_estimate. Make sure to use proper data types for
    //      these parameters.
    //      We gave code full code for one scalar in comments; fill in
    //      missing arguments for the other.
    //   3. Use ERROR_CHECK_OBJECT to check proper creation of objects.
    //      We gave the error check for one scalar; do similar for other 5 scalars.
    vx_scalar strength_thresh      = vxCreateScalar( context, VX_TYPE_FLOAT32, &harris_strength_thresh );
    vx_scalar min_distance         = vxCreateScalar( context, VX_TYPE_FLOAT32, &harris_min_distance );
    vx_scalar sensitivity          = vxCreateScalar( context, VX_TYPE_FLOAT32, &harris_sensitivity );
    vx_scalar epsilon              = vxCreateScalar( context, VX_TYPE_FLOAT32, &lk_epsilon );
    vx_scalar num_iterations       = vxCreateScalar( context, VX_TYPE_UINT32,  &lk_num_iterations );
    vx_scalar use_initial_estimate = vxCreateScalar( context, VX_TYPE_BOOL,    &lk_use_initial_estimate );
    ERROR_CHECK_OBJECT( strength_thresh );
    ERROR_CHECK_OBJECT( min_distance );
    ERROR_CHECK_OBJECT( sensitivity );
    ERROR_CHECK_OBJECT( epsilon );
    ERROR_CHECK_OBJECT( num_iterations );
    ERROR_CHECK_OBJECT( use_initial_estimate );


    ////////********
    // Now all the objects have been created for building the graphs.
    // First, build a graph that performs Harris corner detection and initial pyramid computation.
    // See "VX/vx_nodes.h" for APIs how to add nodes into a graph.
    //
    // TODO STEP 09:********
    //   1. Use vxColorConvertNode and vxChannelExtractNode APIs to get gray
    //      scale image for Harris and Pyramid computation from the input
    //      RGB image. Add these nodes into Harris graph.
    //      We gave code in comments with a missing parameter for you to fill in.
    //   2. Use vxGaussianPyramidNode API to add pyramid computation node.
    //      You need to use the current pyramid from the pyramid delay object.
    //      We gave code in comments with a missing parameter for you to fill in.
    //   3. Use vxHarrisCornersNode API to add a Harris corners node.
    //      You need to use the current keypoints from keypoints delay object.
    //      We gave code in comments with few missing parameters for you to fill in.
    //   4. Use ERROR_CHECK_OBJECT to check proper creation of objects.
    //   5. Release node and virtual objects immediately since the graph
    //      retains references to them.
    //   6. Call vxVerifyGraph to check for any errors in the graph.
    //      Fill in missing parameter in commented code.
    vx_node nodesHarris[] =
    {
        vxColorConvertNode( graphHarris, input_rgb_image, harris_yuv_image ),
        vxChannelExtractNode( graphHarris, harris_yuv_image, VX_CHANNEL_Y, harris_gray_image ),
        vxGaussianPyramidNode( graphHarris, harris_gray_image, currentPyramid ),
        vxHarrisCornersNode( graphHarris, harris_gray_image, strength_thresh, min_distance, sensitivity, harris_gradient_size, harris_block_size, currentKeypoints, NULL )
    };
    for( vx_size i = 0; i < sizeof( nodesHarris ) / sizeof( nodesHarris[0] ); i++ )
    {
        ERROR_CHECK_OBJECT( nodesHarris[i] );
        ERROR_CHECK_STATUS( vxReleaseNode( &nodesHarris[i] ) );
    }
    ERROR_CHECK_STATUS( vxReleaseImage( &harris_yuv_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &harris_gray_image ) );
    ERROR_CHECK_STATUS( vxVerifyGraph( graphHarris ) );


    ////////********
    // Now, build a graph that performs pyramid computation and feature
    // tracking using optical flow.
    //
    // TODO STEP 10:********
    //   1. Use vxColorConvertNode and vxChannelExtractNode APIs to get a gray
    //      scale image for Harris and Pyramid computation from the input
    //      RGB image. Add these nodes into Harris graph.
    //      We gave the code in comments for color convert node; do similar
    //      one for the channel extract node.
    //   2. Use vxGaussianPyramidNode API to add pyramid computation node.
    //      You need to use the current pyramid from the pyramid delay object.
    //      Most of the code is given in the comments; fill in the missing parameter.
    //   3. Use vxOpticalFlowPyrLKNode API to add an optical flow node. You need to
    //      use the current and previous keypoints from the keypoints delay object.
    //      Fill in the missing parameters in commented code.
    //   4. Use ERROR_CHECK_OBJECT to check proper creation of objects.
    //   5. Release node and virtual objects immediately since the graph
    //      retains references to them.
    //   6. Call vxVerifyGraph to check for any errors in the graph.
    //      Fill in the missing parameter in commented code.
    vx_node nodesTrack[] =
    {
        vxColorConvertNode( graphTrack, input_rgb_image, opticalflow_yuv_image ),
        vxChannelExtractNode( graphTrack, opticalflow_yuv_image, VX_CHANNEL_Y, opticalflow_gray_image ),
        vxGaussianPyramidNode( graphTrack, opticalflow_gray_image, currentPyramid ),
        vxOpticalFlowPyrLKNode( graphTrack, previousPyramid, currentPyramid,
                                            previousKeypoints, previousKeypoints, currentKeypoints,
                                            lk_termination, epsilon, num_iterations,
                                            use_initial_estimate, lk_window_dimension )
    };
    for( vx_size i = 0; i < sizeof( nodesTrack ) / sizeof( nodesTrack[0] ); i++ )
    {
        ERROR_CHECK_OBJECT( nodesTrack[i] );
        ERROR_CHECK_STATUS( vxReleaseNode( &nodesTrack[i] ) );
    }
    ERROR_CHECK_STATUS( vxReleaseImage( &opticalflow_yuv_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &opticalflow_gray_image ) );
    ERROR_CHECK_STATUS( vxVerifyGraph( graphTrack ) );


    ////////
    // Process the video sequence frame by frame until the end of sequence or aborted.
    for( int frame_index = 0; !gui.AbortRequested(); frame_index++ )
    {
        ////////********
        // Copy the input RGB frame from OpenCV to OpenVX.
        // In order to do this, you need to use vxAccessImagePatch and vxCommitImagePatch APIs.
        // See "VX/vx_api.h" for the description of these APIs.
        //
        // TODO STEP 11:********
        //   1. Specify the coordinates of image patch by declaring the patch
        //      as a vx_rectangle_t data type. It has four fields, we've given you the first one.
        //      See for the documentation what are the others. The start values should be zeros,
        //      end values should be width (for x) and height (for y).
        //   2. Specify the memory layout of the OpenCV RGB image buffer by
        //      declaring the layout as a vx_imagepatch_addressing_t type.
        //      Remember that you need to specify stride_x and stride_y fields
        //      of vx_imagepatch_addressing_t for the image buffer layout.
        //      The stride_x should be 3 and stride_y should be gui.GetStride().
        //      We've given you the stride_y, add the stride_x.
        //   3. Get the pointer to buffer using gui.GetBuffer() and call
        //      vxAccessImagePatch for VX_WRITE_ONLY usage mode with a pointer
        //      to pointer returned by gui.GetBuffer() so COPY mode is used.
        //      Then immediately call vxCommitImagePatch for the actual copy.
        //      Use the image patch and memory layout in the above two steps.
        //      We've given you the access function, please fill in the commit function.
        //      IMPORTANT: Note that vxCommitImagePatch takes 'void *' instead of 'void **'.
        //   4. Compare the return status with VX_SUCCESS to check if access/
        //      commit are successful. Or use the ERROR_CHECK_STATUS macro.
        vx_rectangle_t cv_rgb_image_region;
        cv_rgb_image_region.start_x    = 0;
        cv_rgb_image_region.start_y    = 0;
        cv_rgb_image_region.end_x      = width;
        cv_rgb_image_region.end_y      = height;
        vx_imagepatch_addressing_t cv_rgb_image_layout;
        cv_rgb_image_layout.stride_x   = 3;
        cv_rgb_image_layout.stride_y   = gui.GetStride();
        vx_uint8 * cv_rgb_image_buffer = gui.GetBuffer();
        ERROR_CHECK_STATUS( vxAccessImagePatch( input_rgb_image, &cv_rgb_image_region, 0,
                                                &cv_rgb_image_layout, ( void ** )&cv_rgb_image_buffer, VX_WRITE_ONLY ) );
        ERROR_CHECK_STATUS( vxCommitImagePatch( input_rgb_image, &cv_rgb_image_region, 0,
                                                &cv_rgb_image_layout, cv_rgb_image_buffer ) );


        ////////********
        // Now that input RGB image is ready, just run a graph.
        // Run Harris at the beginning to initialize the previous keypoints.
        //
        // TODO STEP 12:********
        //   1. Run a graph using vxProcessGraph API. Select Harris graph
        //      if the frame_index == 0 (i.e., the first frame of the video
        //      sequence), otherwise, select the feature tracking graph.
        //   2. Use ERROR_CHECK_STATUS for error checking.
        ERROR_CHECK_STATUS( vxProcessGraph( frame_index == 0 ? graphHarris : graphTrack ) );


        ////////********
        // To mark the keypoints in display, you need to access the output
        // keypoint array and draw each item on the output window using gui.DrawArrow().
        //
        // TODO STEP 13:********
        //   1. Use vxGetReferenceFromDelay API to get the current and previous
        //      keypoints array objects from the keypoints delay object.
        //      Make sure to typecast the vx_reference object to vx_array.
        //      We gave one for the previous previous keypoint array in comments;
        //      do a similar one for the current keypoint array.
        //   2. OpenVX array object has an attribute that keeps the current
        //      number of items in the array. The name of the attribute is
        //      VX_ARRAY_ATTRIBUTE_NUMITEMS and its value is of type vx_size.
        //      Use vxQueryArray API to get number of keypoints in the
        //      current keypoint array data object, representing number of
        //      corners detected in the input RGB image.
        //      IMPORTANT: Read number of items into "num_corners"
        //      because this variable is displayed by code segment below.
        //      We gave most part of this statement in comment; just fill in the
        //      missing parameter.
        //   3. The data items in output keypoint array are of type
        //      vx_keypoint_t (see "VX/vx_types.h"). To access the array
        //      buffer, use vxAccessArrayRange with start index = 0,
        //      end index = number of items in the array, and usage mode =
        //      VX_READ_ONLY. Note that the stride returned by this access
        //      call is not guaranteed to be sizeof(vx_keypoint_t).
        //      Also make sure that num_corners is > 0, because
        //      vxAccessArrayRange expects end index > 0.
        //      We gave the code for previous keypoint array in comment;
        //      do similar one for the current keypoint array.
        //   4. For each item in the keypoint buffer, use vxArrayItem to
        //      access an individual keypoint and draw a marker at (x,y)
        //      using gui.DrawArrow() if tracking_status field of keypoint
        //      is non-zero. Also count number of keypoints with
        //      non-zero tracking_status into "num_tracking" variable.
        //      We gave most of the code; fill in the missing parameters and uncomment.
        //   5. Hand the control of output keypoint buffer over back to
        //      OpenVX framework by calling vxCommitArrayRange API.
        //      We gave the code for previous keypoint array in comment;
        //      do similar one for the current keypoint array.
        //   6. Use ERROR_CHECK_STATUS for error checking.
        vx_size num_corners = 0, num_tracking = 0;
        previousKeypoints = ( vx_array )vxGetReferenceFromDelay( keypointsDelay, -1 );
        currentKeypoints  = ( vx_array )vxGetReferenceFromDelay( keypointsDelay, 0 );
        ERROR_CHECK_OBJECT( currentKeypoints );
        ERROR_CHECK_OBJECT( previousKeypoints );
        ERROR_CHECK_STATUS( vxQueryArray( previousKeypoints, VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_corners, sizeof( num_corners ) ) );
        if( num_corners > 0 )
        {
            vx_size kp_old_stride, kp_new_stride;
            vx_uint8 * kp_old_buf = NULL, * kp_new_buf = NULL;
            ERROR_CHECK_STATUS( vxAccessArrayRange( previousKeypoints, 0, num_corners,
                                                    &kp_old_stride, ( void ** ) &kp_old_buf, VX_READ_ONLY ) );
            ERROR_CHECK_STATUS( vxAccessArrayRange( currentKeypoints, 0, num_corners,
                                                    &kp_new_stride, ( void ** ) &kp_new_buf, VX_READ_ONLY ) );
            for( vx_size i = 0; i < num_corners; i++ )
            {
                vx_keypoint_t * kp_old = (vx_keypoint_t *) ( kp_old_buf + i * kp_old_stride );
                vx_keypoint_t * kp_new = (vx_keypoint_t *) ( kp_new_buf + i * kp_new_stride );
                if( kp_new->tracking_status )
                {
                    num_tracking++;
                    gui.DrawArrow( kp_old->x, kp_old->y, kp_new->x, kp_new->y );
                }
            }
            ERROR_CHECK_STATUS( vxCommitArrayRange( previousKeypoints, 0, num_corners, kp_old_buf ) );
            ERROR_CHECK_STATUS( vxCommitArrayRange( currentKeypoints, 0, num_corners, kp_new_buf ) );
        }


        ////////********
        // Flip the current and previous pyramid and keypoints in the delay objects.
        //
        // TODO STEP 14:********
        //   1. Use vxAgeDelay API to flip the current and previous buffers in delay objects.
        //      You need to call vxAgeDelay for both two delay objects.
        //   2. Use ERROR_CHECK_STATUS for error checking.
        ERROR_CHECK_STATUS( vxAgeDelay( pyramidDelay ) );
        ERROR_CHECK_STATUS( vxAgeDelay( keypointsDelay ) );


        ////////
        // Display the results and grab the next input RGB frame for the next iteration.
        char text[128];
        sprintf( text, "Keyboard ESC/Q-Quit SPACE-Pause [FRAME %d]", frame_index );
        gui.DrawText( 0, 16, text );
        sprintf( text, "Number of Corners: %d [tracking %d]", ( int )num_corners, ( int )num_tracking );
        gui.DrawText( 0, 36, text );
        gui.Show();
        if( !gui.Grab() )
        {
            // Terminate the processing loop if the end of sequence is detected.
            gui.WaitForKey();
            break;
        }
    }

    ////////********
    // Query graph performance using VX_GRAPH_ATTRIBUTE_PERFORMANCE and print timing
    // in milliseconds. Note that time units of vx_perf_t fields are nanoseconds.
    //
    // TODO STEP 15:********
    //   1. Use vxQueryGraph API with VX_GRAPH_ATTRIBUTE_PERFORMANCE to query graph performance.
    //      We gave the attribute query for one graph in comments. Do the same for the second graph.
    //   2. Print the average and min execution times in milliseconds. Use the printf in comments.
    vx_perf_t perfHarris = { 0 }, perfTrack = { 0 };
    ERROR_CHECK_STATUS( vxQueryGraph( graphHarris, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perfHarris, sizeof( perfHarris ) ) );
    ERROR_CHECK_STATUS( vxQueryGraph( graphTrack, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perfTrack, sizeof( perfTrack ) ) );
    printf( "GraphName NumFrames Avg(ms) Min(ms)\n"
            "Harris    %9d %7.3f %7.3f\n"
            "Track     %9d %7.3f %7.3f\n",
            ( int )perfHarris.num, ( float )perfHarris.avg * 1e-6f, ( float )perfHarris.min * 1e-6f,
            ( int )perfTrack.num,  ( float )perfTrack.avg  * 1e-6f, ( float )perfTrack.min  * 1e-6f );


    ////////********
    // Release all the OpenVX objects created in this exercise, and make the context as the last one to release.
    // To release an OpenVX object, you need to call vxRelease<Object> API which takes a pointer to the object.
    // If the release operation is successful, the OpenVX framework will reset the object to NULL.
    //
    // TODO STEP 16:********
    //   1. For releasing all other objects use vxRelease<Object> APIs.
    //      You have to release 2 graph objects, 1 image object, 2 delay objects,
    //      6 scalar objects, and 1 context object.
    //   2. Use ERROR_CHECK_STATUS for error checking.
    ERROR_CHECK_STATUS( vxReleaseGraph( &graphHarris ) );
    ERROR_CHECK_STATUS( vxReleaseGraph( &graphTrack ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &input_rgb_image ) );
    ERROR_CHECK_STATUS( vxReleaseDelay( &pyramidDelay ) );
    ERROR_CHECK_STATUS( vxReleaseDelay( &keypointsDelay ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &strength_thresh ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &min_distance ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &sensitivity ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &epsilon ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &num_iterations ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &use_initial_estimate ) );
    ERROR_CHECK_STATUS( vxReleaseContext( &context ) );


    return 0;
}
