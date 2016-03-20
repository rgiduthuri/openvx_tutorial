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
 * \file    main.cpp
 * \example solution_exercise4
 * \brief   User kernel example.
 *          Look for TODO keyword in comments for the code snippets that you need to write.
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
// User kernel should have a unique enumeration and name for user kernel:
//   USER_LIBRARY_EXAMPLE      - library ID for user kernels in this example
//   USER_KERNEL_PICK_FEATURES - enumeration for "app.userkernels.pick_features" kernel
//
// TODO:********
//   1. Define USER_LIBRARY_EXAMPLE
//   2. Define USER_KERNEL_PICK_FEATURES using VX_KERNEL_BASE() macro
enum user_library_e
{
    USER_LIBRARY_EXAMPLE        = 1,
};
enum user_kernel_e
{
    USER_KERNEL_PICK_FEATURES   = VX_KERNEL_BASE( VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE ) + 0x001,
};

////////
// Constants used by the "pick_features" kernel:
//   PICK_FEATURE_THRESHOLD    - keypoint refresh threshold (i.e., min trackable/total keypoints ratio)
#define PICK_FEATURE_THRESHOLD   0.80f

////////
// The node creation interface for the "app.userkernels.pick_features" kernel.
// This user kernel example expects parameters in the following order:
//   parameter #0  --  input array            of type   VX_TYPE_KEYPOINT
//   parameter #1  --  input image            of format VX_DF_IMAGE_U8
//   parameter #2  --  scalar strength_thresh of type   VX_TYPE_FLOAT32 for Harris corners
//   parameter #3  --  scalar min_distance    of type   VX_TYPE_FLOAT32 for Harris corners
//   parameter #4  --  scalar k_sensitivity   of type   VX_TYPE_FLOAT32 for Harris corners
//   parameter #5  --  scalar gradient_size   of type   VX_TYPE_INT32   for Harris corners
//   parameter #6  --  scalar block_size      of type   VX_TYPE_INT32   for Harris corners
//   parameter #7  --  output array           of type   VX_TYPE_KEYPOINT
//
// TODO:********
//   1. Use vxGetKernelByEnum API to get a kernel object from USER_KERNEL_PICK_FEATURES.
//      Note that you need to use vxGetContext API to get the context from a graph object.
//   2. Use vxCreateGenericNode API to create a node from the kernel object.
//   3. Create scalar objects for gradient_size and block_size parameters.
//   4. Use vxSetParameterByIndex API to set node arguments.
//   5. Release the kernel and scalar objects that are not needed any more.
//   6. Use ERROR_CHECK_OBJECT and ERROR_CHECK_STATUS macros for error detection.
vx_node userPickFeaturesNode( vx_graph  graph,
                              vx_array  input_arr,
                              vx_image  input_image,
                              vx_scalar strength_thresh,
                              vx_scalar min_distance,
                              vx_scalar k_sensitivity,
                              vx_int32  gradient_size,
                              vx_int32  block_size,
                              vx_array  output_arr )
{
    vx_context context = vxGetContext( ( vx_reference ) graph );
    vx_kernel kernel   = vxGetKernelByEnum( context, USER_KERNEL_PICK_FEATURES );
    ERROR_CHECK_OBJECT( kernel );
    vx_node node       = vxCreateGenericNode( graph, kernel );
    ERROR_CHECK_OBJECT( node );

    vx_scalar s_gradient_size = vxCreateScalar( context, VX_TYPE_INT32, &gradient_size );
    vx_scalar s_block_size    = vxCreateScalar( context, VX_TYPE_INT32, &block_size );
    ERROR_CHECK_OBJECT( s_gradient_size );
    ERROR_CHECK_OBJECT( s_block_size );

    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 0, ( vx_reference ) input_arr ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 1, ( vx_reference ) input_image ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 2, ( vx_reference ) strength_thresh ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 3, ( vx_reference ) min_distance ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 4, ( vx_reference ) k_sensitivity ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 5, ( vx_reference ) s_gradient_size ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 6, ( vx_reference ) s_block_size ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 7, ( vx_reference ) output_arr ) );

    ERROR_CHECK_STATUS( vxReleaseScalar( &s_gradient_size ) );
    ERROR_CHECK_STATUS( vxReleaseScalar( &s_block_size ) );
    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );

    return node;
}

////////
// The user kernel input validator callback should check to make sure that all the input
// parameters have correct data types. This user kernel example expects the inputs
// to be valid in the following order:
//   parameter #0  --  input array            of type VX_TYPE_KEYPOINT
//   parameter #1  --  input image            of format VX_DF_IMAGE_U8
//   parameter #2  --  scalar strength_thresh of type VX_TYPE_FLOAT32
//   parameter #3  --  scalar min_distance    of type VX_TYPE_FLOAT32
//   parameter #4  --  scalar k_sensitivity   of type VX_TYPE_FLOAT32
//   parameter #5  --  scalar gradient_size   of type VX_TYPE_INT32
//   parameter #6  --  scalar block_size      of type VX_TYPE_INT32
//
// TODO:********
//   1. Use vxGetParameterByIndex API to get access to the requested parameter.
//   2. Use vxQueryParameter API with VX_PARAMETER_ATTRIBUTE_REF to access the input object.
//   3. If the index is 0, check to make sure that the array itemtype is VX_TYPE_KEYPOINT.
//   4. If the index is 1, check to make sure that image format is VX_DF_FORMAT_U8.
//   5. For index 2..4, check to make sure that the scalar type is VX_TYPE_FLOAT32.
//   6. For index 5..6, check to make sure that the scalar type is VX_TYPE_INT32.
vx_status VX_CALLBACK pick_features_input_validator( vx_node   node,
                                                     vx_uint32 index )
{
    vx_reference ref       = NULL;
    vx_parameter parameter = vxGetParameterByIndex( node, index );
    ERROR_CHECK_STATUS( vxQueryParameter( parameter, VX_PARAMETER_ATTRIBUTE_REF, &ref, sizeof( ref ) ) );
    ERROR_CHECK_STATUS( vxReleaseParameter( &parameter ) );
    ERROR_CHECK_OBJECT( ref );

    if( index == 0 )
    {
        // parameter #0 -- input array of type VX_TYPE_KEYPOINT
        vx_enum type = VX_TYPE_INVALID;
        ERROR_CHECK_STATUS( vxQueryArray( ( vx_array )ref, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type, sizeof( type ) ) );
        ERROR_CHECK_STATUS( vxReleaseArray( ( vx_array * )&ref ) );
        if( type != VX_TYPE_KEYPOINT )
        {
            return VX_ERROR_INVALID_TYPE;
        }
    }
    else if( index == 1 )
    {
        // parameter #1 -- input image of format VX_DF_IMAGE_U8
        vx_df_image format = VX_DF_IMAGE_VIRT;
        ERROR_CHECK_STATUS( vxQueryImage( ( vx_image )ref, VX_IMAGE_ATTRIBUTE_FORMAT, &format, sizeof( format ) ) );
        ERROR_CHECK_STATUS( vxReleaseImage( ( vx_image * )&ref ) );
        if( format != VX_DF_IMAGE_U8 )
        {
            return VX_ERROR_INVALID_FORMAT;
        }
    }
    else if( index <= 6 )
    {
        // parameters #2 .. #6 are scalars
        vx_enum type = VX_TYPE_INVALID;
        ERROR_CHECK_STATUS( vxQueryScalar( ( vx_scalar )ref, VX_SCALAR_ATTRIBUTE_TYPE, &type, sizeof( type ) ) );
        ERROR_CHECK_STATUS( vxReleaseScalar( ( vx_scalar * )&ref ) );
        if( ( index >= 2 && index <= 4 && type != VX_TYPE_FLOAT32 ) ||
            ( index >= 5 && index <= 6 && type != VX_TYPE_INT32 ) )
        {
            return VX_ERROR_INVALID_TYPE;
        }
    }
    else
    {
        // invalid input parameter
        return VX_ERROR_INVALID_PARAMETERS;
    }

    return VX_SUCCESS;
}

////////
// User kernel output validator callback should set the output parameter meta data.
// This user kernel example has only one output parameter with the same dimensions as parameter #0.
//   parameter #7  --  output array of type VX_TYPE_KEYPOINT
//
// TODO:********
//   1. Get the input array capacity from parameter #0.
//   2. Set the output array meta data: itemtype as VX_TYPE_KEYPOINT and capacity the same as in the input array.
vx_status VX_CALLBACK pick_features_output_validator( vx_node        node,
                                                      vx_uint32      index,
                                                      vx_meta_format meta )
{
    if( index == 7 )
    {
        vx_size  capacity  = 0;
        vx_array input_arr = NULL;
        vx_parameter parameter = vxGetParameterByIndex( node, 0 );
        ERROR_CHECK_STATUS( vxQueryParameter( parameter, VX_PARAMETER_ATTRIBUTE_REF, &input_arr, sizeof( input_arr ) ) );
        ERROR_CHECK_STATUS( vxReleaseParameter( &parameter ) );
        ERROR_CHECK_OBJECT( input_arr );
        ERROR_CHECK_STATUS( vxQueryArray( input_arr, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof( capacity ) ) );
        ERROR_CHECK_STATUS( vxReleaseArray( &input_arr ) );

        vx_enum type = VX_TYPE_KEYPOINT;
        ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( meta, VX_ARRAY_ATTRIBUTE_ITEMTYPE, &type,     sizeof( type ) ) );
        ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( meta, VX_ARRAY_ATTRIBUTE_CAPACITY, &capacity, sizeof( capacity ) ) );
    }
    else
    {
        // invalid input parameter
        return VX_ERROR_INVALID_PARAMETERS;
    }

    return VX_SUCCESS;
}

////////
// The user kernel initialization function gets called after all the parameters have been validated.
// The pick_features kernel uses a graph to execute a graph with a Harris corners node.
// This initializer creates and initializes the graph and saves it as node's local data pointer.
//
// TODO:********
//   1. Use vxReadScalarValue API to get gradient_size and block_size values from the
//      corresponding scalar objects (i.e., refs[5] and refs[6]).
//   2. Build a new graph to perform Harris corner detection. Note that you need to get
//      access to context from the node object in order to create a graph object.
//   3. Use vxVerifyGraph API to avoid initialization during the first time processing
//      of Harris graph.
//   4. Store Harris graph in the node as node local data pointer.
vx_status VX_CALLBACK pick_features_initialize( vx_node              node,
                                                const vx_reference * refs,
                                                vx_uint32            num )
{
    vx_int32 gradient_size = 0, block_size = 0;
    ERROR_CHECK_STATUS( vxReadScalarValue( ( vx_scalar )refs[5], &gradient_size ) );
    ERROR_CHECK_STATUS( vxReadScalarValue( ( vx_scalar )refs[6], &block_size ) );

    vx_context context   = vxGetContext( ( vx_reference ) node );
    vx_graph graphHarris = vxCreateGraph( context );
    ERROR_CHECK_OBJECT( graphHarris );
    vx_node nodeHarris   = vxHarrisCornersNode( graphHarris, ( vx_image )refs[1],
                                                ( vx_scalar )refs[2], ( vx_scalar )refs[3], ( vx_scalar )refs[4],
                                                gradient_size, block_size, ( vx_array )refs[7], NULL );
    ERROR_CHECK_OBJECT( nodeHarris );
    ERROR_CHECK_STATUS( vxReleaseNode( &nodeHarris ) );

    ERROR_CHECK_STATUS( vxVerifyGraph( graphHarris ) );

    ERROR_CHECK_STATUS( vxSetNodeAttribute( node, VX_NODE_ATTRIBUTE_LOCAL_DATA_PTR, &graphHarris, sizeof( graphHarris ) ) );

    return VX_SUCCESS;
}

////////
// The user kernel deinitialization function gets called during node garbage collection.
// The pick_features kernel uses a graph to execute a graph with Harris corners node.
// This deinitializer has to release the graph object saved in node's local data pointer.
//
// TODO:********
//   1. Get the Harris graph from node local data pointer and release the graph.
vx_status VX_CALLBACK pick_features_deinitialize( vx_node              node,
                                                  const vx_reference * refs,
                                                  vx_uint32            num )
{
    vx_graph graphHarris = NULL;
    ERROR_CHECK_STATUS( vxQueryNode( node, VX_NODE_ATTRIBUTE_LOCAL_DATA_PTR, &graphHarris, sizeof( graphHarris ) ) );
    ERROR_CHECK_STATUS( vxReleaseGraph( &graphHarris ) );

    return VX_SUCCESS;
}

////////
// The user kernel host side function gets called to execute the user kernel node.
// The pick_features kernel needs to calculate the ratio of keypoints being tracked/total.
// If this ratio is less than PICK_FEATURE_THRESHOLD, then just run the Harris corners.
// Otherwise, just copy the input keypoints to output array.
//
// TODO:********
//   1. Compute the number of tracked features in the input keypoint array.
//   2. Copy the input keypoints into output array, if the number of keypoints in
//      input array is greater than ZERO and ratio of tracked features to
//      number of keypoints is greater than or equal to PICK_FEATURE_THRESHOLD.
//   3. If not copied in above step 2, i.e., tracked feature ratio is less than
//      PICK_FEATURE_THRESHOLD or number of keypoints in input array is ZERO,
//      just run the Harris graph to generate new keypoints.
//      Note that the Harris graph is stored as a node local data pointer.
vx_status VX_CALLBACK pick_features_host_side_function( vx_node              node,
                                                        const vx_reference * refs,
                                                        vx_uint32            num )
{
    vx_float32 tracked_feature_ratio = 0.0f;

    vx_array input_arr  = ( vx_array )refs[0];
    vx_array output_arr = ( vx_array )refs[7];

    vx_size kp_numitems = 0;
    ERROR_CHECK_STATUS( vxQueryArray( input_arr, VX_ARRAY_ATTRIBUTE_NUMITEMS, &kp_numitems, sizeof( kp_numitems ) ) );
    if( kp_numitems > 0 )
    {
        vx_keypoint_t * kp_src_base = NULL;
        vx_size kp_src_stride = 0;
        ERROR_CHECK_STATUS( vxAccessArrayRange( input_arr, 0, kp_numitems, &kp_src_stride,
                                                ( void ** ) &kp_src_base, VX_READ_ONLY ) );
        vx_size kp_numtracked = 0;
        for( vx_size i = 0; i < kp_numitems; ++i )
        {
            vx_keypoint_t * kp_src = &vxArrayItem( vx_keypoint_t, kp_src_base, i, kp_src_stride );
            if( kp_src->tracking_status )
            {
                kp_numtracked++;
            }
        }
        tracked_feature_ratio = ( vx_float32 )kp_numtracked / kp_numitems;
        if( tracked_feature_ratio >= PICK_FEATURE_THRESHOLD )
        {
            ERROR_CHECK_STATUS( vxTruncateArray( output_arr, 0 ) );
            ERROR_CHECK_STATUS( vxAddArrayItems( output_arr, kp_numitems, kp_src_base, kp_src_stride ) );
        }
        ERROR_CHECK_STATUS( vxCommitArrayRange( input_arr, 0, kp_numitems, kp_src_base ) );
    }

    if( tracked_feature_ratio < PICK_FEATURE_THRESHOLD )
    {
        vx_graph graphHarris = NULL;
        ERROR_CHECK_STATUS( vxQueryNode( node, VX_NODE_ATTRIBUTE_LOCAL_DATA_PTR, &graphHarris, sizeof( graphHarris ) ) );
        ERROR_CHECK_STATUS( vxProcessGraph( graphHarris ) );
    }

    return VX_SUCCESS;
}

////////
// User kernels needs to be registered with every OpenVX context before use in a graph.
//
// TODO:********
//   1. Use vxAddKernel API to register "app.userkernels.pick_features" with
//      kernel enumeration = USER_KERNEL_PICK_FEATURES, numParams = 8, and
//      all of the user kernel callback functions you implemented above.
//   2. Use vxAddParameterToKernel API to specify direction, data_type, and
//      state of all 8 parameters to the kernel. Look into the comments of
//      userPickFeaturesNode function (above) to details about the order of
//      kernel parameters and their types.
//   3. Use vxFinalizeKernel API to make the kernel ready to use in a graph.
//      Note that the kernel object is still valid after this call.
//      So you need to call vxReleaseKernel before returning from this function.
vx_status registerUserKernel( vx_context context )
{
    vx_kernel kernel = vxAddKernel( context,
                                    "app.userkernels.pick_features",
                                    USER_KERNEL_PICK_FEATURES,
                                    pick_features_host_side_function,
                                    8,   // numParams
                                    pick_features_input_validator,
                                    pick_features_output_validator,
                                    pick_features_initialize,
                                    pick_features_deinitialize );
    ERROR_CHECK_OBJECT( kernel );

    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 0, VX_INPUT,  VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED ) ); // input_kp
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 1, VX_INPUT,  VX_TYPE_IMAGE,  VX_PARAMETER_STATE_REQUIRED ) ); // input_image
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 2, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED ) ); // strength_thresh
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 3, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED ) ); // min_distance
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 4, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED ) ); // sensitivity
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 5, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED ) ); // gradient_size
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 6, VX_INPUT,  VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED ) ); // block_size
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 7, VX_OUTPUT, VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED ) ); // output_kp
    ERROR_CHECK_STATUS( vxFinalizeKernel( kernel ) );
    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );

    vxAddLogEntry( ( vx_reference ) context, VX_SUCCESS, "OK: registered user kernel app.userkernels.pick_features\n" );
    return VX_SUCCESS;
}

////////
// log_callback function implements a mechanism to print log messages
// from the OpenVX framework onto console.
void log_callback( vx_context    context,
                   vx_reference  ref,
                   vx_status     status,
                   const vx_char string[] )
{
    printf( "LOG: [ %3d ] %s\n", status, string );
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
    // Get default video sequence when nothing is specified on the command-line and
    // instantiate OpenCV GUI module for reading input RGB images and displaying
    // the image with the OpenVX results.
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
    //   harris_k_sensitivity    - sensitivity threshold k from the Harris-Stephens
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
    vx_uint32  width                   = gui.GetWidth();
    vx_uint32  height                  = gui.GetHeight();
    vx_size    max_keypoint_count      = 10000;
    vx_float32 harris_strength_thresh  = 0.0005f;
    vx_float32 harris_min_distance     = 5.0f;
    vx_float32 harris_k_sensitivity    = 0.04f;
    vx_int32   harris_gradient_size    = 3;
    vx_int32   harris_block_size       = 3;
    vx_uint32  lk_pyramid_levels       = 6;
    vx_float32 lk_pyramid_scale        = VX_SCALE_PYRAMID_HALF;
    vx_enum    lk_termination          = VX_TERM_CRITERIA_BOTH;
    vx_float32 lk_epsilon              = 0.01f;
    vx_uint32  lk_num_iterations       = 5;
    vx_bool    lk_use_initial_estimate = vx_false_e;
    vx_uint32  lk_window_dimension     = 6;

    ////////
    // Create the OpenVX context and make sure the returned context is valid and
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
    // Create OpenVX image object for input RGB image.
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
    vx_pyramid pyramidExemplar = vxCreatePyramid( context, lk_pyramid_levels,
                                                  lk_pyramid_scale, width, height, VX_DF_IMAGE_U8 );
    vx_array keypointsExemplar = vxCreateArray( context, VX_TYPE_KEYPOINT,
                                                max_keypoint_count );
    ERROR_CHECK_OBJECT( pyramidExemplar );
    ERROR_CHECK_OBJECT( keypointsExemplar );
    vx_delay pyramidDelay   = vxCreateDelay( context, ( vx_reference )pyramidExemplar, 2 );
    vx_delay keypointsDelay = vxCreateDelay( context, ( vx_reference )keypointsExemplar, 2 );
    ERROR_CHECK_OBJECT( pyramidDelay );
    ERROR_CHECK_OBJECT( keypointsDelay );
    ERROR_CHECK_STATUS( vxReleasePyramid( &pyramidExemplar ) );
    ERROR_CHECK_STATUS( vxReleaseArray( &keypointsExemplar ) );

    ////////********
    // An object from a delay slot can be accessed using vxGetReferenceFromDelay API.
    // You need to use index = 0 for the current object and index = -1 for the previous object.
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
    vx_image harris_yuv_image       = vxCreateVirtualImage( graphHarris, width, height, VX_DF_IMAGE_IYUV );
    vx_image harris_luma_image      = vxCreateVirtualImage( graphHarris, width, height, VX_DF_IMAGE_U8 );
    vx_image opticalflow_yuv_image  = vxCreateVirtualImage( graphTrack,  width, height, VX_DF_IMAGE_IYUV );
    vx_image opticalflow_luma_image = vxCreateVirtualImage( graphTrack,  width, height, VX_DF_IMAGE_U8 );
    ERROR_CHECK_OBJECT( harris_yuv_image );
    ERROR_CHECK_OBJECT( harris_luma_image );
    ERROR_CHECK_OBJECT( opticalflow_yuv_image );
    ERROR_CHECK_OBJECT( opticalflow_luma_image );

    ////////********
    // The Harris corner detector and optical flow nodes (see "VX/vx_nodes.h")
    // take strength_thresh, min_distance, sensitivity, epsilon,
    // num_iterations, and use_initial_estimate parameters as scalar
    // data objects. So, you need to create scalar objects with the corresponding
    // configuration parameters.
    vx_scalar strength_thresh      = vxCreateScalar( context, VX_TYPE_FLOAT32, &harris_strength_thresh );
    vx_scalar min_distance         = vxCreateScalar( context, VX_TYPE_FLOAT32, &harris_min_distance );
    vx_scalar sensitivity          = vxCreateScalar( context, VX_TYPE_FLOAT32, &harris_k_sensitivity );
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
    // The pick features user node requires an intermediate keypoint array,
    // which will then be passed to optical flow node. So, you need to create
    // a keypoint array objects with capacity as keypointDelay exemplar.
    //
    // TODO:********
    //   1. Create array data objects of VX_TYPE_KEYPOINT for feature points
    //      coming out of "pick_features" user node. Make sure that the
    //      array capacity is "max_keypoint_count".
    vx_array featureKeypoints = vxCreateArray( context, VX_TYPE_KEYPOINT, max_keypoint_count );
    ERROR_CHECK_OBJECT( featureKeypoints );

    ////////********
    // Now all the objects have been created for building the graphs.
    // First, build a graph that performs Harris corner detection and initial pyramid computation.
    // See "VX/vx_nodes.h" for APIs how to add nodes into a graph.
    vx_node nodesHarris[] =
    {
        vxColorConvertNode(    graphHarris, input_rgb_image, harris_yuv_image ),
        vxChannelExtractNode(  graphHarris, harris_yuv_image, VX_CHANNEL_Y, harris_luma_image ),
        vxGaussianPyramidNode( graphHarris, harris_luma_image, currentPyramid ),
        vxHarrisCornersNode(   graphHarris, harris_luma_image, strength_thresh, min_distance, sensitivity,
                               harris_gradient_size, harris_block_size, currentKeypoints, NULL )
    };
    for( vx_size i = 0; i < sizeof( nodesHarris ) / sizeof( nodesHarris[0] ); i++ )
    {
        ERROR_CHECK_OBJECT( nodesHarris[i] );
        ERROR_CHECK_STATUS( vxReleaseNode( &nodesHarris[i] ) );
    }
    ERROR_CHECK_STATUS( vxReleaseImage( &harris_yuv_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &harris_luma_image ) );
    ERROR_CHECK_STATUS( vxVerifyGraph( graphHarris ) );

    ////////********
    // Now, build a graph that performs pyramid computation and feature
    // tracking using optical flow and "pick_features" user node.
    // Note that you need to use "featureKeypoint" array as the output from
    // "pick_features" node and input to optical flow node.
    // Also note that "pick_features" expects Level 0 of "previousPyramid" as input.
    //
    // TODO:********
    //   1. Use vxGetPyramidLevel API to get Level of "previousPyramid".
    //   2. Use userPickFeaturesNode function to add "pick_features" node.
    vx_image previousPyramidLevel0 = vxGetPyramidLevel( previousPyramid, 0 );
    vx_node nodesTrack[] =
    {
        vxColorConvertNode( graphTrack, input_rgb_image, opticalflow_yuv_image ),
        vxChannelExtractNode( graphTrack, opticalflow_yuv_image, VX_CHANNEL_Y, opticalflow_luma_image ),
        vxGaussianPyramidNode( graphTrack, opticalflow_luma_image, currentPyramid ),
        userPickFeaturesNode( graphTrack, previousKeypoints, previousPyramidLevel0, strength_thresh,
                              min_distance, sensitivity, harris_gradient_size, harris_block_size, featureKeypoints ),
        vxOpticalFlowPyrLKNode( graphTrack, previousPyramid, currentPyramid, featureKeypoints, featureKeypoints,
                                currentKeypoints, lk_termination, epsilon, num_iterations, use_initial_estimate, lk_window_dimension )
    };
    for( vx_size i = 0; i < sizeof( nodesTrack ) / sizeof( nodesTrack[0] ); i++ )
    {
        ERROR_CHECK_OBJECT( nodesTrack[i] );
        ERROR_CHECK_STATUS( vxReleaseNode( &nodesTrack[i] ) );
    }
    ERROR_CHECK_STATUS( vxReleaseImage( &previousPyramidLevel0 ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &opticalflow_yuv_image ) );
    ERROR_CHECK_STATUS( vxReleaseImage( &opticalflow_luma_image ) );
    ERROR_CHECK_STATUS( vxVerifyGraph( graphTrack ) );

    ////////
    // Process the video sequence frame by frame until the end of sequence or aborted.
    for( int frame_index = 0; !gui.AbortRequested(); frame_index++ )
    {
        ////////
        // Copy the input RGB frame from OpenCV to OpenVX.
        // In order to do this, you need to use vxAccessImagePatch and vxCommitImagePatch APIs.
        // See "VX/vx_api.h" for the description of these APIs.
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
        ERROR_CHECK_STATUS( vxProcessGraph( frame_index == 0 ? graphHarris : graphTrack ) );

        ////////********
        // To mark the keypoints in display, you need to access the output
        // keypoint array and draw each item on the output window using gui.DrawArrow().
        vx_size num_corners = 0, num_tracking = 0;
        currentKeypoints = ( vx_array )vxGetReferenceFromDelay( keypointsDelay, 0 );
        ERROR_CHECK_OBJECT( currentKeypoints );
        ERROR_CHECK_STATUS( vxQueryArray( featureKeypoints,
                                          VX_ARRAY_ATTRIBUTE_NUMITEMS, &num_corners, sizeof( num_corners ) ) );
        if( num_corners > 0 )
        {
            vx_size kp_old_stride, kp_new_stride;
            vx_keypoint_t * kp_old_buf = NULL, * kp_new_buf = NULL;
            ERROR_CHECK_STATUS( vxAccessArrayRange( featureKeypoints, 0, num_corners,
                                         &kp_old_stride, ( void ** ) &kp_old_buf, VX_READ_ONLY ) );
            ERROR_CHECK_STATUS( vxAccessArrayRange( currentKeypoints, 0, num_corners,
                                         &kp_new_stride, ( void ** ) &kp_new_buf, VX_READ_ONLY ) );
            for( vx_size i = 0; i < num_corners; i++ )
            {
                vx_keypoint_t * kp_old = &vxArrayItem( vx_keypoint_t, kp_old_buf, i, kp_old_stride );
                vx_keypoint_t * kp_new = &vxArrayItem( vx_keypoint_t, kp_new_buf, i, kp_new_stride );
                if( kp_new->tracking_status )
                {
                    num_tracking++;
                    gui.DrawArrow( kp_old->x, kp_old->y, kp_new->x, kp_new->y );
                }
            }
            ERROR_CHECK_STATUS( vxCommitArrayRange( featureKeypoints, 0, num_corners, kp_old_buf ) );
            ERROR_CHECK_STATUS( vxCommitArrayRange( currentKeypoints, 0, num_corners, kp_new_buf ) );
        }

        ////////********
        // Flip the current and previous pyramid and keypoints in the delay objects.
        ERROR_CHECK_STATUS( vxAgeDelay( pyramidDelay ) );
        ERROR_CHECK_STATUS( vxAgeDelay( keypointsDelay ) );

        ////////
        // Display the results and grab the next input RGB frame for the next iteration.
        char text[128];
        sprintf( text, "Keyboard ESC/Q-Quit SPACE-Pause [FRAME %d]", frame_index );
        gui.DrawText( 0, 16, text );
        sprintf( text, "Number of Corners: %d [tracking %d %.1f%%]", ( int )num_corners, ( int )num_tracking,
                 num_corners ? ( 100.0f * num_tracking / num_corners ) : 0.0f );
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
    vx_perf_t perfHarris = { 0 }, perfTrack = { 0 };
    ERROR_CHECK_STATUS( vxQueryGraph( graphHarris, VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perfHarris, sizeof( perfHarris ) ) );
    ERROR_CHECK_STATUS( vxQueryGraph( graphTrack,  VX_GRAPH_ATTRIBUTE_PERFORMANCE, &perfTrack,  sizeof( perfTrack ) ) );
    printf( "GraphName NumFrames Avg(ms) Min(ms)\n"
            "Harris    %9d %7.3f %7.3f\n"
            "Track     %9d %7.3f %7.3f\n",
            ( int )perfHarris.num, ( float )perfHarris.avg * 1e-6f, ( float )perfHarris.min * 1e-6f,
            ( int )perfTrack.num,  ( float )perfTrack.avg  * 1e-6f, ( float )perfTrack.min  * 1e-6f );

    ////////********
    // Release all the OpenVX objects created in this exercise, and make the context as the last one to release.
    // To release an OpenVX object, you need to call vxRelease<Object> API which takes a pointer to the object.
    // If the release operation is successful, the OpenVX framework will reset the object to NULL.
    ERROR_CHECK_STATUS( vxReleaseGraph( &graphHarris ) );
    ERROR_CHECK_STATUS( vxReleaseGraph( &graphTrack ) );
    ERROR_CHECK_STATUS( vxReleaseArray( &featureKeypoints ) );
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
