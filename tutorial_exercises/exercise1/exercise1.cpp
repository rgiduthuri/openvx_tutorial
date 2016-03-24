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
 * \file    exercise1.cpp
 * \example exercise1
 * \brief   Harris corners example.
 *          Look for TODO STEP keyword in comments for the code snippets that you need to write.
 * \author  Radhakrishna Giduthuri <radha.giduthuri@ieee.org>
 *          Kari Pulli             <kari.pulli@gmail.com>
 */

////////
// Include OpenCV wrapper for image capture and display.
#include "opencv_camera_display.h"

////////********
// The most important top-level OpenVX header files are "VX/vx.h" and "VX/vxu.h".
// The "VX/vx.h" includes all headers needed to support functionality of the
// OpenVX specification, except for immediate mode functions, and it includes:
//    VX/vx_types.h     -- type definitions required by the OpenVX standard
//    VX/vx_api.h       -- all framework API definitions
//    VX/vx_kernels.h   -- list of supported kernels in the OpenVX standard
//    VX/vx_nodes.h     -- easier-to-use functions for the kernels
//    VX/vx_vendors.h
// The "VX/vxu.h" defines immediate mode utility functions.
//
// TODO STEP 01:********
//   1. Uncomment the lines below to include OpenVX header files.
#include <VX/vx.h>
//#include <VX/vxu.h>

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

////////********
// log_callback function implements a mechanism to print log messages
// from OpenVX framework onto console.
//
// TODO STEP 03b (see 03a below):********
//   1. Find the function signature for the log_callback.
//      First, go to definition of vxRegisterLogCallback.
//      Find the type of its second argument (it matches vx_log_*_f).
//      Follow that type into its definition.
//      You'll see the function takes in four arguments, add them here.
//   2. Uncomment the body of the log callback function.
//        LOG: [<status>] <message>
//   3. Move to STEP 03c.
//      Hint: use the Find functionality, CTRL-F or CMD-F, type 02c, hit ENTER.
void log_callback( /* add the function arguments and their types here */ )
{
//    printf( "LOG: [ status = %d ] %s\n", status, string );
//    fflush( stdout );
}

////////
// main() has all the OpenVX application code for this exercise.
// Command-line usage:
//   % solution_exercise1 [<video-sequence>|<camera-device-number>]
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
    //   harris_strength_thresh - minimum threshold score to keep a corner
    //                            (computed using the normalized Sobel kernel)
    //   harris_min_distance    - radial L2 distance for non-max suppression
    //   harris_k_sensitivity   - sensitivity threshold k from the Harris-Stephens
    //   harris_gradient_size   - window size for gradient computation
    //   harris_block_size      - block window size used to compute the
    //                            Harris corner score
    vx_uint32  width                  = gui.GetWidth();
    vx_uint32  height                 = gui.GetHeight();
    vx_float32 harris_strength_thresh = 0.0005f;
    vx_float32 harris_min_distance    = 5.0f;
    vx_float32 harris_k_sensitivity   = 0.04f;
    vx_int32   harris_gradient_size   = 3;
    vx_int32   harris_block_size      = 3;

    ////////********
    // Create the OpenVX context and make sure the returned context is valid.
    //
    // TODO STEP 02:********
    //   1. Create an OpenVX context.
    //      First, set cursor on vx_context type,
    //      and right-click -> Follow Symbol Under Cursor (or hit F2).
    //      Read the documentation. See the reference to function vxCreate...
    //      Start typing the function instead of the NULL in the context assignment below.
    //      Move cursor over the function name, hit F2 again to read its documentation.
    //      Finish creating the context.
    //
    //   2. Use ERROR_CHECK_OBJECT macro to check if context creation was successful.
    //      Start typing ERROR_CHECK_OBJECT, then with cursor on top, hit F2 to see the definition.
    //      Finish calling the error check by passing it context as an argument.
    vx_context context = NULL;

    ////////********
    // Register the log_callback that you implemented to be able to receive
    // any log messages from the OpenVX framework.
    //
    // TODO STEP 03a:********
    //   1. Uncomment the line below.
    //      Fill in the first argument for vxRegisterLogCallback to register a log callback function.
    //      See the documentation (use F2).
    //      The second argument is a function pointer to log_callback, move next to STEP 03b.
    //      Easy way to find that is to put cursor on top of log_callback and hit F2.
//  vxRegisterLogCallback( /* Fill in this argument */, log_callback, vx_false_e );

    // TODO STEP 03c:********
    //   1. Uncomment the line below, try that you get a log output
    //      (in the Application Output tab below in the IDE).
//  vxAddLogEntry( ( vx_reference ) context, VX_FAILURE, "Hello there!\n" );

    ////////********
    // Create OpenVX image object for input and OpenVX array object for output.
    //
    // TODO STEP 04:********
    //   1. See from vx_image type documentation with which function to create an image.
    //      Start typing the function name to replace NULL below, use autocomplete.
    //      Look into documentation (F2).
    //      The first argument should be obvious.
    //      For the second and third use local variables defined above, one of them is called width.
    //      For the last one use VX_DF_IMAGE_RGB.
    //   2. Use vxCreateArray API to create an array object with keypoint data type.
    //      See the documentation what should be the second argument (it's a type enum for keypoints).
    //   3. Use ERROR_CHECK_OBJECT to check proper creation of objects.
    //      We gave the one for the array, do a similar check for the image.
    vx_image input_rgb_image       = NULL;
    vx_array output_keypoint_array = NULL; //vxCreateArray( context, /* Fill this in */, 10000 );
//  ERROR_CHECK_OBJECT( output_keypoint_array );

    ////////********
    // The Harris corner detector algorithm expects input to be an 8-bit image.
    // Given that the input is an RGB image, it is best to extract a gray scale
    // from RGB image, which requires two steps:
    //   - perform RGB to IYUV color conversion
    //   - extract Y channel from IYUV image
    // This requires two intermediate OpenVX image objects. Since you're
    // going to use immediate mode functions, you need to use vxCreateImage
    // to create the image objects, but not the vxCreateVirtualImage API.
    //
    // TODO STEP 05:********
    //   1. Create an IYUV image and a U8 image (for Y channel) with the same
    //      dimensions as the input RGB image. The image formats for
    //      IYUV and U8 images are VX_DF_IMAGE_IYUV and VX_DF_IMAGE_U8.
    //   2. Use ERROR_CHECK_OBJECT to check that the objects are valid.
    vx_image yuv_image        = NULL;
    vx_image gray_scale_image = NULL;

    ////////********
    // The immediate mode Harris corner detector function takes the
    // strength_thresh, min_distance, and sensitivity parameters as scalar
    // data objects. So, you need to create three scalar objects with
    // corresponding configuration parameters.
    //
    // TODO STEP 06:********
    //   1. Create scalar data objects of VX_TYPE_FLOAT32 for strength_thresh,
    //      min_distance, and sensitivity, with initial values as harris_strength_thresh,
    //      harris_min_distance, and harris_k_sensitivity.
    //      The first one is given below.
    //   2. Use ERROR_CHECK_OBJECT to check the objects.
    vx_scalar strength_thresh = vxCreateScalar( context, VX_TYPE_FLOAT32, &harris_strength_thresh );
    vx_scalar min_distance    = NULL; // vxCreateScalar( /* Fill this in */ );
    vx_scalar sensitivity     = NULL; // vxCreateScalar( /* Fill this in */ );

    ////////
    // Process the video sequence frame by frame until the end of sequence or aborted.
    for( int frame_index = 0; !gui.AbortRequested(); frame_index++ )
    {
        ////////********
        // Copy the input RGB frame from OpenCV to OpenVX.
        // In order to do this, you need to use vxAccessImagePatch and vxCommitImagePatch APIs.
        // See "VX/vx_api.h" for the description of these APIs.
        //
        // TODO STEP 07:********
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
        //   4. Compare the return status with VX_SUCCESS to check if access/
        //      commit are successful. Or use the ERROR_CHECK_STATUS macro.
        vx_rectangle_t cv_rgb_image_region;
        cv_rgb_image_region.start_x = 0;
        vx_imagepatch_addressing_t cv_rgb_image_layout;
        cv_rgb_image_layout.stride_y   = gui.GetStride();
        vx_uint8 * cv_rgb_image_buffer = gui.GetBuffer();
//        ERROR_CHECK_STATUS( vxAccessImagePatch( input_rgb_image, &cv_rgb_image_region, 0,
//                                                &cv_rgb_image_layout, ( void ** )&cv_rgb_image_buffer,
//                                                VX_WRITE_ONLY ) );
//        ERROR_CHECK_STATUS( vxCommitImagePatch( /* Fill in the parameters */ ) );

        ////////********
        // In order to compute Harris corners from input RGB image, first you
        // need to convert the input RGB image into a gray scale image, followed by
        // running the Harris corner detector function. All the immediate mode
        // functions you need are declared in "VX/vxu.h".
        //
        // TODO STEP 08:********
        //   1. Convert the input RGB image to IYUV image using vxuColorConvert API.
        //   2. Extract Y channel from IYUV image into a gray scale image using
        //      vxuChannelExtract API with VX_CHANNEL_Y as the channel.
        //   3. Compute Harris corner detector using vxuHarrisCorners API.
        //      The num_corners parameter to vxuHarrisCorners is optional,
        //      you need to set it to NULL in this exercise.
        //   4. Use ERROR_CHECK_STATUS for error checking.
//        ERROR_CHECK_STATUS( vxuColorConvert( context, input_rgb_image, yuv_image ) );
//        ERROR_CHECK_STATUS( vxuChannelExtract( /* Fill in the parameters */ ) );
//        ERROR_CHECK_STATUS( vxuHarrisCorners( context, gray_scale_image, strength_thresh,
//                                              min_distance, sensitivity, harris_gradient_size,
//                                              harris_block_size, output_keypoint_array, NULL ) );

        ////////********
        // To mark the keypoints in display, you need to access the output
        // keypoint array and draw each item on the output window using
        // gui.DrawPoint().
        //
        // TODO STEP 09:********
        //   1. OpenVX array object has an attribute that stores the current
        //      number of items. The name of the attribute is
        //      VX_ARRAY_ATTRIBUTE_NUMITEMS and its value is of type vx_size.
        //      Use vxQueryArray API to get the number of keypoints in the
        //      output_keypoint_array data object, representing number of
        //      corners detected in the input RGB image.
        //      IMPORTANT: read the number of items into "num_corners"
        //      because this variable is displayed by code segment below.
        //   2. The data items in output keypoint array are of type
        //      vx_keypoint_t (see "VX/vx_types.h"). To access the array
        //      buffer, use vxAccessArrayRange with start index = 0,
        //      end index = number of items in the array, and usage mode =
        //      VX_READ_ONLY. Note that the stride returned by this access
        //      call is not guaranteed to be sizeof(vx_keypoint_t).
        //      Also make sure that num_corners is > 0, because
        //      vxAccessArrayRange expects an end index > 0.
        //      We've given you this code.
        //   3. For each item in the keypoint buffer, use vxArrayItem to
        //      access individual keypoint and draw a marker at (x,y)
        //      using gui.DrawPoint(). The vx_keypoint_t has x & y data fields.
        //   4. Handover the control of output keypoint buffer back to
        //      OpenVX framework by calling vxCommitArrayRange API.
        //   5. Use ERROR_CHECK_STATUS for error checking.
        vx_size num_corners = 0;
//        ERROR_CHECK_STATUS( vxQueryArray( output_keypoint_array,
//                                          VX_ARRAY_ATTRIBUTE_NUMITEMS,
//                                          &num_corners,
//                                          sizeof( num_corners ) ) );
        if( num_corners > 0 )
        {
            vx_size kp_stride;
            vx_keypoint_t * kp_buf = NULL;
            ERROR_CHECK_STATUS( vxAccessArrayRange( output_keypoint_array, 0, num_corners,
                                                    &kp_stride, ( void ** ) &kp_buf, VX_READ_ONLY ) );
            for( vx_size i = 0; i < num_corners; i++ )
            {
//                vx_keypoint_t * kp = /* Get the array item */
//                gui.DrawPoint( kp->x, kp->y );
            }
            ERROR_CHECK_STATUS( vxCommitArrayRange( output_keypoint_array, 0, num_corners, kp_buf ) );
        }

        ////////
        // Display the results and grab the next input RGB frame for the next iteration.
        char text[128];
        sprintf( text, "Keyboard ESC/Q-Quit SPACE-Pause [FRAME %d]", frame_index );
        gui.DrawText( 0, 16, text );
        sprintf( text, "Number of Corners: %d", ( int )num_corners );
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
    // Release all the OpenVX objects created in this exercise, and make the context as the last one to release.
    // To release an OpenVX object, you need to call vxRelease<Object> API which takes a pointer to the object.
    // If the release operation is successful, the OpenVX framework will reset the object to NULL.
    //
    // TODO STEP 10:********
    //   1. Release all the image objects using vxReleaseImage API.
    //   2. Release all other objects using vxRelease<Object> APIs.
    //   3. Use ERROR_CHECK_STATUS for error checking.
    // Release three images.
    // Release one array.
    // Release three Scalars.
//    ERROR_CHECK_STATUS( vxReleaseContext( &context ) );

    return 0;
}
