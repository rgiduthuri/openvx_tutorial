/*
 * Copyright (c) 2019 Stephen Ramm
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
 * \file    trackin_example.c
 * \example centroid_tracking
 * \brief   Demontration application for centroid tracking
 * \author  Stephen Ramm <stephen.v.ramm@gmail.com>
 */

#define CV_AA 16
#define NUM_KEYPOINTS 1000
#define VIDEO_FILE "PETS09-S1-L1-View001.avi"
#define START_X 700     // These points define a rectangle
#define START_Y 225     // centred on the head of the guy in white
#define END_X 720
#define END_Y 250
#include "centroid_tracking.h"
#include <opencv2/opencv.hpp>
#include <stdio.h>
#include <string>

/*
Demonstration app for the tracking example
Building requires something like:
g++ tracking_example.cpp centroid_tracking.c -lopenvx -lm -I ~/openvx/api-docs/include/ -I /usr/local/include/opencv4 -lopencv_core -lopencv_videoio -lopencv_imgproc -lopencv_highgui

*/ 
void VX_CALLBACK log_callback( vx_context    context,
                               vx_reference  ref,
                               vx_status     status,
                               const vx_char string[] )
{
    printf( "LOG: [ status = %d ] %s\n", status, string );
    fflush( stdout );
}

vx_graph initial_feature_detection_graph(
    vx_context context, 
    vx_rectangle_t *bounding_box, 
    vx_image initial_image,
    vx_pyramid initial_pyramid,
    vx_array output_data, 
    vx_array output_corners,
    vx_array original_corners,
    vx_scalar valid)
{
    /* 
    Create the graph that performs the initial feature detection, given the bounding box.
    This graph will search for features using  FAST corners, in a region of interest of the
    input image bounded by the given rectangle
    */
    vx_graph graph = vxCreateGraph(context);
    vx_image yuv_image = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_IYUV);
    ERROR_CHECK_OBJECT(yuv_image)
    vx_image y_image = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(y_image)
    vx_image roi = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT);
    ERROR_CHECK_OBJECT(roi)
    vx_array bounds = vxCreateArray(context, VX_TYPE_RECTANGLE, 1);
    ERROR_CHECK_OBJECT(bounds)
    ERROR_CHECK_STATUS(vxAddArrayItems(bounds, 1, bounding_box, sizeof(*bounding_box)))
    vx_float32 fast_corners_strength = 3.0;
    vx_scalar strength_thresh = vxCreateScalar(context, VX_TYPE_FLOAT32, &fast_corners_strength);
    ERROR_CHECK_OBJECT(strength_thresh)
    vx_array first_corners = vxCreateVirtualArray(graph, VX_TYPE_KEYPOINT, NUM_KEYPOINTS);
    ERROR_CHECK_OBJECT(first_corners)
    ERROR_CHECK_OBJECT(vxColorConvertNode(graph, initial_image, yuv_image))
    ERROR_CHECK_OBJECT(vxChannelExtractNode(graph, yuv_image, VX_CHANNEL_Y, y_image))
    ERROR_CHECK_OBJECT(clearOutsideBoundsNode(graph, y_image, bounds, roi))
    ERROR_CHECK_OBJECT(vxGaussianPyramidNode(graph, y_image, initial_pyramid))
    ERROR_CHECK_OBJECT(vxFastCornersNode(graph, roi, strength_thresh, vx_true_e, first_corners, NULL))
    ERROR_CHECK_OBJECT(intialCentroidCalculationNode(graph, bounds, first_corners, output_data, output_corners, valid))
    ERROR_CHECK_OBJECT(vxCopyNode(graph, (vx_reference)output_corners, (vx_reference)original_corners))
    ERROR_CHECK_STATUS(vxReleaseImage(&yuv_image))
    ERROR_CHECK_STATUS(vxReleaseImage(&y_image))
    ERROR_CHECK_STATUS(vxReleaseImage(&roi))
    ERROR_CHECK_STATUS(vxReleaseArray(&bounds))
    ERROR_CHECK_STATUS(vxReleaseScalar(&strength_thresh))
    ERROR_CHECK_STATUS(vxReleaseArray(&first_corners))
    return graph;
}

vx_graph centroid_tracking_graph(
    vx_context context, vx_image input_image, vx_array original_corners,
    vx_delay images, vx_delay tracking_data, vx_delay corners,
    vx_scalar valid)
{
    /*
    Extract intensity from the RGB input
    Create pyramid
    Do optical flow
    Do centroid tracking clculation
    */
    vx_enum    lk_termination          = VX_TERM_CRITERIA_BOTH; // iteration termination criteria (eps & iterations)
    vx_float32 lk_epsilon              = 0.01f;                 // convergence criterion
    vx_uint32  lk_num_iterations       = 5;                     // maximum number of iterations
    vx_bool    lk_use_initial_estimate = vx_false_e;            // don't use initial estimate
    vx_uint32  lk_window_dimension     = 6;                     // window size for evaluation
    vx_scalar epsilon              = vxCreateScalar( context, VX_TYPE_FLOAT32, &lk_epsilon );
    vx_scalar num_iterations       = vxCreateScalar( context, VX_TYPE_UINT32,  &lk_num_iterations );
    vx_scalar use_initial_estimate = vxCreateScalar( context, VX_TYPE_BOOL,    &lk_use_initial_estimate );
    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph)
    vxAddLogEntry((vx_reference)context, VX_FAILURE, "About to create virtual objects");
    vx_array unfiltered_keypoints = vxCreateVirtualArray(graph, VX_TYPE_KEYPOINT, NUM_KEYPOINTS);
    ERROR_CHECK_OBJECT(unfiltered_keypoints)
    vx_image yuv_image = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_IYUV);
    ERROR_CHECK_OBJECT(yuv_image)
    vx_image y_image = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(y_image)
    vxAddLogEntry((vx_reference)context, VX_FAILURE, "About to get pyramids from delay");
    vx_pyramid current_pyramid = (vx_pyramid)vxGetReferenceFromDelay(images, 0);
    ERROR_CHECK_OBJECT(current_pyramid)
    vx_pyramid previous_pyramid = (vx_pyramid)vxGetReferenceFromDelay(images, -1);
    ERROR_CHECK_OBJECT(previous_pyramid)
    vxAddLogEntry((vx_reference)context, VX_FAILURE, "About to get corners from delay");
    vx_array current_corners = (vx_array)vxGetReferenceFromDelay(corners, 0);
    ERROR_CHECK_OBJECT(current_corners)
    vx_array previous_corners = (vx_array)vxGetReferenceFromDelay(corners, -1);
    ERROR_CHECK_OBJECT(previous_corners)
    vxAddLogEntry((vx_reference)context, VX_FAILURE, "About to get data from delay");
    vx_array current_data = (vx_array)vxGetReferenceFromDelay(tracking_data, 0);
    ERROR_CHECK_OBJECT(current_data)
    vx_array previous_data = (vx_array)vxGetReferenceFromDelay(tracking_data, -1);
    ERROR_CHECK_OBJECT(previous_data)
    vx_enum array_type;
    ERROR_CHECK_STATUS(vxQueryArray(current_data, VX_ARRAY_ITEMTYPE, &array_type, sizeof(array_type)))

    ERROR_CHECK_OBJECT(vxColorConvertNode(graph, input_image, yuv_image))
    ERROR_CHECK_OBJECT(vxChannelExtractNode(graph, yuv_image, VX_CHANNEL_Y, y_image))
    ERROR_CHECK_OBJECT(vxGaussianPyramidNode(graph, y_image, current_pyramid))
    ERROR_CHECK_OBJECT(vxOpticalFlowPyrLKNode(graph, previous_pyramid, current_pyramid, 
                                            previous_corners, previous_corners, unfiltered_keypoints,
                                            lk_termination, epsilon, num_iterations,
                                            use_initial_estimate, lk_window_dimension ))
    vxAddLogEntry((vx_reference)context, VX_FAILURE, "About to insert user node");
    printf("tracking_data_delay[0] holds items type %d\n", array_type);
    ERROR_CHECK_STATUS(vxQueryArray(previous_data, VX_ARRAY_ITEMTYPE, &array_type, sizeof(array_type)))
    printf("tracking_data_delay[-1] holds items type %d\n", array_type);
    ERROR_CHECK_STATUS(vxQueryArray(current_corners, VX_ARRAY_ITEMTYPE, &array_type, sizeof(array_type)))
    printf("corners_delay[0] holds items type %d\n", array_type);
    ERROR_CHECK_STATUS(vxQueryArray(previous_corners, VX_ARRAY_ITEMTYPE, &array_type, sizeof(array_type)))
    printf("corners_delay[-1] holds items type %d\n", array_type);
    ERROR_CHECK_OBJECT(trackCentroidsNode(graph, original_corners, previous_data, unfiltered_keypoints, current_data, current_corners, valid))
    vxAddLogEntry((vx_reference)context, VX_FAILURE, "about to release some objects");
    ERROR_CHECK_STATUS(vxReleaseScalar(&epsilon))
    ERROR_CHECK_STATUS(vxReleaseScalar(&num_iterations))
    ERROR_CHECK_STATUS(vxReleaseArray(&unfiltered_keypoints))
    ERROR_CHECK_STATUS(vxReleaseImage(&yuv_image))
    ERROR_CHECK_STATUS(vxReleaseImage(&y_image))
    return graph;
}

void copy_cv_to_vx(cv::Mat input, vx_image output)
{
       vx_rectangle_t cv_rgb_image_region;
        cv_rgb_image_region.start_x    = 0;
        cv_rgb_image_region.start_y    = 0;
        cv_rgb_image_region.end_x      = input.cols;
        cv_rgb_image_region.end_y      = input.rows;

        vx_imagepatch_addressing_t cv_rgb_image_layout;
        cv_rgb_image_layout.stride_x   = 3;
        cv_rgb_image_layout.stride_y   = input.step;

        vx_uint8 * cv_rgb_image_buffer = input.data;
        ERROR_CHECK_STATUS( vxCopyImagePatch( output, &cv_rgb_image_region, 0,
                                              &cv_rgb_image_layout, cv_rgb_image_buffer,
                                              VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST ) )
}

void draw_rectangle_on_buffer(const vx_rectangle_t *rect, cv::Mat buffer)
{
    cv::Scalar  color   = cv::Scalar( 0, 255, 255 );
    cv::line( buffer, cv::Point( rect->start_x, rect->start_y ), cv::Point( rect->end_x, rect->start_y ),color, 1 );
    cv::line( buffer, cv::Point( rect->start_x, rect->end_y ), cv::Point( rect->end_x, rect->end_y ),color, 1 );
    cv::line( buffer, cv::Point( rect->start_x, rect->start_y ), cv::Point( rect->start_x, rect->end_y ),color, 1 );
    cv::line( buffer, cv::Point( rect->end_x, rect->start_y ), cv::Point( rect->end_x, rect->end_y ),color, 1 );
}

void draw_point_on_buffer( int x, int y, cv::Mat buffer )
{
    cv::Point  center( x, y );
    cv::circle( buffer, center, 1, cv::Scalar( 0, 0, 255 ) );
}

void draw_keypoints_on_buffer(vx_array keypoints, cv::Mat buffer)
{
    char *point_array;
    vx_size num_points, stride;
    vx_map_id map_id;
    ERROR_CHECK_STATUS(vxQueryArray(keypoints, VX_ARRAY_NUMITEMS, &num_points, sizeof(num_points)))
    ERROR_CHECK_STATUS(vxMapArrayRange(keypoints, 0, num_points, &map_id, &stride, (void**)&point_array, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0 ))
    for (;num_points; --num_points, point_array += stride)
    {
        vx_keypoint_t *keypoint = (vx_keypoint_t*)point_array;
        if (keypoint->tracking_status)
            draw_point_on_buffer(keypoint->x, keypoint->y, buffer);
    }
    ERROR_CHECK_STATUS(vxUnmapArrayRange(keypoints, map_id))
}

int main(int argc, const char ** argv)
{
    /* OpenCV things ------------------------------------------------------------------------------------- */
    std::string video_sequence(VIDEO_FILE);
    cv::VideoCapture m_cap(video_sequence);
    cv::Mat m_imgBGR;
    cv::Mat m_imgRGB;
    if( !m_cap.isOpened())
    {
        printf( "ERROR: unable to open: \"%s\"\n", video_sequence.c_str() );
        exit( 1 );
    }
    vx_uint32 width = m_cap.get( cv::CAP_PROP_FRAME_WIDTH );
    vx_uint32 height = m_cap.get( cv::CAP_PROP_FRAME_HEIGHT );

    printf( "OK: FILE %s %dx%d\n", video_sequence.c_str(), width, height );
    cv::namedWindow(video_sequence);
    m_cap >> m_imgBGR;
    if( m_imgBGR.empty() )
    {
        printf( "ERROR: input has no video\n" );
        return 1;
    }
    cv::cvtColor( m_imgBGR, m_imgRGB, cv::COLOR_BGR2RGB );
    /* --------------------------------------------------------------------------------------------------- */
    /* OpenVX things ------------------------------------------------------------------------------------- */
    vx_context context = vxCreateContext();
//    vxRegisterLogCallback( context, log_callback, vx_false_e );
    vx_uint32  lk_pyramid_levels       = 6;                     // number of pyramid levels for optical flow
    vx_float32 lk_pyramid_scale        = VX_SCALE_PYRAMID_HALF; // pyramid levels scale by factor of two
    ERROR_CHECK_OBJECT(context)
    ERROR_CHECK_STATUS(registerCentroidNodes(context))
    vx_image frame = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    ERROR_CHECK_OBJECT(frame)
    vx_rectangle_t bounding_box;
    userTrackingData tracking_data;
    vx_array tracking_data_exemplar = vxCreateArray(context, user_struct_userTrackingData, 1);
    ERROR_CHECK_OBJECT(tracking_data_exemplar)
    vx_delay tracking_data_delay = vxCreateDelay(context, (vx_reference)tracking_data_exemplar, 2);
    ERROR_CHECK_OBJECT(tracking_data_delay)
    ERROR_CHECK_STATUS(vxReleaseArray(&tracking_data_exemplar))
    vx_array corners = vxCreateArray(context, VX_TYPE_KEYPOINT, NUM_KEYPOINTS);
    ERROR_CHECK_OBJECT(corners);
    vx_array crumbs = vxCreateArray(context, VX_TYPE_KEYPOINT, NUM_KEYPOINTS);
    ERROR_CHECK_OBJECT(crumbs);
    vx_delay corners_delay = vxCreateDelay(context, (vx_reference)corners, 2);
    ERROR_CHECK_OBJECT(corners_delay)
    vx_pyramid pyramid_exemplar = vxCreatePyramid(context, lk_pyramid_levels, lk_pyramid_scale, width, height, VX_DF_IMAGE_U8);
    ERROR_CHECK_OBJECT(pyramid_exemplar)
    vx_delay pyramid_delay = vxCreateDelay(context, (vx_reference)pyramid_exemplar, 2);
    ERROR_CHECK_OBJECT(pyramid_delay)
    ERROR_CHECK_STATUS(vxReleasePyramid(&pyramid_exemplar))
    vx_bool valid = vx_true_e;
    vx_scalar valid_scalar = vxCreateScalar(context, VX_TYPE_BOOL, &valid);
    ERROR_CHECK_OBJECT(valid_scalar);
    /* Here we set the bounding box. Ideally this would be done as the result of some object detection on the first frame */
    bounding_box.start_y = START_Y;
    bounding_box.end_y = END_Y;
    bounding_box.start_x = START_X;
    bounding_box.end_x = END_X;
    /* create & verify the graphs */
    vx_graph initial_graph = initial_feature_detection_graph(context, &bounding_box, frame, (vx_pyramid)vxGetReferenceFromDelay(pyramid_delay, 0),
        (vx_array)vxGetReferenceFromDelay(tracking_data_delay, 0), (vx_array)vxGetReferenceFromDelay(corners_delay, 0), corners, valid_scalar);
    ERROR_CHECK_STATUS(vxVerifyGraph(initial_graph))
	vxAddLogEntry((vx_reference)context, VX_FAILURE, "Verified first graph");
    vx_graph tracking_graph = centroid_tracking_graph(context, frame, corners, pyramid_delay, tracking_data_delay, corners_delay, valid_scalar);
	vxAddLogEntry((vx_reference)context, VX_FAILURE, "Created second graph");
    ERROR_CHECK_STATUS(vxVerifyGraph(tracking_graph))
	vxAddLogEntry((vx_reference)context, VX_FAILURE, "Verified second graph");
    copy_cv_to_vx(m_imgRGB, frame);
    /* Here we get the initial data for the tracking by running our initial feature detection graph */
    ERROR_CHECK_STATUS(vxProcessGraph(initial_graph))
	vxAddLogEntry((vx_reference)context, VX_FAILURE, "Processed first graph");
    char key = 0;
    for( int frame_index = 0; key != 'q' && key != 'Q' && key != 27; frame_index++ )
    {
        char text[128];
        vx_keypoint_t centroid;
        sprintf( text, "Keyboard ESC/Q-Quit SPACE-Pause [FRAME %d]", frame_index );
        cv::putText( m_imgBGR, text, cv::Point( 0, 16 ),
                     cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar( 128, 0, 0 ), 1, CV_AA );
        /* Get the current data from the delay */
        ERROR_CHECK_STATUS(vxCopyArrayRange((vx_array)vxGetReferenceFromDelay(tracking_data_delay, 0), 0, 1, sizeof(tracking_data), &tracking_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST))
        /* Draw the bounding box on the OpenCV image and display the frame */
        //draw_keypoints_on_buffer((vx_array)vxGetReferenceFromDelay(corners_delay, 0), m_imgBGR);
        draw_rectangle_on_buffer(&tracking_data.bounding_box, m_imgBGR);
        /* Store the centroid of the current bouding box in the breadcrumb trail and draw it */
        centroid.x = (tracking_data.bounding_box.start_x + tracking_data.bounding_box.end_x) / 2;
        centroid.y = (tracking_data.bounding_box.start_y + tracking_data.bounding_box.end_y) / 2;
        centroid.tracking_status = 1;
        vxAddArrayItems(crumbs, 1, &centroid, sizeof(centroid));
        draw_keypoints_on_buffer(crumbs, m_imgBGR);
        /* Se/e if tracking is still valid */
        ERROR_CHECK_STATUS(vxCopyScalar(valid_scalar, &valid, VX_READ_ONLY, VX_MEMORY_TYPE_HOST))
        if (! valid)
        {
            sprintf( text, "LOST TRACKING!" );
            cv::putText( m_imgBGR, text, cv::Point( 0, 32 ),
                        cv::FONT_HERSHEY_COMPLEX_SMALL, 0.8, cv::Scalar( 128, 0, 0 ), 1, CV_AA );
        }
        cv::imshow( video_sequence, m_imgBGR );
        m_cap >> m_imgBGR;
        if( m_imgBGR.empty() )
        {
            // Terminate the processing loop if the end of sequence is detected.
            cv::waitKey(0);
            break;
        }
        else
        {
            if (valid)
            {
                cv::cvtColor( m_imgBGR, m_imgRGB, cv::COLOR_BGR2RGB );
                /* Age the delays, get the next frame and run our tracking graph on it */
                ERROR_CHECK_STATUS(vxAgeDelay(tracking_data_delay))
                ERROR_CHECK_STATUS(vxAgeDelay(pyramid_delay))
                ERROR_CHECK_STATUS(vxAgeDelay(corners_delay))
                copy_cv_to_vx(m_imgRGB, frame);
                ERROR_CHECK_STATUS(vxProcessGraph(tracking_graph))
            }
            key = cv::waitKey( 1 );
            if ( ' '==key )
            {
                key = cv::waitKey(0);
            }
        }
        
    }
    // This will release the context and all the resources in it
    ERROR_CHECK_STATUS(vxReleaseContext(&context));
}
