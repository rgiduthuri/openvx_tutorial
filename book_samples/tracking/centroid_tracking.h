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

#ifndef _CENTROID_TRACKING_H_
#define _CENTROID_TRACKING_H_

/*!
 * \file    centroid_tracking.h
 * \example centroid_tracking
 * \brief   Defines types and APIs for user nodes required for the centroid_tracking
 * example.
 * \author  Stephen Ramm <stephen.v.ramm@gmail.com>
 */

#ifdef  __cplusplus
extern "C" {
#endif

#include "VX/vx.h"
#include <stdio.h>
#include <stdlib.h>
#include "readImage.h"
#include "writeImage.h"

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

enum user_library_e
{
    USER_LIBRARY_EXAMPLE        = 1,
};

enum user_kernel_e
{
    USER_KERNEL_INITIAL_CENTROID_CALCULATION    = VX_KERNEL_BASE( VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE ) + 0x004,
    USER_KERNEL_TRACK_CENTROIDS                 = VX_KERNEL_BASE( VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE ) + 0x005,
    USER_KERNEL_CLEAR_OUTSIDE_BOUNDS            = VX_KERNEL_BASE( VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE ) + 0x006,
};

typedef struct {
	vx_rectangle_t		bounding_box;	// The last calculated object bounding box
	vx_coordinates2df_t	bb_centroid;	// The original bounding box centroid
	vx_coordinates2df_t bb_std_dev;		// The original bounding box standard deviation
	vx_coordinates2df_t	spread_ratio;	// The ratio of bounding box to features std.dev.
	vx_coordinates2df_t displacement;	// The displacement of bounding box from features
	vx_coordinates2df_t bb_vector;		// The rate of change in displacement of the bounding box
	vx_coordinates2df_t bb_zoom;		// The rate of change in scale of the bounding box
	vx_uint32			num_corners;	// The last valid number of features
} userTrackingData;

extern vx_enum user_struct_userTrackingData;

vx_node intialCentroidCalculationNode(
	/*
		Create a node to perform the initial centroid tracking calculations.
		The initial data consists of detected features and an object bounding box.
		The output scalar "valid" is a boolean that determines if tracking can continue.
		We use the otherwise unsused "scale" and "orientation" fields of the vx_keypoint_t
		struct to record the x and y coordinates of the feature position relative to the
		original centroid of the object.
	*/
	vx_graph 	graph,
	vx_array	bounding_box,	// Object coordinates (input)
	vx_array	corners,		// Detected features (input)
	vx_array	output_data,	// Holds a userTrackingData (output)
	vx_array	output_corners,	// Output parameter of filtered features, must be same capacity as input array
	vx_scalar	valid			// Holds a vx_bool
	);
	
vx_node trackCentroidsNode(
	/*
		Create a node to perform the centroid tracking update calculation
		We caculate a new position and size for the bounding box, and also the
		rate of change of deisplacement and size.
		We reject any features that are not behaving correctly, and recalculate the ratio and displacement.
		We signal with an output boolean if tracking can continue, or if new features need to be found
	*/
	vx_graph	graph,
	vx_array	originals, 		// Original features in case we need to recalculate
	vx_array	corners,		// Input features
	vx_array	input_data,		// Holds a userTrackingData (input)
	vx_array	output_data,	// Holds a userTrackingData (output)
	vx_array	output_corners,	// Filtered features (output), same capacity as input array
	vx_scalar	valid			// Holds a vx_bool
	);
	
vx_node clearOutsideBoundsNode(
	/*
		Create a node to clear all pixels outside a bounding box
	*/
	vx_graph	graph,
	vx_image	input_image,
	vx_array	bounds,	// Holds a vx_rect (input)
	vx_image	output_image
	);

vx_status registerCentroidNodes(vx_context context);
#ifdef  __cplusplus
}
#endif

#endif  /* _CENTROID_TRACKING_H_ */
