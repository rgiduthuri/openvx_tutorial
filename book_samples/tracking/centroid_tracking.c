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
 * \file    centroid_tracking.c
 * \example centroid_tracking
 * \brief   Implementation of centroid tracking APIs
 * \author  Stephen Ramm <stephen.v.ramm@gmail.com>
 */

#include "centroid_tracking.h"
#include <math.h>
#include <string.h>

vx_enum user_struct_userTrackingData;

void print_coordinates2df(const char * tag, vx_coordinates2df_t *coords)
{
	printf("%s (%f, %f)\n", tag, coords->x, coords->y);
}

void print_rectangle(const char * tag, vx_rectangle_t *rect)
{
	printf("%s [(%d, %d), (%d, %d)]\n", tag, rect->start_x, rect->start_y, rect->end_x, rect->end_y);
}

void print_trackingdata(const char * tag, userTrackingData *td)
{
	printf("%s\n", tag);
	printf("Number of key points %d\n", td->num_corners);
	print_rectangle("Bounding box", &td->bounding_box);
	print_coordinates2df("Bounding box centroid", &td->bb_centroid);
	print_coordinates2df("Bounding box standard deviation", &td->bb_std_dev);
	print_coordinates2df("Bounding box vector", &td->bb_vector);
	print_coordinates2df("Bounding box zoom", &td->bb_zoom);
	print_coordinates2df("Displacement", &td->displacement);
	print_coordinates2df("Spread", &td->spread_ratio);
}

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
	)
{
    vx_context context = vxGetContext( ( vx_reference ) graph );
    vx_kernel kernel = vxGetKernelByEnum( context, USER_KERNEL_INITIAL_CENTROID_CALCULATION);
    ERROR_CHECK_OBJECT( kernel );
    vx_node node       = vxCreateGenericNode( graph, kernel );
    ERROR_CHECK_OBJECT( node );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 0, ( vx_reference ) bounding_box ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 1, ( vx_reference ) corners ) );
	ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 2, ( vx_reference ) output_data ) );
	ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 3, ( vx_reference ) output_corners ) );
	ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 4, ( vx_reference ) valid ) );
    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );
    return node;
}

vx_status VX_CALLBACK initialCentroidCalculation_validator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
	/* NEEDS WORK: Check type of references, allow all outputs to have unspecified metadata (virtuals) and if so, set */
	if (num != 5)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
    // parameter #0 -- check array type
    vx_enum param_type;
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[0], VX_ARRAY_ITEMTYPE, &param_type, sizeof(param_type)));
    if(param_type != VX_TYPE_RECTANGLE) // check that the scalar holds a rectangle type; also allow VX_UINT64...
    {
		vxAddLogEntry((vx_reference)node, VX_FAILURE, "Invalid array type for parameter 0");
        return VX_ERROR_INVALID_TYPE;
    }
	
    // parameter #1 -- check array type
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ITEMTYPE, &param_type, sizeof(param_type)));
    if(param_type != VX_TYPE_KEYPOINT) // check that the array holds vx_keypoint_t structures
    {
		vxAddLogEntry((vx_reference)node, VX_FAILURE, "Invalid array type for parameter 1");
        return VX_ERROR_INVALID_TYPE;
    }
	
	// parameter #2 -- check array type
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ITEMTYPE, &param_type, sizeof(param_type)));
    if(param_type != user_struct_userTrackingData) // check that the scalar holds a user type of the correct sort
    {
 		vxAddLogEntry((vx_reference)node, VX_FAILURE, "Invalid array type for parameter 2");
       	return VX_ERROR_INVALID_TYPE;
    }
	ERROR_CHECK_STATUS(vxSetMetaFormatFromReference(metas[2], parameters[2]));
	
    // set output metadata - output array should be the same size and type as the input array.
	ERROR_CHECK_STATUS(vxSetMetaFormatFromReference(metas[3], parameters[1]));
	// parameter #4 -- check scalar type of output, set if virtual
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[4], VX_SCALAR_TYPE, &param_type, sizeof(param_type)));

    if(param_type != VX_TYPE_BOOL) // check that the scalar holds a vx_bool
    {
    	vxAddLogEntry((vx_reference)node, VX_FAILURE, "Invalid scalar type for parameter 4");
        return VX_ERROR_INVALID_TYPE;
    }
	ERROR_CHECK_STATUS(vxSetMetaFormatFromReference(metas[4], parameters[4]));
	
	return VX_SUCCESS;
}

vx_status VX_CALLBACK initialCentroidCalculation_function(vx_node node, const vx_reference * refs, vx_uint32 num)
{
	/*
		Initial centroid calculation function.
		From the input bounding box (parameter 0) and the input corners (parameter 1) calculate the
		output userTrackingData, output corners and validity
	*/
	vx_array bounding_box = (vx_array)refs[0];
	vx_array corners = (vx_array)refs[1];
	vx_array output_data = (vx_array)refs[2];
	vx_array output_corners = (vx_array)refs[3];
	vx_scalar scalar_valid = (vx_scalar)refs[4];
	userTrackingData tracking_data;
	vx_bool valid = vx_true_e;
	vx_size num_corners = 0, orig_num_corners = 0;
	double sumx = 0.0, sumy = 0.0, sumsqx = 0.0, sumsqy = 0.0, meanx, meany, sigmax, sigmay;
	vx_map_id input_map_id;
	vx_size stride;
	char  *array_data = NULL;
	ERROR_CHECK_STATUS(vxTruncateArray(output_data, 0));
	ERROR_CHECK_STATUS(vxTruncateArray(output_corners, 0));
	// Initial value of the bounding box in the tracking data is the input bounding box
	ERROR_CHECK_STATUS( vxCopyArrayRange(bounding_box, 0, 1, sizeof(tracking_data.bounding_box), &tracking_data.bounding_box, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
	// Initial value of the bounding box centroid:
	tracking_data.bb_centroid.x = (tracking_data.bounding_box.end_x + tracking_data.bounding_box.start_x) / 2.0;
	tracking_data.bb_centroid.y = (tracking_data.bounding_box.end_y + tracking_data.bounding_box.start_y) / 2.0;
	// Initial value of the bounding box standard deviations
	tracking_data.bb_std_dev.x = (tracking_data.bounding_box.end_x - tracking_data.bounding_box.start_x) / 2.0;
	tracking_data.bb_std_dev.y = (tracking_data.bounding_box.end_y - tracking_data.bounding_box.start_y) / 2.0;
	// Check for validity - we must have a bounding box with some area:
	if (tracking_data.bb_std_dev.x < 1.0 || tracking_data.bb_std_dev.y < 1.0 )
	{
		valid = vx_false_e;
	}
	// Initial velocities are all set to zero
	tracking_data.bb_vector.x = 0.0;
	tracking_data.bb_vector.y = 0.0;
	tracking_data.bb_zoom.x = 0.0;
	tracking_data.bb_zoom.y = 0.0;
	// Find the the sums and sums of squares of the corners, rejecting any that do not lie within the bounding box
	// Copy good data to the output corner array
	ERROR_CHECK_STATUS(vxQueryArray(corners, VX_ARRAY_NUMITEMS, &orig_num_corners, sizeof(num_corners)));
	ERROR_CHECK_STATUS(vxMapArrayRange(corners, 0, orig_num_corners, &input_map_id, &stride, (void **)&array_data, VX_READ_ONLY, VX_MEMORY_TYPE_NONE, 0));
	for (vx_uint32 i = 0; i < orig_num_corners; ++i, array_data += stride)
	{
		vx_keypoint_t *feature = (vx_keypoint_t *)array_data;
		// a tracking_status of zero indicates a lost point
		// we will mark a point as lost if it is already outside the bounding box
		// or on the bounding margin, as may be the case initially
		if (feature->tracking_status != 0 &&
			feature->x > tracking_data.bounding_box.start_x &&
			feature->x < tracking_data.bounding_box.end_x &&
			feature->y > tracking_data.bounding_box.start_y &&
			feature->y < tracking_data.bounding_box.end_y)
		{
			sumx += feature->x;
			sumy += feature->y;
			sumsqx += feature->x * feature->x;
			sumsqy += feature->y * feature->y;
			vxAddArrayItems(output_corners, 1, array_data, stride);
			++num_corners;
		}
		else
		{
			feature->tracking_status = 0;
		}
	}
	ERROR_CHECK_STATUS(vxUnmapArrayRange(corners, input_map_id));
	// Now calculate centroid (mean) and standard deviation of the corners using the sums and sums of squares.
	// We can also assess validity during this operation
	// From the means and standard deviations we calculate the spread ratio and normalised displacement
	printf("Initial number of corners: %ld\n", num_corners);
	if (num_corners < 2)
	{
		// need at least two features to be able to establish a centroid and spread
		valid = vx_false_e;
		tracking_data.spread_ratio.x = 1.0;
		tracking_data.spread_ratio.y = 1.0;
		tracking_data.displacement.x = 0.0;
		tracking_data.displacement.y = 0.0;
	}
	else
	{
		meanx = sumx / num_corners;
		meany = sumy / num_corners;
		sigmax = sqrt( sumsqx / num_corners - meanx * meanx);
		sigmay = sqrt( sumsqy / num_corners - meany * meany);
		if (sigmax < 1.0 || sigmay < 1.0)
		{
			// Need some area in the features to be able to calculate a new bounding box
			valid = vx_false_e;
			sigmax = 1.0; // just to stop divide-by-zero errors
			sigmay = 1.0;
		}
		tracking_data.spread_ratio.x = tracking_data.bb_std_dev.x / sigmax;
		tracking_data.spread_ratio.y = tracking_data.bb_std_dev.y / sigmay;
		tracking_data.displacement.x = (tracking_data.bb_centroid.x - meanx) / sigmax;
		tracking_data.displacement.y = (tracking_data.bb_centroid.y - meany) / sigmay;
	}
		
	tracking_data.num_corners = num_corners;
	// set output "output_data" parameter
	ERROR_CHECK_STATUS(vxAddArrayItems(output_data, 1, &tracking_data, sizeof(tracking_data)))
	// Set output "valid" parameter
	ERROR_CHECK_STATUS(vxCopyScalar(scalar_valid, &valid, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	print_trackingdata("initial data", &tracking_data);
	return VX_SUCCESS;
}

vx_status registerInitialCentroidCalculation_kernel(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context,
                                       "app.userkernels.initial_centroid_calculation",
                                       USER_KERNEL_INITIAL_CENTROID_CALCULATION,
                                       initialCentroidCalculation_function,
                                       5,   // numParams
                                       initialCentroidCalculation_validator,
                                       NULL,
                                       NULL);
    ERROR_CHECK_OBJECT(kernel);

    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT,  VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED)); // input - the initial bounding box
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT,  VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED)); // input - the initial array of key points
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED)); // output- tracking data
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED)); // output- filtered array of key points
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // output- validity
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    vxAddLogEntry((vx_reference)context, VX_FAILURE, "OK: registered user kernel app.userkernels.initial_centroid_calculation\n");
    return VX_SUCCESS;
}
vx_node trackCentroidsNode(
	/*
		Create a node to perform the centroid tracking update calculation
		We caculate a new position and size for the bounding box, and also the
		rate of change of deisplacement and size.
		We reject any features that are not behaving correctly, and recalculate the ratio and displacement.
		We signal with an output boolean if tracking can continue, or if new features need to be found
	*/
	vx_graph	graph,
	vx_array	originals, 		// Holds the original features in case we need to recalculate
	vx_array	input_data,		// Holds a userTrackingData (input)
	vx_array	corners,		// Input features
	vx_array	output_data,	// Holds a userTrackingData (output)
	vx_array	output_corners,	// Filtered features (output)
	vx_scalar	valid			// Holds a vx_bool
	)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_TRACK_CENTROIDS);
    ERROR_CHECK_OBJECT(kernel);
    vx_node node = vxCreateGenericNode(graph, kernel);
    ERROR_CHECK_OBJECT(node);
    ERROR_CHECK_OBJECT(corners);
	ERROR_CHECK_OBJECT(originals);
    ERROR_CHECK_OBJECT(input_data);
    ERROR_CHECK_OBJECT(output_data);
    ERROR_CHECK_OBJECT(output_corners);
    ERROR_CHECK_OBJECT(valid);
	ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference)originals));
	ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference)input_data));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 2, (vx_reference)corners));
	ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 3, (vx_reference)output_data));
	ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 4, (vx_reference)output_corners));
	ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 5, (vx_reference)valid));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return node;
}
vx_status VX_CALLBACK trackCentroids_validator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
	/* NEEDS WORK: Check type of references, allow all outputs to have unspecified metadata (virtuals) and if so, set */
	if (num != 6)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
    vx_enum param_type;

	vxAddLogEntry((vx_reference)node, VX_FAILURE, "checking parameter 0");
	// parameter #0 -- check array type
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[0], VX_ARRAY_ITEMTYPE, &param_type, sizeof(param_type)));
    if(param_type != VX_TYPE_KEYPOINT) // check that the array holds keypoints
    {
        return VX_ERROR_INVALID_TYPE;
    }

	vxAddLogEntry((vx_reference)node, VX_FAILURE, "checking parameter 1");
	// parameter #1 -- check array type

    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ITEMTYPE, &param_type, sizeof(param_type)));
    if(param_type != user_struct_userTrackingData) // check that the array holds user type of correct sort
    {
		printf("Parameter 0 is of wrong type %d, should be %d\n", param_type, user_struct_userTrackingData);
        return VX_ERROR_INVALID_TYPE;
    }

	vxAddLogEntry((vx_reference)node, VX_FAILURE, "checking parameter 2");
	// parameter #2 -- check array type
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[2], VX_ARRAY_ITEMTYPE, &param_type, sizeof(param_type)));
    if(param_type != VX_TYPE_KEYPOINT) // check that the array holds keypoints
    {
        return VX_ERROR_INVALID_TYPE;
    }

	vxAddLogEntry((vx_reference)node, VX_FAILURE, "about to process output parameters");
	vxAddLogEntry((vx_reference)node, VX_FAILURE, "checking parameter 3");
    // parameter #3 -- check array type of output
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[3], VX_ARRAY_ITEMTYPE, &param_type, sizeof(param_type)));
    if(param_type != user_struct_userTrackingData) // check that the array holds a user type of the correct sort
    {
        return VX_ERROR_INVALID_TYPE;
    }
	ERROR_CHECK_STATUS(vxSetMetaFormatFromReference(metas[3], parameters[1]))
	
	vxAddLogEntry((vx_reference)node, VX_FAILURE, "checking parameter 4");
    // parameter #4 -- check array type & set if unspecified
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[4], VX_ARRAY_ITEMTYPE, &param_type, sizeof(param_type)));
    if(param_type != VX_TYPE_KEYPOINT && param_type != 0) // check that the array holds vx_keypoint_t structures or is virtual
    {
        return VX_ERROR_INVALID_TYPE;
    }
    // set output metadata - output array should be the same size and type as the input array.
	ERROR_CHECK_STATUS(vxSetMetaFormatFromReference(metas[4], parameters[2]));

	// parameter #5 -- check scalar type of output, set if virtual
    ERROR_CHECK_STATUS(vxQueryScalar((vx_scalar)parameters[5], VX_SCALAR_TYPE, &param_type, sizeof(param_type)));
    if(param_type != VX_TYPE_BOOL && param_type != 0) // check that the scalar holds a vx_bool
    {
        return VX_ERROR_INVALID_TYPE;
    }
	vxAddLogEntry((vx_reference)node, VX_FAILURE, "checking parameter 5");
	param_type = VX_TYPE_BOOL;
    ERROR_CHECK_STATUS(vxSetMetaFormatFromReference(metas[5], parameters[5]))
	ERROR_CHECK_STATUS(vxSetMetaFormatAttribute( metas[5], VX_SCALAR_TYPE, &param_type, sizeof(param_type)));

    return VX_SUCCESS;
}

static vx_size calculate_new_bounding_box(vx_size orig_num_corners, userTrackingData *tracking_data, vx_keypoint_t *old_features, vx_keypoint_t *features, vx_size stride)
{
	/* This function calculates a new bounding box in tracking_data, and returns the number of corners used */
	// Find the the sums and sums of squares of the corners
	double sumx = 0.0, sumy = 0.0, sumsqx = 0.0, sumsqy = 0.0, meanx, meany, sigmax, sigmay;
	double osumx = 0.0, osumy = 0.0, osumsqx = 0.0, osumsqy = 0.0;
	vx_size num_corners = 0;
	vx_char * ptr = (char *)features, * old_ptr = (char *)old_features;
	for (vx_uint32 i = 0; i < orig_num_corners; ++i, ptr+=stride, old_ptr += stride)
	{
		vx_keypoint_t * feature = (vx_keypoint_t*)ptr;
		vx_keypoint_t * old_feature = (vx_keypoint_t*) old_ptr;
		// a tracking_status of zero indicates a lost point
		// and we don't process old points
		if (feature->tracking_status != 0)
		{
			sumx += feature->x;
			sumy += feature->y;
			sumsqx += feature->x * feature->x;
			sumsqy += feature->y * feature->y;
			osumx += old_feature->x;
			osumy += old_feature->y;
			osumsqx += old_feature->x * old_feature->x;
			osumsqy += old_feature->y * old_feature->y;
			++num_corners;
		}
	}
	// Now calculate centroid (mean) and standard deviation of the corners using the sums and sums of squares.
	// We can also assess validity during this operation. Notice that during tracking we consider tracking of
	// the object lost if any feature is lost - it may be possible to recover
	// From the means and standard deviations we calculate the new bounding box estimate
	if (num_corners >= 2)
	{
		// need at least two features to be able to establish a centroid and spread
		if ( tracking_data->num_corners != num_corners)
		{
			printf("Number of corners was %d, is now %ld\n", tracking_data->num_corners, num_corners);
			meanx = osumx / num_corners;
			meany = osumy / num_corners;
			sigmax = sqrt( osumsqx / num_corners - meanx * meanx);
			sigmay = sqrt( osumsqy / num_corners - meany * meany);
			printf("mean (stddev) = [ %f (%f), %f (%f) ]\n", meanx, sigmax, meany, sigmay);
			if (sigmax < 1.0 || sigmay < 1.0)
			{
				// Need some area in the features to be able to calculate a new bounding box
				sigmax = 1.0; // just to stop divide-by-zero errors
				sigmay = 1.0;
			}
			tracking_data->spread_ratio.x = tracking_data->bb_std_dev.x / sigmax;
			tracking_data->spread_ratio.y = tracking_data->bb_std_dev.y / sigmay;
			tracking_data->displacement.x = (tracking_data->bb_centroid.x - meanx) / sigmax;
			tracking_data->displacement.y = (tracking_data->bb_centroid.y - meany) / sigmay;
		}

		meanx = sumx / num_corners;
		meany = sumy / num_corners;
		sigmax = sqrt( sumsqx / num_corners - meanx * meanx);
		sigmay = sqrt( sumsqy / num_corners - meany * meany);
		if (sigmax < 1.0 || sigmay < 1.0)
		{
			// Need some area in the features to be able to calculate a new bounding box
			sigmax = 1.0; // just to stop divide-by-zero errors
			sigmay = 1.0;
		}
		// Calculate new bounding box
		vx_rectangle_t bb;

		bb.start_x = meanx + (tracking_data->displacement.x - tracking_data->spread_ratio.x) * sigmax;
		bb.end_x = meanx + (tracking_data->displacement.x + tracking_data->spread_ratio.x) * sigmax + 0.5;
		bb.start_y = meany + (tracking_data->displacement.y - tracking_data->spread_ratio.y) * sigmay;
		bb.end_y = meany + (tracking_data->displacement.y + tracking_data->spread_ratio.y) * sigmay + 0.5;
		// Calculate bounding box velocities using simple differences, apply some smoothing using a simple IIR filter
		double bbmeanxdelta = (bb.start_x + bb.end_x - tracking_data->bounding_box.start_x - tracking_data->bounding_box.end_x) / 2.0;
		double bbmeanydelta = (bb.start_y + bb.end_y - tracking_data->bounding_box.start_y - tracking_data->bounding_box.end_y) / 2.0;
		double bbsizexdelta = (bb.end_x - bb.start_x - tracking_data->bounding_box.end_x + tracking_data->bounding_box.start_x) / 2.0;
		double bbsizeydelta = (bb.end_y - bb.start_y - tracking_data->bounding_box.end_y + tracking_data->bounding_box.start_y) / 2.0;
		tracking_data->bb_vector.x = tracking_data->bb_vector.x * 0.25 + bbmeanxdelta * 0.75;
		tracking_data->bb_vector.y = tracking_data->bb_vector.y * 0.25 + bbmeanydelta * 0.75;
		tracking_data->bb_zoom.x = tracking_data->bb_zoom.x * 0.25 + bbsizexdelta * 0.75;
		tracking_data->bb_zoom.y = tracking_data->bb_zoom.y * 0.25 + bbsizeydelta * 0.75;
		// set new bounding box
		tracking_data->bounding_box.start_x = bb.start_x;
		tracking_data->bounding_box.end_x = bb.end_x;
		tracking_data->bounding_box.start_y = bb.start_y;
		tracking_data->bounding_box.end_y = bb.end_y;
	}
	else
	{
		printf("Less than 2 corners!\n");
	}
	
	tracking_data->num_corners = num_corners;
	return num_corners;
}

static int validate_corners(vx_size orig_num_corners, userTrackingData *tracking_data, vx_keypoint_t *features, vx_size stride)
{
/* This function validates the corners, setting them to non-valid if they are outside the bounding box
   it returns the number of corners that were newly invalidated
   */
	int rejected_corners = 0;
	char *ptr = (char *)features;
	for (vx_uint32 i = 0; i < orig_num_corners; ++i, ptr+=stride)
	{
		vx_keypoint_t *feature = (vx_keypoint_t*)ptr;
		// a tracking_status of zero indicates a lost point
		// we will mark a point as lost if it is outside the bounding box
		if (feature->tracking_status != 0 && (
			feature->x < tracking_data->bounding_box.start_x - 1||
			feature->x > tracking_data->bounding_box.end_x +1||
			feature->y < tracking_data->bounding_box.start_y -1||
			feature->y > tracking_data->bounding_box.end_y +1))
		{
			++rejected_corners;
			feature->tracking_status = 0;
		}
	}
	return rejected_corners;
}

vx_status VX_CALLBACK trackCentroids_function(vx_node node, const vx_reference * refs, vx_uint32 num)
{
	/*
		track centroids function.
		From the input tracking data (parameter 0) and the input corners (parameter 1) calculate the
		output userTrackingData (parameter 2), output corners (parameter 3) and validity
		This is done in an iterative process:
		do
			Calculate a new bounding box from the key points
			reject any key points outside the new bounding box
		until no key points were rejected
	*/
	vx_array originals = (vx_array)refs[0];
	vx_array input_data = (vx_array)refs[1];
	vx_array corners = (vx_array)refs[2];
	vx_array output_data = (vx_array)refs[3];
	vx_array output_corners = (vx_array)refs[4];
	vx_scalar valid = (vx_scalar)refs[5];

	userTrackingData tracking_data;
	vx_bool b_valid = vx_true_e;
	vx_size orig_num_corners = 0, corners_used, corners_validated;
	double sumx = 0.0, sumy = 0.0, sumsqx = 0.0, sumsqy = 0.0, meanx, meany, sigmax, sigmay;
	vx_map_id input_map_id, original_map_id;
	vx_size stride;
	vx_keypoint_t *array_data = NULL;
	vx_keypoint_t *old_array_data = NULL;
	// empty the output arrays
	ERROR_CHECK_STATUS(vxTruncateArray(output_corners, 0))
	ERROR_CHECK_STATUS(vxTruncateArray(output_data, 0))
	// Get the input tracking data and map the input corners
	ERROR_CHECK_STATUS(vxCopyArrayRange(input_data, 0, 1, sizeof(tracking_data), &tracking_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
	ERROR_CHECK_STATUS(vxQueryArray(corners, VX_ARRAY_NUMITEMS, &orig_num_corners, sizeof(orig_num_corners)));
	ERROR_CHECK_STATUS(vxMapArrayRange(originals, 0, orig_num_corners, &original_map_id, &stride, (void **)&old_array_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
	ERROR_CHECK_STATUS(vxMapArrayRange(corners, 0, orig_num_corners, &input_map_id, &stride, (void **)&array_data, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0));
	do
	{
		corners_used = calculate_new_bounding_box(orig_num_corners, &tracking_data, old_array_data, array_data, stride);
	} while (validate_corners(orig_num_corners, &tracking_data, array_data, stride));
	
	// Copy all the valid data to the output corner array
	ERROR_CHECK_STATUS(vxAddArrayItems(output_corners, orig_num_corners, array_data, stride))
	ERROR_CHECK_STATUS(vxUnmapArrayRange(originals, original_map_id));
	ERROR_CHECK_STATUS(vxUnmapArrayRange(corners, input_map_id));
	if (corners_used < 2)
	{
		// need at least two features to be able to establish a centroid and spread
		vxAddLogEntry((vx_reference)node, VX_FAILURE, "No more valid data!");
		b_valid = vx_false_e;
	}
	// set output "output_data" parameter
	ERROR_CHECK_STATUS(vxAddArrayItems(output_data, 1, &tracking_data, sizeof(tracking_data)));
	// Set output "valid" parameter
	ERROR_CHECK_STATUS(vxCopyScalar(valid, &b_valid, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	//print_trackingdata("New tracking data", &tracking_data);
	return VX_SUCCESS;
}

vx_status registerTrackCentroids_kernel(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context,
                                       "app.userkernels.track_centroids",
                                       USER_KERNEL_TRACK_CENTROIDS,
                                       trackCentroids_function,
                                       6,   // numParams
                                       trackCentroids_validator,
                                       NULL,
                                       NULL);
    ERROR_CHECK_OBJECT(kernel);
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT,  VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED)); // input - original keypoints
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT,  VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED)); // input - tracking data
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_INPUT,  VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED)); // input - the initial array of key points
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 3, VX_OUTPUT, VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED)); // output- tracking data
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 4, VX_OUTPUT, VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED)); // output- filtered array of key points
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 5, VX_OUTPUT, VX_TYPE_SCALAR, VX_PARAMETER_STATE_REQUIRED)); // output- validity
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    vxAddLogEntry((vx_reference)context, VX_SUCCESS, "OK: registered user kernel app.userkernels.track_centroids\n");
    return VX_SUCCESS;
}
	
vx_node clearOutsideBoundsNode(
	/*
		Create a node to draw a bounding box or other details on an image
	*/
	vx_graph	graph,
	vx_image	input_image,
	vx_array	bounds,	// Holds a vx_rectangle_t (input)
	vx_image	output_image
	)
{
    vx_context context = vxGetContext((vx_reference)graph);
    vx_kernel kernel = vxGetKernelByEnum(context, USER_KERNEL_CLEAR_OUTSIDE_BOUNDS);
    ERROR_CHECK_OBJECT(kernel);
    vx_node node = vxCreateGenericNode(graph, kernel);
    ERROR_CHECK_OBJECT(node);
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference)input_image));
	ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference)bounds));
	ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 2, (vx_reference)output_image));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
    return node;

}
vx_status VX_CALLBACK clearOutsideBounds_function(vx_node node, const vx_reference * refs, vx_uint32 num)
{
	/*
		Output image equals input image but only for the bounding box region. Everything else is zeroed out...
	*/
	vx_image input_image=(vx_image)refs[0];
	vx_image output_image=(vx_image)refs[2];
	vx_rectangle_t bounds;
	vx_map_id map_id;
	vx_imagepatch_addressing_t addr;
	vx_pixel_value_t pixel;
	void *ptr;
	/* Zero out the output image, assume setting u32 to zero will be good for any format we would use*/
	pixel.U32 = 0;
	ERROR_CHECK_STATUS(vxSetImagePixelValues(output_image, &pixel));
	/* Get the bounding box */
	ERROR_CHECK_STATUS(vxCopyArrayRange((vx_array)refs[1], 0, 1, sizeof(bounds), &bounds, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));
	printf("Got bounding box [(%d, %d), (%d, %d)]\n", bounds.start_x, bounds.start_y, bounds.end_x, bounds.end_y);
	/* Map the input image for reading the area of the bounding box */
	ERROR_CHECK_STATUS(vxMapImagePatch(input_image, &bounds, 0, &map_id, &addr, &ptr, VX_READ_ONLY, VX_MEMORY_TYPE_NONE, 0));
	/* Now copy just what is inside the bounding box, from the input image to the output */
	ERROR_CHECK_STATUS(vxCopyImagePatch(output_image, &bounds, 0, &addr, ptr, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
	/* unmap the input image */
	ERROR_CHECK_STATUS(vxUnmapImagePatch(input_image, map_id));
	return VX_SUCCESS;
}

vx_status VX_CALLBACK clearOutsideBounds_validator(vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
	/* NEEDS WORK: Check type of references, allow all outputs to have unspecified metadata (virtuals) and if so, set */
	if (num != 3)
	{
		return VX_ERROR_INVALID_PARAMETERS;
	}
    vx_enum param_type;
	vx_df_image image_format;
    
	// parameter #1 -- check scalar type
    ERROR_CHECK_STATUS(vxQueryArray((vx_array)parameters[1], VX_ARRAY_ITEMTYPE, &param_type, sizeof(param_type)));
    if(param_type != VX_TYPE_RECTANGLE) // check that the scalar holds a user type of the correct sort
    {
	    return VX_ERROR_INVALID_TYPE;
    }

    // parameter #2 -- set output metatdata equal to input metadata
	ERROR_CHECK_STATUS(vxSetMetaFormatFromReference(metas[2], parameters[0]));
	return VX_SUCCESS;
}

vx_status registerClearOutsideBounds_kernel(vx_context context)
{
    vx_kernel kernel = vxAddUserKernel(context,
                                       "app.userkernels.clear_outside_bounds",
                                       USER_KERNEL_CLEAR_OUTSIDE_BOUNDS,
                                       clearOutsideBounds_function,
                                       3,   // numParams
                                       clearOutsideBounds_validator,
                                       NULL,
                                       NULL);
    ERROR_CHECK_OBJECT(kernel);
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT,  VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED)); // input - the image to modify
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_INPUT,  VX_TYPE_ARRAY, VX_PARAMETER_STATE_REQUIRED)); // input - the bounds data
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 2, VX_OUTPUT, VX_TYPE_IMAGE, VX_PARAMETER_STATE_REQUIRED)); // output- the modified image
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));
    ERROR_CHECK_STATUS(vxReleaseKernel(&kernel));
	vxAddLogEntry((vx_reference)context, VX_SUCCESS, "OK: registered user kernel app.userkernels.clear_outside_bounds\n");
    return VX_SUCCESS;
}

vx_status registerCentroidNodes(vx_context context)
/*
	Register the user struct and the three kernels 
 */
{
	user_struct_userTrackingData = vxRegisterUserStruct(context, sizeof(userTrackingData));
	return registerInitialCentroidCalculation_kernel(context) |
		   registerTrackCentroids_kernel(context) |
		   registerClearOutsideBounds_kernel(context);
}