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
 * \file    example4a.c
 * \example example4a
 * \brief   In this example we re-imnplement example three, using graph mode.
 * \author  Stephen Ramm <stephen.v.ramm@gmail.com>
 */


#include <stdio.h>
#include <stdlib.h>
#include "VX/vx.h"
#include "writeImage.h"

void errorCheck(vx_context *context_p, vx_status status, const char *message)
{
    if (status)
    {
      puts("ERROR! ");
      puts(message);
      vxReleaseContext(context_p);
      exit(1);
    }
}

vx_image makeInputImage(vx_context context, vx_uint32 width, vx_uint32 height)
{
  vx_image image = vxCreateImage(context, 100U, 100U, VX_DF_IMAGE_U8);
  if (width > 48)
      width = 48;
  if (height > 48)
      height = 48;
  vx_rectangle_t rect = {
    .start_x = 50 - width, .start_y = 50 - height, .end_x = 50 + width, .end_y = 50 + height
  };

  if (VX_SUCCESS == vxGetStatus((vx_reference)image))
  {
    vx_image roi = vxCreateImageFromROI(image, &rect);
    vx_pixel_value_t pixel_white, pixel_black;
    pixel_white.U8 = 255;
    pixel_black.U8 = 0;
    if (VX_SUCCESS == vxGetStatus((vx_reference)roi) &&
        VX_SUCCESS == vxSetImagePixelValues(image, &pixel_black) &&
        VX_SUCCESS == vxSetImagePixelValues(roi, &pixel_white))
      vxReleaseImage(&roi);
    else
      vxReleaseImage(&image);
  }
  return image;
}

vx_graph makeTestGraph(vx_context context)
{
    vx_graph graph = vxCreateGraph(context);
    int i;
    vx_image imagesU8[5], imagesS16[3];
    vx_image input = vxCreateImage(context, 100U, 100U, VX_DF_IMAGE_U8);

    for (i = 0; i < 5; ++i)
      imagesU8[i] = vxCreateVirtualImage(graph, 100, 100, VX_DF_IMAGE_U8);
    for (i = 0; i < 3; ++i)
        imagesS16[i] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_VIRT);

    vx_matrix warp_matrix = vxCreateMatrix(context, VX_TYPE_FLOAT32, 2U, 3U);
    vx_float32 matrix_values[6] = {0.0, 1.0, 1.0, 0.0, 0.0, 0.0 };       /* Rotate through 90 degrees */
    vx_float32 strength_thresh_value = 128.0;
    vx_scalar strength_thresh = vxCreateScalar(context, VX_TYPE_FLOAT32, &strength_thresh_value);
    vx_array corners = vxCreateArray(context, VX_TYPE_KEYPOINT, 100);
    vx_size num_corners_value = 0;
    vx_int32 shift_value = 1;
    vx_scalar num_corners = vxCreateScalar(context, VX_TYPE_SIZE, &num_corners_value);
    vx_scalar shift = vxCreateScalar(context, VX_TYPE_INT32, &shift_value);

    vxCopyMatrix(warp_matrix, matrix_values, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    /* Create the nodes to do the processing, order of creation is not important */
    vx_node last_node = vxFastCornersNode(graph, imagesU8[4], strength_thresh, vx_true_e, corners, num_corners);
    vxDilate3x3Node(graph, imagesU8[3], imagesU8[4]);
    vxConvertDepthNode(graph, imagesS16[2], imagesU8[3], VX_CONVERT_POLICY_SATURATE, shift);
    vxMagnitudeNode(graph, imagesS16[0], imagesS16[1], imagesS16[2]);
    vxSobel3x3Node(graph, imagesU8[2], imagesS16[0], imagesS16[1]);
    vxOrNode(graph, imagesU8[0], imagesU8[1], imagesU8[2]);
    vxWarpAffineNode(graph, imagesU8[0], warp_matrix, VX_INTERPOLATION_NEAREST_NEIGHBOR, imagesU8[1]);

    /* Setup input parameter using a Copy node */
    vxAddParameterToGraph(graph, vxGetParameterByIndex(vxCopyNode(graph, (vx_reference)input, (vx_reference)imagesU8[0]), 0));

    /* Setup the output parameters from the last node */
    vxAddParameterToGraph(graph, vxGetParameterByIndex(last_node, 3));    /* array of corners */
    vxAddParameterToGraph(graph, vxGetParameterByIndex(last_node, 4));    /* number of corners */

    /* Add another output parameter to the graph */
    vx_image output = vxCreateImage(context, 100U, 100U, VX_DF_IMAGE_U8);
    vxAddParameterToGraph(graph, vxGetParameterByIndex(vxCopyNode(graph, (vx_reference)imagesU8[4], (vx_reference)output), 1));

    /* Release resources */
    vxReleaseImage(&input);
    vxReleaseImage(&output);
    for (i = 0; i < 5; ++i)
        vxReleaseImage(&imagesU8[i]);
    for (i = 0; i < 3; ++i)
        vxReleaseImage(&imagesS16[i]);
    vxReleaseMatrix(&warp_matrix);
    vxReleaseScalar(&strength_thresh);
    vxReleaseScalar(&num_corners);
    vxReleaseScalar(&shift);
    vxReleaseArray(&corners);

    return graph;
}

vx_reference getGraphParameter(vx_graph graph, vx_uint32 index)
{
    vx_parameter p = vxGetGraphParameterByIndex(graph, index);
    vx_reference ref = NULL;
    vxQueryParameter(p, VX_PARAMETER_REF, &ref, sizeof(ref));
    vxReleaseParameter(&p);
    return ref;
}

void showResults(vx_graph graph, vx_image image, const char * message)
{
    vx_context context = vxGetContext((vx_reference)graph);
    puts(message);
    vxSetGraphParameterByIndex(graph, 0, (vx_reference)image);
    if (VX_SUCCESS == vxProcessGraph(graph))
    {
        vx_size num_corners_value = 0;
        vx_keypoint_t *kp = calloc( 100, sizeof(vx_keypoint_t));
        errorCheck(&context, vxCopyScalar((vx_scalar)getGraphParameter(graph, 2), &num_corners_value,
                                          VX_READ_ONLY, VX_MEMORY_TYPE_HOST), "vxCopyScalar failed");
        printf("Found %zu corners with non-max suppression\n", num_corners_value);

        /* Array can only hold 100 values */
        if (num_corners_value > 100)
            num_corners_value = 100;

        errorCheck(&context, vxCopyArrayRange((vx_array)getGraphParameter(graph, 1), 0,
                                              num_corners_value, sizeof(vx_keypoint_t), kp,
                                              VX_READ_ONLY, VX_MEMORY_TYPE_HOST), "vxCopyArrayRange failed");
        for (int i=0; i<num_corners_value; ++i)
        {
            printf("Entry %3d: x = %d, y = %d\n", i, kp[i].x, kp[i].y);
        }

        free(kp);
    }
    else
    {
        printf("Graph processing failed!");
    }
}

int main(void)
{
    vx_context context = vxCreateContext();
    errorCheck(&context, vxGetStatus((vx_reference)context), "Could not create a vx_context\n");

    vx_graph graph = makeTestGraph(context);

    vx_image image1 = makeInputImage(context, 30, 10);
    vx_image image2 = makeInputImage(context, 25, 25);

    showResults(graph, image1, "Results for Image 1");
    writeImage((vx_image)getGraphParameter(graph, 3), "example4-1.pgm");
    showResults(graph, image2, "Results for Image 2");
    writeImage((vx_image)getGraphParameter(graph, 3), "example4-2.pgm");

    vxReleaseContext(&context);
    return 0;
}
