/*
 * Copyright (c) 2019 Victor Erukhimov
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
 * \file    houghLines.c
 * \example houghLines
 * \brief   This OpenVX sample finds lines in an input image using Hough
 * Transform and draws them on top of of the input image.
 * \author  Victor Erukhimov <relrotciv@gmail.com>
 */

#include <VX/vx.h>
#include <vxa/vxa.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "readImage.h"
#include "writeImage.h"

void log_callback(vx_context context, vx_reference ref,
  vx_status status, const char* string)
{
    printf("Log message: status %d, text: %s\n", (int)status, string);
}

vx_graph makeHoughLinesGraph(vx_context context, vx_image input,
  vx_image* binary, vx_array lines)
{
    vx_uint32 width, height;
    vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(vx_uint32));
    vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(vx_uint32));
    printf("Read width %d, height %d\n", width, height);

    int widthr = width/4;
    int heightr = height/4;

    vx_graph graph = vxCreateGraph(context);

    #define nums16 (3)
    vx_image virt_s16[nums16];

    /* create virtual images */
    vx_image virt_nv12 = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_NV12);
    vx_image virt_y = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
    vx_image virt_yr = vxCreateVirtualImage(graph, widthr, heightr,
      VX_DF_IMAGE_U8);
    vx_image binary_thresh = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);

    for(int i = 0; i < nums16; i++)
    {
      virt_s16[i] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
    }

    *binary = vxCreateImage(context, widthr, heightr, VX_DF_IMAGE_U8);

    /* extract grayscale channel */
    vxColorConvertNode(graph, input, virt_nv12);
    vxChannelExtractNode(graph, virt_nv12, VX_CHANNEL_Y, virt_y);

    /* resize down */
    vxScaleImageNode(graph, virt_y, virt_yr, VX_INTERPOLATION_BILINEAR);

    /* compute gradient */
    vxSobel3x3Node(graph, virt_yr, virt_s16[0], virt_s16[1]);
    vxMagnitudeNode(graph, virt_s16[0], virt_s16[1], virt_s16[2]);

    /* setup threshold value */
    vx_threshold thresh = vxCreateThresholdForImage(context,
      VX_THRESHOLD_TYPE_BINARY, VX_DF_IMAGE_S16, VX_DF_IMAGE_U8);
    vx_pixel_value_t pixel_value;
    pixel_value.S16 = 256;
    vxCopyThresholdValue(thresh, &pixel_value, VX_WRITE_ONLY,
      VX_MEMORY_TYPE_HOST);

    vx_status status = vxGetStatus((vx_reference)thresh);
    if(status != VX_SUCCESS)
    {
      printf("Issue with thresh: %d\n", status);
    }

    vx_node thresh_node = vxThresholdNode(graph, virt_s16[2], thresh,
      binary_thresh);
    status = vxGetStatus((vx_reference)thresh_node);
    if(status != VX_SUCCESS)
    {
      printf("Issue with threshold node: %d\n", status);
    }

    /* dilate the threshold output */
    vxDilate3x3Node(graph, binary_thresh, *binary);

    /* run hough transform */
    vx_hough_lines_p_t hough_params;
    hough_params.rho = 1.0f;
    hough_params.theta = 3.14f/180;
    hough_params.threshold = 100;
    hough_params.line_length = 100;
    hough_params.line_gap = 10;
    hough_params.theta_max = 3.14;
    hough_params.theta_min = 0.0;

    vx_scalar num_lines = vxCreateScalar(context, VX_TYPE_SIZE, NULL);
    vxHoughLinesPNode(graph, *binary, &hough_params, lines, num_lines);

    return graph;
}

int main(int argc, char **argv)
{
    if (argc != 4)
    {
        printf("Find straight lines in an image\n"
               "%s <input> <binary> <lines>\n", (char *)argv[0]);
        exit(0);
    }

    const char* input_filename = argv[1];
    const char* binary_filename = argv[2];
    const char* lines_filename = argv[3];

    vx_context context = vxCreateContext();
    vx_image image, binary;
    vxa_read_image((const char *)input_filename, context, &image);

    vx_uint32 width, height;
    vxQueryImage(image, VX_IMAGE_WIDTH, &width, sizeof(vx_uint32));
    vxQueryImage(image, VX_IMAGE_HEIGHT, &height, sizeof(vx_uint32));

    /* create an array for storing hough lines output */
    const vx_size max_num_lines = 2000;
    vx_array lines = vxCreateArray(context, VX_TYPE_LINE_2D, max_num_lines);
    vx_graph graph = makeHoughLinesGraph(context, image, &binary, lines);

    vxRegisterLogCallback(context, log_callback, vx_true_e);

    vxProcessGraph(graph);

    vxa_write_image(binary, binary_filename);

    // draw the lines
    vx_pixel_value_t color;
    color.RGB[0] = 0;
    color.RGB[1] = 255;
    color.RGB[2] = 0;
    vx_image image_lines;
    vx_size _num_lines;
    vxQueryArray(lines, VX_ARRAY_NUMITEMS, &_num_lines, sizeof(_num_lines));
    draw_lines(context, binary, lines, _num_lines,
      &color, 2, &image_lines);
    vxa_write_image(image_lines, lines_filename);

    vxReleaseContext(&context);
    return(0);
}
