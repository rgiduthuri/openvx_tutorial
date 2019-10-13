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
 * \file    undistort-remap.c
 * \example houghLines
 * \brief   This OpenVX sample applies an undistort remap transformation computed by the
 * undistortOpenCV.cpp sample.
 * \author  Victor Erukhimov <relrotciv@gmail.com>
 */

#include <stdio.h>
#include <stdlib.h>
#include <VX/vx.h>
#include "vxa/vxa.h"

vx_graph makeRemapGraph(vx_context context, vx_image input_image,
  vx_remap remap, vx_image output_image)
{
    /* Create virtual images */
    const int numu8 = 2;
    vx_image virtu8[numu8][3];
    int i, j;

    vx_graph graph = vxCreateGraph(context);

    for(i = 0; i < numu8; i++)
      for (j = 0; j < 3; j++)
        virtu8[i][j] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);

    /* Create the same processing subgraph for each channel */
    enum vx_channel_e channels[] = {VX_CHANNEL_R, VX_CHANNEL_G, VX_CHANNEL_B};

    for(i = 0; i < 3; i++)
    {
      /* First, extract input and logo R, G, and B channels to individual
      virtual images */
      vxChannelExtractNode(graph, input_image, channels[i], virtu8[0][i]);

      /* Add remap nodes */
      vxRemapNode(graph, virtu8[0][i], remap, VX_INTERPOLATION_BILINEAR,
        virtu8[1][i]);
    }

    vxChannelCombineNode(graph, virtu8[1][0], virtu8[1][1], virtu8[1][2],
      NULL, output_image);

    for (i = 0; i < numu8; i++)
      for(j = 0; j < 3; j++)
        vxReleaseImage(&virtu8[i][j]);

    return graph;
}

void log_callback(vx_context context, vx_reference ref,
  vx_status status, const char* string)
{
    printf("Log message: status %d, text: %s\n", (int)status, string);
}

int main(int argc, char **argv)
{
    if(argc != 4)
    {
      printf("undistort <remap> <input image> <output image>\n");
      return(-1);
    }

    const char* remap_filename = argv[1];
    const char* image_filename = argv[2];
    const char* output_filename = argv[3];

    vx_context context = vxCreateContext();

    /* Read the input images*/
    vx_image input_image;
    if(vxa_read_image(image_filename, context, &input_image) != 1)
    {
      printf("Error reading image 1\n");
      return(-1);
    }

    int width, height;
    vx_remap remap;
    if(vxa_import_opencv_remap(remap_filename, "remap", context, &remap,
      &width, &height) != 1)
    {
      printf("Error reading remap1\n");
      return(-1);
    }

    /* Create an output image */
    vx_image output_image = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);

    /* Create a graph */
    vx_status status;
    vx_graph graph = makeRemapGraph(context, input_image,
      remap, output_image);

    vxRegisterLogCallback(context, log_callback, vx_true_e);

    if((status = vxVerifyGraph(graph)))
    {
      printf("Graph verification failed, error code %d, %d\n",
        (int)status, (int)VX_ERROR_NOT_SUFFICIENT);
    }
    else if (vxProcessGraph(graph))
        printf("Error processing graph\n");
    else if (vxa_write_image(output_image, output_filename) != 1)
        printf("Problem writing the output image\n");
    vxReleaseContext(&context);
}
