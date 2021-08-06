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
 * \file    stitch_debug.c
 * \example stitch_debug
 * \brief   This OpenVX sample blends two images using predefined remap
 * transformations and blending coefficients (see homography-opencv.cpp),
 * computing intermediate results for debugging purposes.
 * \author  Victor Erukhimov <relrotciv@gmail.com>
 */


#include <stdio.h>
#include <stdlib.h>
#include "VX/vx.h"
#include "vxa/vxa.h"

vx_graph makeFilterGraph(vx_context context, vx_image image1, vx_image image2,
  vx_remap remap1, vx_image coeffs1, vx_remap remap2, vx_image coeffs2,
  vx_image output, vx_image output_remapped1, vx_image output_remapped2,
  vx_image output_weighted1, vx_image output_weighted2)
{
    /* Create virtual images */
    const int numu8 = 5;
    vx_image virtu8[numu8][3];
    const int nums16 = 1;
    vx_image virts16[nums16][3];

    vx_graph graph = vxCreateGraph(context);

    int i, j;

    for(i = 0; i < numu8; i++)
    {
      for (j = 0; j < 3; j++)
        virtu8[i][j] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
    }
    for(i = 0; i < nums16; i++)
    {
      for (j = 0; j < 3; j++)
        virts16[i][j] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
    }

    int output_width, output_height;
    vxQueryImage(output, VX_IMAGE_WIDTH, &output_width, sizeof(int));
    vxQueryImage(output, VX_IMAGE_HEIGHT, &output_height, sizeof(int));

    /* Create temporary images */
    vx_image image_remapped1[3], image_remapped2[3],
            image_weighted1[3], image_weighted2[3];

    for(i = 0; i < 3; i++)
    {
      image_remapped1[i] = vxCreateImage(context, output_width, output_height,
        VX_DF_IMAGE_U8);
      image_remapped2[i] = vxCreateImage(context, output_width, output_height,
        VX_DF_IMAGE_U8);
      image_weighted1[i] = vxCreateImage(context, output_width, output_height,
        VX_DF_IMAGE_S16);
      image_weighted2[i] = vxCreateImage(context, output_width, output_height,
        VX_DF_IMAGE_S16);
    }


    /* Create the same processing subgraph for each channel */
    enum vx_channel_e channels[] = {VX_CHANNEL_R, VX_CHANNEL_G, VX_CHANNEL_B};

    float _scale = 1.0f/(1<<12);
    vx_scalar scale = vxCreateScalar(context, VX_TYPE_FLOAT32, &_scale);

    int _shift = 0;
    vx_scalar shift = vxCreateScalar(context, VX_TYPE_INT32, &_shift);

    for(i = 0; i < 3; i++)
    {
      /* First, extract input and logo R, G, and B channels to individual
      virtual images */
      vxChannelExtractNode(graph, image1, channels[i], virtu8[0][i]);
      vxChannelExtractNode(graph, image2, channels[i], virtu8[1][i]);

      /* Add remap nodes */
      vxRemapNode(graph, virtu8[0][i], remap1, VX_INTERPOLATION_BILINEAR,
        image_remapped1[i]);
      vxRemapNode(graph, virtu8[1][i], remap2, VX_INTERPOLATION_BILINEAR,
        image_remapped2[i]);

      /* add multiply nodes */
      vxMultiplyNode(graph, image_remapped1[i], coeffs1, scale,
        VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN,
        image_weighted1[i]);
      vxMultiplyNode(graph, image_remapped2[i], coeffs2, scale,
        VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN,
        image_weighted2[i]);

      vxAddNode(graph, image_weighted1[i], image_weighted2[i],
        VX_CONVERT_POLICY_SATURATE, virts16[0][i]);

      /* convert from S16 to U8 */
      vxConvertDepthNode(graph, virts16[0][i], virtu8[2][i],
        VX_CONVERT_POLICY_SATURATE, shift);

      vxConvertDepthNode(graph, image_weighted1[i], virtu8[3][i],
          VX_CONVERT_POLICY_SATURATE, shift);
      vxConvertDepthNode(graph, image_weighted2[i], virtu8[4][i],
          VX_CONVERT_POLICY_SATURATE, shift);
    }

    vxChannelCombineNode(graph, virtu8[2][0], virtu8[2][1], virtu8[2][2],
      NULL, output);
    vxChannelCombineNode(graph, image_remapped1[0], image_remapped1[1],
      image_remapped1[2], NULL, output_remapped1);
    vxChannelCombineNode(graph, image_remapped2[0], image_remapped2[1],
        image_remapped2[2], NULL, output_remapped2);
    vxChannelCombineNode(graph, virtu8[3][0], virtu8[3][1],
        virtu8[3][2], NULL, output_weighted1);
    vxChannelCombineNode(graph, virtu8[4][0], virtu8[4][1],
        virtu8[4][2], NULL, output_weighted2);

    for (i = 0; i < numu8; i++)
      for(j = 0; j < 3; j++)
        vxReleaseImage(&virtu8[i][j]);
    for (i = 0; i < nums16; ++i)
      for(j = 0; j < 3; j++)
        vxReleaseImage(&virts16[i][j]);

    for(i = 0; i < 3; i++)
    {
      vxReleaseImage(&image_remapped1[i]);
      vxReleaseImage(&image_remapped2[i]);
      vxReleaseImage(&image_weighted1[i]);
      vxReleaseImage(&image_weighted2[i]);
    }

    vxReleaseScalar(&scale);
    vxReleaseScalar(&shift);
    return graph;
}

void log_callback(vx_context context, vx_reference ref,
  vx_status status, const char* string)
{
    printf("Log message: status %d, text: %s\n", (int)status, string);
}

int main(int argc, char **argv)
{
    int i;

    if(argc != 5)
    {
      printf("stitch_debug <image 1> <image 2> <stitch config> <output image>\n");
      return(-1);
    }

    const char* image1_filename = argv[1];
    const char* image2_filename = argv[2];
    const char* config_filename = argv[3];
    const char* output_filename = argv[4];

    vx_context context = vxCreateContext();

    /* Read the input images*/
    vx_image image1, image2;
    if(vxa_read_image(image1_filename, context, &image1) != 1)
    {
      printf("Error reading image 1\n");
      return(-1);
    }
    if(vxa_read_image(image2_filename, context, &image2) != 1)
    {
      printf("Error reading image 2\n");
      return(-1);
    }

    /* Read config images and remaps */
    vx_image coeffs1, coeffs2;
    if(vxa_import_opencv_image(config_filename, "coeffs1", context,
      &coeffs1, NULL, NULL) != 1)
    {
      printf("Error reading coeffs1\n");
      return(-1);
    }
    if(vxa_import_opencv_image(config_filename, "coeffs2", context,
      &coeffs2, NULL, NULL) != 1)
    {
      printf("Error reading coeffs2\n");
      return(-1);
    }

    int width, height;
    vx_remap remap1, remap2;
    if(vxa_import_opencv_remap(config_filename, "remap1", context, &remap1,
      &width, &height) != 1)
    {
      printf("Error reading remap1\n");
      return(-1);
    }
    if(vxa_import_opencv_remap(config_filename, "remap2", context, &remap2,
      NULL, NULL) != 1)
    {
      printf("Error reading remap2\n");
      return(-1);
    }
    /* Create an output image */
    vx_image output = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    vx_image remapped1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    vx_image remapped2 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    vx_image weighted1 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    vx_image weighted2 = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);

    /* Create a graph */
    vx_status status;
    vx_graph graph = makeFilterGraph(context, image1, image2,
      remap1, coeffs1, remap2, coeffs2, output, remapped1, remapped2,
      weighted1, weighted2);

    vxRegisterLogCallback(context, log_callback, vx_true_e);

    if((status = vxVerifyGraph(graph)))
    {
      printf("Graph verification failed, error code %d, %d\n",
        (int)status, (int)VX_ERROR_NOT_SUFFICIENT);
    }
    else if (vxProcessGraph(graph))
        printf("Error processing graph\n");
    else if (vxa_write_image(output, output_filename) != 1)
        printf("Problem writing the output image\n");

    /* Save indermediate results */
    vxa_write_image(remapped1, "remapped1.jpg");
    vxa_write_image(remapped2, "remapped2.jpg");
    vxa_write_image(weighted1, "weighted1.jpg");
    vxa_write_image(weighted2, "weighted2.jpg");

    vxReleaseImage(&image1);
    vxReleaseImage(&image2);
    vxReleaseRemap(&remap1);
    vxReleaseRemap(&remap2);
    vxReleaseImage(&coeffs1);
    vxReleaseImage(&coeffs2);
    vxReleaseImage(&output);
    vxReleaseImage(&remapped1);
    vxReleaseImage(&remapped2);
    vxReleaseImage(&weighted1);
    vxReleaseImage(&weighted2);

    vxReleaseContext(&context);
}
