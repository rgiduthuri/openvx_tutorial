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
 * \file    stitch-multiband.c
 * \example stitch
 * \brief   This OpenVX sample blends two images using predefined remap
 * transformations and blending coefficients (see homography-opencv.cpp)
 * \author  Victor Erukhimov <relrotciv@gmail.com>
 */

#include <stdio.h>
#include <stdlib.h>
#include <VX/vx.h>
#include <VX/vxu.h>
#include "vxa/vxa.h"

const int max_pyr_levels = 4;

vx_status _vxLaplacianPyramidNode(vx_graph graph, vx_image image, vx_pyramid pyr_image, vx_image output)
{
  vx_context context = vxGetContext((vx_reference)graph);

  /* get the number of pyramid levels */
  vx_size level_num;
  vxQueryPyramid(pyr_image, VX_PYRAMID_LEVELS, &level_num, sizeof(vx_size));


  /* get width and height of the input image */
  vx_uint32 width, height;
  vxQueryImage(image, VX_IMAGE_WIDTH, &width, sizeof(width));
  vxQueryImage(image, VX_IMAGE_HEIGHT, &height, sizeof(height));

  vx_pyramid pyr_gauss = vxCreateVirtualPyramid(graph, level_num + 1, VX_SCALE_PYRAMID_HALF,
    width, height, VX_DF_IMAGE_U8);

  vxGaussianPyramidNode(graph, image, pyr_gauss);

  for(int i = 0; i < level_num; i++)
  {
    vx_image level1 = vxGetPyramidLevel(pyr_gauss, i);
    vx_image level2 = vxGetPyramidLevel(pyr_gauss, i + 1);

    // get width and height of the level1
    vx_uint32 width_level1, height_level1;
    vxQueryImage(level1, VX_IMAGE_WIDTH, &width_level1, sizeof(width_level1));
    vxQueryImage(level1, VX_IMAGE_HEIGHT, &height_level1, sizeof(height_level1));

    vx_image upscale = vxCreateVirtualImage(graph, width_level1, height_level1, VX_DF_IMAGE_U8);
    vx_image smoothed = vxCreateVirtualImage(graph, width_level1, height_level1, VX_DF_IMAGE_U8);

    vxScaleImageNode(graph, level2, upscale, VX_INTERPOLATION_NEAREST_NEIGHBOR);
    vxGaussian3x3Node(graph, upscale, smoothed);

    vx_image laplacian_level = vxGetPyramidLevel(pyr_image, i);
    vxSubtractNode(graph, level1, smoothed, VX_CONVERT_POLICY_SATURATE, laplacian_level);

    vxReleaseImage(&level1);
    vxReleaseImage(&level2);
    vxReleaseImage(&upscale);
    vxReleaseImage(&smoothed);
    vxReleaseImage(&laplacian_level);
  }

  vxHalfScaleGaussianNode(graph, vxGetPyramidLevel(pyr_gauss, level_num - 1), output, 5);

  vxReleasePyramid(&pyr_gauss);

  return VX_SUCCESS;
}

vx_status _vxLaplacianReconstructNode(vx_graph graph, vx_pyramid pyr_image, vx_image input, vx_image output)
{
  vx_context context = vxGetContext((vx_reference)graph);

  /* get the number of pyramid levels */
  vx_size level_num;
  vxQueryPyramid(pyr_image, VX_PYRAMID_LEVELS, &level_num, sizeof(vx_size));

  vx_image sum[level_num];

  int _shift = 0;
  vx_scalar shift = vxCreateScalar(context, VX_TYPE_INT32, &_shift);

  for(int i = level_num - 1; i >= 0; i--)
  {
    vx_image level1 = vxGetPyramidLevel(pyr_image, i);
    vx_image level2 = i == level_num - 1 ? input : sum[i + 1];

    // get width and height of the level2
    vx_uint32 width_level2, height_level2;
    vxQueryImage(level2, VX_IMAGE_WIDTH, &width_level2, sizeof(width_level2));
    vxQueryImage(level2, VX_IMAGE_HEIGHT, &height_level2, sizeof(height_level2));

    // upsample the current level
    vx_image upscale = vxCreateVirtualImage(graph, 2*width_level2, 2*height_level2, VX_DF_IMAGE_U8);
    vx_image smoothed = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
    vxScaleImageNode(graph, level2, upscale, VX_INTERPOLATION_NEAREST_NEIGHBOR);
    vxGaussian3x3Node(graph, upscale, smoothed);

    // add it with the next level
    vx_image _sum = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
    vxAddNode(graph, smoothed, level1, VX_CONVERT_POLICY_SATURATE, _sum);

    sum[i] = vxCreateVirtualImage(graph, 2*width_level2, 2*height_level2, VX_DF_IMAGE_U8);
    vxConvertDepthNode(graph, _sum, i > 0 ? sum[i] : output, VX_CONVERT_POLICY_SATURATE, shift);

    vxReleaseImage(&upscale);
    vxReleaseImage(&smoothed);
    vxReleaseImage(&level1);
    vxReleaseImage(&_sum);
  }

  vxReleaseScalar(&shift);

  for(int i = 0; i < level_num - 1; i++)
  {
    vxReleaseImage(&sum[i]);
  }

  return(VX_SUCCESS);
}

void createBlendingWeightImages(vx_graph graph, vx_image coeffs1, vx_image coeffs2,
  int pyr_levels, vx_image* pyr_coeff_levels1, vx_image* pyr_coeff_levels2)
{
  vx_context context = vxGetContext((vx_reference)graph);

  int _shift4 = 4;
  vx_scalar shift4 = vxCreateScalar(context, VX_TYPE_INT32, &_shift4);

  int _shift0 = 0;
  vx_scalar shift0 = vxCreateScalar(context, VX_TYPE_INT32, &_shift0);

  float _scale = 1.0f/2;
  vx_scalar scale = vxCreateScalar(context, VX_TYPE_FLOAT32, &_scale);

  const int numu8 = 4;
  vx_image coeff_levels[numu8][max_pyr_levels];

  const int nums16 = 4;
  vx_image coeff_levels_s16[nums16][max_pyr_levels];

  for(int i = 0; i < numu8; i++)
  {
    for(int j = 0; j < pyr_levels; j++)
    {
      coeff_levels[i][j] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
    }
  }

  for(int i = 0; i < nums16; i++)
  {
    for(int j = 0; j < pyr_levels; j++)
    {
      coeff_levels_s16[i][j] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
    }
  }

  vxConvertDepthNode(graph, coeffs1, coeff_levels[0][0],
    VX_CONVERT_POLICY_SATURATE, shift4);
  vxGaussian3x3Node(graph, coeff_levels[0][0], coeff_levels[2][0]);

  vxConvertDepthNode(graph, coeffs2, coeff_levels[1][0],
    VX_CONVERT_POLICY_SATURATE, shift4);
  vxGaussian3x3Node(graph, coeff_levels[1][0], coeff_levels[3][0]);

  // building a pyramid and applying smoothing to each level
  for(int j = 1; j < pyr_levels; j++)
  {
    vxHalfScaleGaussianNode(graph, coeff_levels[0][j - 1],
      coeff_levels[0][j], 3);
    vxGaussian3x3Node(graph, coeff_levels[0][j], coeff_levels[2][j]);

    vxHalfScaleGaussianNode(graph, coeff_levels[1][j - 1],
      coeff_levels[1][j], 3);
    vxGaussian3x3Node(graph, coeff_levels[1][j], coeff_levels[3][j]);
  }

  // prepare a lookup table
  vx_lut lut = vxCreateLUT(context, VX_TYPE_INT16, 1024);

  // obtain the index of the 0 element
  vx_uint32 lut_offset;
  vxQueryLUT(lut, VX_LUT_OFFSET, &lut_offset, sizeof(vx_uint32));

  // fill the lut values
  // the sum of 2 U8 values varies from 0 to 510, definitely lower than 512
  // we are going to create an inverse image normalised by 512, so if the input
  // image value is x (0 <= x <= 510), the ouptut value after LUT should be
  // y = (vx_uint16)(512/x)
  vx_int16 lut_data[1024];
  for(int i = 0; i < lut_offset; i++)
  {
    lut_data[i] = 0;
  }

  for(int i = 0; i < 512; i++)
  {
    lut_data[i + lut_offset] = 510/(i == 0 ? 1 : i);
  }

  vxCopyLUT(lut, lut_data, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

  // summing up weights and building a normalization image
  for(int j = 0; j < pyr_levels; j++)
  {
    vxAddNode(graph, coeff_levels[2][j], coeff_levels[3][j],
      VX_CONVERT_POLICY_SATURATE, coeff_levels_s16[0][j]);

    vxTableLookupNode(graph, coeff_levels_s16[0][j], lut,
      coeff_levels_s16[1][j]);

    vxMultiplyNode(graph, coeff_levels[2][j], coeff_levels_s16[1][j], scale,
      VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, coeff_levels_s16[2][j]);
    vxConvertDepthNode(graph, coeff_levels_s16[2][j], pyr_coeff_levels1[j],
      VX_CONVERT_POLICY_SATURATE, shift0);

    vxMultiplyNode(graph, coeff_levels[3][j], coeff_levels_s16[1][j], scale,
      VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_ZERO, coeff_levels_s16[3][j]);
    vxConvertDepthNode(graph, coeff_levels_s16[3][j], pyr_coeff_levels2[j],
      VX_CONVERT_POLICY_SATURATE, shift0);

  }

  vxReleaseScalar(&shift4);
  vxReleaseScalar(&shift0);
  vxReleaseScalar(&scale);
}

vx_graph makeGraph(vx_context context, vx_image image1, vx_image image2,
  vx_remap remap1, vx_image coeffs1, vx_remap remap2, vx_image coeffs2,
  int pyr_levels, vx_image output)
{
    /* Create virtual images */
    const int numu8 = 8;
    vx_image virtu8[numu8][3];

    const int nums16 = 5;
    vx_image virts16[nums16][max_pyr_levels][3];

    vx_image pyr_img_levels[3][max_pyr_levels][3];
    vx_image pyr_coeff_levels1[max_pyr_levels], pyr_coeff_levels2[max_pyr_levels];

    vx_pyramid pyr_image1[3], pyr_image2[3], pyr_output[3],
      pyr_coeffs1, pyr_coeffs2;

    if(pyr_levels > max_pyr_levels)
    {
      return NULL;
    }

    vx_graph graph = vxCreateGraph(context);

    int i, j, s;

    /* get width and height of the input images */
    vx_uint32 width, height;
    vxQueryRemap(remap1, VX_REMAP_DESTINATION_WIDTH, &width, sizeof(width));
    vxQueryRemap(remap1, VX_REMAP_DESTINATION_HEIGHT, &height, sizeof(height));

    // create virtual images
    for(i = 0; i < numu8; i++)
      for (j = 0; j < 3; j++)
      {
        if(i == 2 || i == 3)
        {
          virtu8[i][j] = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
        }
        else if(i >= 4 && i <= 6)
        {
          virtu8[i][j] = vxCreateVirtualImage(graph, width/(1 << (pyr_levels - 1)),
            height/(1 << (pyr_levels - 1)), VX_DF_IMAGE_U8);
        }
        else
        {
          virtu8[i][j] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
        }
      }

    for(s = 0; s < nums16; s++)
      for (j = 0; j < pyr_levels; j++)
        for(i = 0; i < 3; i++)
          virts16[s][j][i] = vxCreateVirtualImage(graph, 0, 0,
            VX_DF_IMAGE_S16);

    /* Create the same processing subgraph for each channel */
    enum vx_channel_e channels[] = {VX_CHANNEL_R, VX_CHANNEL_G, VX_CHANNEL_B};

    float _scale = 1.0f/(1<<8);//1.0f/(1<<12);
    vx_scalar scale = vxCreateScalar(context, VX_TYPE_FLOAT32, &_scale);

    int _shift = 0;
    vx_scalar shift = vxCreateScalar(context, VX_TYPE_INT32, &_shift);

    // create pyramids for images and coefficients
    for(i = 0; i < 3; i++)
    {
      pyr_image1[i] = vxCreatePyramid(context, pyr_levels - 1,
        VX_SCALE_PYRAMID_HALF, width, height, VX_DF_IMAGE_S16);
      pyr_image2[i] = vxCreatePyramid(context, pyr_levels - 1,
        VX_SCALE_PYRAMID_HALF, width, height, VX_DF_IMAGE_S16);
      pyr_output[i] = vxCreatePyramid(context, pyr_levels - 1,
        VX_SCALE_PYRAMID_HALF, width, height, VX_DF_IMAGE_S16);
    }

    // build a gaussian pyramid images for each level
    for(j = 0; j < pyr_levels; j++)
    {
      pyr_coeff_levels1[j] = vxCreateVirtualImage(graph, 0, 0,
        VX_DF_IMAGE_U8);
      pyr_coeff_levels2[j] = vxCreateVirtualImage(graph, 0, 0,
        VX_DF_IMAGE_U8);
    }

    createBlendingWeightImages(graph, coeffs1, coeffs2, pyr_levels,
      pyr_coeff_levels1, pyr_coeff_levels2);

    for(i = 0; i < 3; i++)
    {
      /* First, extract input and logo R, G, and B channels to individual
      virtual images */
      vxChannelExtractNode(graph, image1, channels[i], virtu8[0][i]);
      vxChannelExtractNode(graph, image2, channels[i], virtu8[1][i]);

      /* Add remap nodes */
      vxRemapNode(graph, virtu8[0][i], remap1, VX_INTERPOLATION_BILINEAR,
        virtu8[2][i]);
      vxRemapNode(graph, virtu8[1][i], remap2, VX_INTERPOLATION_BILINEAR,
        virtu8[3][i]);

      // compute laplacian pyramid for each image channel
      _vxLaplacianPyramidNode(graph, virtu8[2][i], pyr_image1[i], virtu8[4][i]);
      _vxLaplacianPyramidNode(graph, virtu8[3][i], pyr_image2[i], virtu8[5][i]);

      for(int j = 0; j < pyr_levels - 1; j++)
      {
        pyr_img_levels[0][j][i] = vxGetPyramidLevel(pyr_image1[i], j);
        pyr_img_levels[1][j][i] = vxGetPyramidLevel(pyr_image2[i], j);
        pyr_img_levels[2][j][i] = vxGetPyramidLevel(pyr_output[i], j);

        // add multiply nodes
        vxMultiplyNode(graph, pyr_img_levels[0][j][i], pyr_coeff_levels1[j],
          scale, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN,
          virts16[0][j][i]);
        vxMultiplyNode(graph, pyr_img_levels[1][j][i], pyr_coeff_levels2[j],
          scale, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN,
          virts16[1][j][i]);
        vxAddNode(graph, virts16[0][j][i], virts16[1][j][i],
          VX_CONVERT_POLICY_SATURATE, pyr_img_levels[2][j][i]);
      }

      // add multiply nodes for the last pyramid levels
      vxMultiplyNode(graph, virtu8[4][i], pyr_coeff_levels1[pyr_levels - 1],
        scale, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN,
        virts16[2][pyr_levels - 1][i]);
      vxMultiplyNode(graph, virtu8[5][i], pyr_coeff_levels2[pyr_levels - 1],
        scale, VX_CONVERT_POLICY_SATURATE, VX_ROUND_POLICY_TO_NEAREST_EVEN,
        virts16[3][pyr_levels - 1][i]);
      vxAddNode(graph, virts16[2][pyr_levels - 1][i],
        virts16[3][pyr_levels - 1][i], VX_CONVERT_POLICY_SATURATE,
        virts16[4][pyr_levels - 1][i]);

      // convert from S16 to U8
      vxConvertDepthNode(graph, virts16[4][pyr_levels - 1][i], virtu8[6][i],
        VX_CONVERT_POLICY_SATURATE, shift);

      _vxLaplacianReconstructNode(graph, pyr_output[i], virtu8[6][i],
        virtu8[7][i]);
    }

      vxChannelCombineNode(graph, virtu8[7][0], virtu8[7][1], virtu8[7][2],
      NULL, output);

    for (i = 0; i < numu8; i++)
      for(j = 0; j < 3; j++)
        vxReleaseImage(&virtu8[i][j]);


    for (s = 0; s < nums16; s++)
      for(j = 0; j < pyr_levels; j++)
        for(i = 0; i < 3; i++)
          vxReleaseImage(&virts16[s][j][i]);

    for(i = 0; i < 3; i++)
    {
      vxReleasePyramid(&pyr_image1[i]);
      vxReleasePyramid(&pyr_image2[i]);
      vxReleasePyramid(&pyr_output[i]);
    }

    for (s = 0; s < 3; s++)
      for(j = 0; j < pyr_levels - 1; j++)
        for(i = 0; i < 3; i++)
          vxReleaseImage(&pyr_img_levels[s][j][i]);

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
    if(argc != 5)
    {
      printf("stitch <image 1> <image 2> <stitch config> <output image>\n");
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

    /* number of pyramid levels */
    const int pyr_levels = 4;

    /* Create a graph */
    vx_status status;
    vx_graph graph = makeGraph(context, image1, image2,
      remap1, coeffs1, remap2, coeffs2, pyr_levels, output);
/*
    vx_uint32 num_nodes;
    vxQueryGraph(graph, VX_GRAPH_NUMNODES, &num_nodes, sizeof(num_nodes));
    printf("Number of nodes: %d\n", (int)num_nodes);
*/
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

    vxReleaseContext(&context);
}
