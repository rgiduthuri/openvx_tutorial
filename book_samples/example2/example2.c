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
 * \file    example2.c
 * \example example2
 * \brief   In this example we build on the first, and use an affine
 * transformation to tate the image by 90 degrees, and then logically OR it
 * with the original. Using Fast Corners on the result yeilds a larger number
 * of corners, it should be obvious that there will be 3 times as many.
 * \author  Stephen Ramm <stephen.v.ramm@gmail.com>
 */


#include <stdio.h>
#include <stdlib.h>
#include "VX/vx.h"
#include "VX/vxu.h"

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

vx_image makeInputImage(vx_context context)
{
  vx_image image = vxCreateImage(context, 100U, 100U, VX_DF_IMAGE_U8);
  vx_rectangle_t rect = {
    .start_x = 20, .start_y = 40, .end_x=80, .end_y = 60
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

int main(void)
{
  vx_context context = vxCreateContext();

  errorCheck(&context, vxGetStatus((vx_reference)context), "Could not create a vx_context\n");

  vx_image image1 = makeInputImage(context);

  errorCheck(&context, vxGetStatus((vx_reference)image1), "Could not create first image");

  vx_image image2 = vxCreateImage(context, 100U, 100U, VX_DF_IMAGE_U8);
  vx_image image3 = vxCreateImage(context, 100U, 100U, VX_DF_IMAGE_U8);
  vx_matrix warp_matrix = vxCreateMatrix(context, VX_TYPE_FLOAT32, 2U, 3U);
  vx_float32 matrix_values[3][2] = {       /* Rotate through 90 degrees */
      {0.0, 1.0},         /* x coefficients */
      {1.0, 0.0},         /* y coefficients */
      {0.0, 0.0}          /* offsets */
  };
  vx_float32 strength_thresh_value = 128.0;
  vx_scalar strength_thresh = vxCreateScalar(context, VX_TYPE_FLOAT32, &strength_thresh_value);
  vx_array corners = vxCreateArray(context, VX_TYPE_KEYPOINT, 100);
  vx_size num_corners_value = 0;
  vx_scalar num_corners = vxCreateScalar(context, VX_TYPE_SIZE, &num_corners_value);
  vx_keypoint_t *kp = calloc( 100, sizeof(vx_keypoint_t));

  errorCheck(&context,
             kp == NULL ||
             vxGetStatus((vx_reference)strength_thresh) ||
             vxGetStatus((vx_reference)corners) ||
             vxGetStatus((vx_reference)num_corners) ||
             vxGetStatus((vx_reference)image2) ||
             vxGetStatus((vx_reference)image3) ||
             vxGetStatus((vx_reference)warp_matrix),
             "Could not create objects");

  errorCheck(&context, vxCopyMatrix(warp_matrix, matrix_values, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST), "Could not initialise the matrix");

  errorCheck(&context, /* Now image2 set to image 1 rotated */
		 vxuWarpAffine(context, image1, warp_matrix, VX_INTERPOLATION_NEAREST_NEIGHBOR, image2) ||
		/* image3 set to logical OR of images 1 and 2 */
		vxuOr(context, image1, image2, image3) ||
		/*And now count the corners */
		vxuFastCorners(context, image3, strength_thresh, vx_true_e, corners, num_corners),
		"Image functions failed");

  errorCheck(&context, vxCopyScalar(num_corners, &num_corners_value, VX_READ_ONLY, VX_MEMORY_TYPE_HOST), "vxCopyScalar failed");
  printf("Found %zu corners with non-max suppression\n", num_corners_value);

  errorCheck(&context, vxCopyArrayRange( corners, 0, num_corners_value, sizeof(vx_keypoint_t), kp,
                                        VX_READ_ONLY, VX_MEMORY_TYPE_HOST), "vxCopyArrayRange failed");
  for (int i=0; i<num_corners_value; ++i)
  {
    printf("Entry %3d: x = %d, y = %d\n", i, kp[i].x, kp[i].y);
  }

  free(kp);
  vxReleaseContext(&context);
  return 0;
}
