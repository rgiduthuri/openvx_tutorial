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
 * \file    example1.c
 * \example example1
 * \brief   Use the image creation functions to create a white rectangle on a black background
 * and count the corners in the result using the Fast Corners function.
 * \author  Stephen Ramm <stephen.v.ramm@gmail.com>
 */


#include <stdio.h>
#include <stdlib.h>
#include <VX/vx.h>
#include <VX/vxu.h>

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

  errorCheck(&context, vxGetStatus((vx_reference)image1), "Could not create image");

  vx_float32 strength_thresh_value = 128.0;
  vx_scalar strength_thresh = vxCreateScalar(context, VX_TYPE_FLOAT32, &strength_thresh_value);
  vx_array corners = vxCreateArray(context, VX_TYPE_KEYPOINT, 100);
  vx_array corners1 = vxCreateArray(context, VX_TYPE_KEYPOINT, 100);
  vx_size num_corners_value = 0;
  vx_scalar num_corners = vxCreateScalar(context, VX_TYPE_SIZE, &num_corners_value);
  vx_scalar num_corners1 = vxCreateScalar(context, VX_TYPE_SIZE, &num_corners_value);
  vx_keypoint_t *kp = calloc( 100, sizeof(vx_keypoint_t));

  errorCheck(&context,
             kp == NULL ||
             vxGetStatus((vx_reference)strength_thresh) ||
             vxGetStatus((vx_reference)corners) ||
             vxGetStatus((vx_reference)num_corners) ||
             vxGetStatus((vx_reference)corners1) ||
             vxGetStatus((vx_reference)num_corners1),
             "Could not create parameters for FastCorners");

  errorCheck(&context, vxuFastCorners(context, image1, strength_thresh, vx_true_e, corners, num_corners), "Fast Corners function failed");
  errorCheck(&context, vxuFastCorners(context, image1, strength_thresh, vx_false_e, corners1, num_corners1), "Fast Corners function failed");

  errorCheck(&context, vxCopyScalar(num_corners, &num_corners_value, VX_READ_ONLY, VX_MEMORY_TYPE_HOST), "vxCopyScalar failed");
  printf("Found %zu corners with non-max suppression\n", num_corners_value);

  errorCheck(&context, vxCopyArrayRange( corners, 0, num_corners_value, sizeof(vx_keypoint_t), kp,
                                        VX_READ_ONLY, VX_MEMORY_TYPE_HOST), "vxCopyArrayRange failed");
  for (int i=0; i<num_corners_value; ++i)
  {
    printf("Entry %3d: x = %d, y = %d\n", i, kp[i].x, kp[i].y);
  }

  errorCheck(&context, vxCopyScalar(num_corners1, &num_corners_value, VX_READ_ONLY, VX_MEMORY_TYPE_HOST), "vxCopyScalar failed");
  printf("Found %zu corners without non-max suppression\n", num_corners_value);

  errorCheck(&context, vxCopyArrayRange( corners1, 0, num_corners_value, sizeof(vx_keypoint_t), kp,
                                        VX_READ_ONLY, VX_MEMORY_TYPE_HOST), "vxCopyArrayRange failed");
  for (int i=0; i<num_corners_value; ++i)
  {
    printf("Entry %3d: x = %d, y = %d\n", i, kp[i].x, kp[i].y);
  }

  free(kp);
  vxReleaseContext(&context);
  return 0;
}
