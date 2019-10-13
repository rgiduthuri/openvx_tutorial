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
 * \file    undistortOpenCV.cpp
 * \example undistortOpenCV
 * \brief   This OpenCV sample creates a remap transformation that undistorts
 * an input image.
 * \author  Victor Erukhimov <relrotciv@gmail.com>
 */

#include "opencv2/opencv.hpp"

using namespace cv;

int main(int argc, char** argv)
{
  if(argc != 5)
  {
    printf("undistortOpenCV <camera params> <input image> <output image> <undistortMap>\n");
    return(0);
  }

  const char* camera_file = argv[1];
  const char* in_fname = argv[2];
  const char* out_fname = argv[3];
  const char* map_fname = argv[4];

  FileStorage fs(camera_file, FileStorage::READ);
  Mat intrinsic_params, dist_coeffs;
  fs["camera_matrix"] >> intrinsic_params;
  fs["distortion_coefficients"] >> dist_coeffs;
  int width, height;
  fs["image_width"] >> width;
  fs["image_height"] >> height;

  printf("Read width = %d, height = %d\n", width, height);
  std::cout << intrinsic_params << std::endl;
  std::cout << dist_coeffs << std::endl;

  Mat map1, map2, new_camera;
  initUndistortRectifyMap(intrinsic_params, dist_coeffs, Mat(),
    intrinsic_params, Size(width, height), CV_32FC2, map1, map2);

  printf("Completed undistort map\n");

  FileStorage fs1(map_fname, FileStorage::WRITE);
  fs1 << "remap" << map1;
	fs1 << "remap_src_width" << width;
	fs1 << "remap_src_height" << height;
	fs1 << "remap_dst_width" << width;
	fs1 << "remap_dst_height" << height;

  // now apply the remap to the input image
  Mat input_image = imread(in_fname);
  Mat output_image(input_image.cols, input_image.rows, input_image.type());
  remap(input_image, output_image, map1, Mat(), INTER_LINEAR);
  imwrite(out_fname, output_image);
}
