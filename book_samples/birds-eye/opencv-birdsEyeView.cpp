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
 * \file    opencv-birdsEyeView.cpp
 * \example opencv-birdsEyeView
 * \brief   This sample implements the bird's eye view algorithm from
 * the OpenVX sample birdsEyeView.c using OpenCV.
 * \author  Victor Erukhimov <relrotciv@gmail.com>
 */


#include <math.h>
#include <opencv2/opencv.hpp>

using namespace cv;

int main(int argc, char** argv)
{
  if(argc != 3)
  {
    printf("opencv-birdsEyeView <input image> <output image>\n");
    exit(0);
  }

  Mat input = imread(argv[1]);
  Mat temp;

  Mat K = (Mat_<float>(3,3) << 8.4026236186715255e+02, 0., 3.7724917600845038e+02,
                              0., 8.3752885759166338e+02, 4.6712164335800873e+02,
                              0., 0., 1.);
  K = K*4.0f;
  K.at<float>(2, 2) = 1.0f;
  std::cout << "K = " << K << std::endl;
  std::cout << "Kinv = " << K.inv() << std::endl;

  Point3f p0(482.0f*4, 332.0f*4, 1.0f);
  Point3f pu = Mat(K.inv()*Mat(p0)).at<Point3f>(0);
  float phi = acos(-1) + atan(1.0f/pu.y);

  std::cout << "p0 = (" << p0 << "), pu = (" << pu << "), phi = " << phi << std::endl;

  // calculate homography, rotation around x axis to make the camera look down
  Mat H1 = Mat::zeros(3, 3, CV_32F);
  H1.at<float>(0, 0) = 1.0f;
  H1.at<float>(1, 1) = cos(phi);
  H1.at<float>(1, 2) = sin(phi);
  H1.at<float>(2, 1) = -sin(phi);
  H1.at<float>(2, 2) = cos(phi);

  std::cout << "phi = " << phi << std::endl;
  std::cout << "H1 = " << H1 << std::endl;

  // now we need to adjust offset and scale to map input image to
  // visible coordinates in the output image.
  Mat H = K*H1*K.inv();
  const Point3f p1(p0.x, p0.y*1.2, 1);
  const Point3f p2(p0.x, input.rows, 1);
  Point3f p1h = Mat(H*Mat(p1)).at<Point3f>(0, 0);
  p1h *= 1/p1h.z;
  Point3f p2h = Mat(H*Mat(p2)).at<Point3f>(0, 0);
  p2h *= 1/p2h.z;
  Mat scaleY = Mat::eye(3, 3, CV_32F);

  float scale = (p2h.y - p1h.y)/input.rows;
  scaleY.at<float>(0, 2) = input.cols*scale/2 - p0.x;
  scaleY.at<float>(1, 2) = -p1h.y;
  scaleY.at<float>(2, 2) = scale;

  std::cout << "scaleY = " << scaleY << std::endl << std::endl;
  std::cout << "H = " << H << std::endl << std::endl;
  std::cout << "K*H1 = " << K*H1 << std::endl << std::endl;

  H = scaleY*H;

  std::cout << "scaleY*H = " << H << std::endl << std::endl;

  Point3f corners[]= {Point3f(0, p0.y*1.15, 1), Point3f(input.cols, p0.y*1.15, 1),
    Point3f(input.cols, input.rows, 1), Point3f(0, input.rows, 1),
    p0, Point3f(p0.x, p0.y*1.1, 1)};

  for(int i = 0; i < sizeof(corners)/sizeof(Point3f); i++)
  {
    Point3f ph1 = Mat(K.inv()*Mat(corners[i])).at<Point3f>(0, 0);
    Point3f ph2 = Mat(H*Mat(corners[i])).at<Point3f>(0, 0);
    std::cout << "point " << i << " maps to: " << std::endl <<
      "  uni: (" << ph1.x/ph1.z << " " << ph1.y/ph1.z << ")" << std::endl <<
      "  output: (" << ph2.x/ph2.z << " " << ph2.y/ph2.z << ")" << std::endl;
  }

  Mat output;
  warpPerspective(input, output, H, input.size(), INTER_LINEAR, BORDER_CONSTANT, Scalar(0, 0, 0));
  imwrite(argv[2], output);

  return(0);
}
