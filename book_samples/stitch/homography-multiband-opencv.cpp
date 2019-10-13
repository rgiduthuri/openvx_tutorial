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
 * \file    homography-multiblend-opencv.cpp
 * \example homography-multiblend-opencv
 * \brief   This example finds a homography transformation between two
 * input images, and computes remap transformations and and blending
 * coefficients using OpenCV.
 * \author  Victor Erukhimov <relrotciv@gmail.com>
 */

#include "opencv2/opencv_modules.hpp"
#include <stdio.h>
#include <iostream>
#include <vector>

# include "opencv2/core/core.hpp"
# include "opencv2/imgproc/imgproc.hpp"
# include "opencv2/features2d/features2d.hpp"
# include "opencv2/highgui/highgui.hpp"
# include "opencv2/calib3d/calib3d.hpp"

using namespace cv;

void readme();

void drawMatchesEx(const Mat& img1, const std::vector<KeyPoint>& keypoints1,
                  const Mat& img2, const std::vector<KeyPoint>& keypoints2,
                  const std::vector<DMatch>& matches, Mat& outImg,
                  const Scalar& matchColor = Scalar::all(-1),
                  const Scalar& singlePointColor = Scalar::all(-1),
                  const std::vector<char>& matchesMask = std::vector<char>())
{
  Size size1 = img1.size();
  Size size2 = img2.size();

  float resize1 = min(640.0f/img1.cols, 480.0f/img1.rows);
  float resize2 = min(640.0f/img2.cols, 480.0f/img2.rows);
  Size size1r = Size(size1.width*resize1, size1.height*resize1);
  Size size2r = Size(size2.width*resize2, size2.height*resize2);

  // resize images
  Mat img1r, img2r;
  resize(img1, img1r, size1r);
  resize(img2, img2r, size2r);

  //resize keypoints_
  std::vector<KeyPoint> keypoints1r, keypoints2r;
  for(std::vector<KeyPoint>::const_iterator it = keypoints1.begin();
    it != keypoints1.end(); it++)
  {
    KeyPoint p = *it;
    p.pt *= resize1;
    keypoints1r.push_back(p);
  }
  for(std::vector<KeyPoint>::const_iterator it = keypoints2.begin();
    it != keypoints2.end(); it++)
  {
    KeyPoint p = *it;
    p.pt *= resize1;
    keypoints2r.push_back(p);
  }

  drawMatches(img1r, keypoints1r, img2r, keypoints2r, matches, outImg,
    matchColor, singlePointColor, matchesMask, DrawMatchesFlags::DEFAULT);
}

void drawMatches2(const Mat& img1, const std::vector<Point2f>& points1,
                  const Mat& img2, const std::vector<Point2f>& points2,
                  Mat& outImg)
{
  std::vector<KeyPoint> keypoints1, keypoints2;
  std::vector<DMatch> matches;

  for(int i = 0; i < points1.size(); i++)
  {
    keypoints1.push_back(KeyPoint(points1[i], 1.0));
    keypoints2.push_back(KeyPoint(points2[i], 1.0));
    matches.push_back(DMatch(i, i, 0.0));
  }

  drawMatchesEx(img1, keypoints1, img2, keypoints2, matches, outImg);
}

void imshow_ex(const char* window_name, const Mat& img, int width)
{
  Mat imgr(img.rows*width/img.cols, width, img.type());
  resize(img, imgr, imgr.size());
  imshow(window_name, imgr);
}

void generateWeightImage(const Size& size, Mat& weights)
{
  weights = Mat(size.height, size.width, CV_32FC1);
  for(int y = 0; y < size.height; y++)
  {
    float wy = 1.0f - fabs(2.0f*y/size.height - 1.0f);
    for(int x = 0; x < size.width; x++)
    {
      float wx = 1.0f - fabs(2.0f*x/size.width - 1.0f);
      weights.at<float>(y, x) = wx*wy;
    }
  }
}

void gaussianPyramid(const Mat& src, std::vector<Mat>& pyramid)
{
  src.copyTo(pyramid[0]);
  for(size_t i = 1; i < pyramid.size(); i++)
  {
    pyrDown(pyramid[i - 1], pyramid[i]);
  }
}

void laplacianPyramid(const Mat& src, std::vector<Mat>& pyramid)
{
  Mat current_level = src;
  for(size_t i = 0; i < pyramid.size() - 1; i++)
  {
    Mat next_level, gauss;
    pyrDown(current_level, next_level);
    pyrUp(next_level, gauss);
    std::cout << "current level size: " << current_level.size() <<
      ", gauss size: " << gauss.size() << std::endl;
    pyramid[i] = current_level - gauss;
    std::cout << "2" << std::endl;
    current_level = next_level;
  }
  pyramid.back() = current_level;
}

void imageFromLaplacianPyramid(const std::vector<Mat>& pyramid, Mat& dst)
{
  Mat current;
  std::cout << "starting laplacian reconstruction" << std::endl;
  for(int i = (int)pyramid.size() - 2; i >= 0; i--)
  {
    Mat upper_level;
    pyrUp(i == pyramid.size() - 2 ? pyramid.back() : current, upper_level);
    std::cout << "pyrUp complete, i = " << i << std::endl;
    current = pyramid[i] + upper_level;
    std::cout << "Done adding images " << i << std::endl;

    double minVal, maxVal;
    minMaxLoc(pyramid[i], &minVal, &maxVal);
    std::cout << "pyramid min/max value: " << minVal << " " << maxVal << std::endl;

    minMaxLoc(current, &minVal, &maxVal);
    std::cout << "current min/max value: " << minVal << " " << maxVal << std::endl;
  }

  std::cout << "Creating dst..." << std::endl;
  dst = Mat(current.size(), current.type());
  std::cout << "Copying to dst..." << std::endl;
  current.copyTo(dst);
  std::cout << "Complete!" << std::endl;
}

void stitch(const Mat& img1, const Mat& coeff1,
            const Mat& img2, const Mat& coeff2,
            int pyr_levels, Mat& output, bool verbose = false)
{
  // convert input images to s16
  Mat img1s16(img1.size(), CV_16SC3), img2s16(img2.size(), CV_16SC3);
  img1.convertTo(img1s16, CV_16SC3);
  img2.convertTo(img2s16, CV_16SC3);

  // convert weight coeffs to color
  Mat coeff1c3(coeff1.size(), CV_16SC3), coeff2c3(coeff1.size(), CV_16SC3);
  Mat _coeff1c3[3] = {coeff1, coeff1, coeff1};
  Mat _coeff2c3[3] = {coeff2, coeff2, coeff2};
  merge(_coeff1c3, 3, coeff1c3);
  merge(_coeff2c3, 3, coeff2c3);

  // create laplacian pyramids from images, and gaussian pyramids from coeffs
  std::vector<Mat> pyr_img1, pyr_coeff1, pyr_img2, pyr_coeff2;
  pyr_img1.resize(pyr_levels);
  pyr_coeff1.resize(pyr_levels);
  pyr_img2.resize(pyr_levels);
  pyr_coeff2.resize(pyr_levels);

  std::cout << "starting to build pyramids" << std::endl;

  double minVal, maxVal;
  minMaxLoc(coeff1, &minVal, &maxVal);
  std::cout << "coeff1 min/max value: " << minVal << " " << maxVal << std::endl;

  laplacianPyramid(img1s16, pyr_img1);
  laplacianPyramid(img2s16, pyr_img2);
  gaussianPyramid(coeff1c3, pyr_coeff1);
  gaussianPyramid(coeff2c3, pyr_coeff2);

  std::cout << "built pyramids" << std::endl;

  // multi-band blending
  std::vector<Mat> pyr_output(pyr_img1.size());
  for(int i = 0; i < pyr_levels; i++)
  {
    // blur the blend masks
    GaussianBlur(pyr_coeff1[i], pyr_coeff1[i], Size(5, 5), 0.0);
    GaussianBlur(pyr_coeff2[i], pyr_coeff2[i], Size(5, 5), 0.0);

    Mat coeff_sum = pyr_coeff1[i] + pyr_coeff2[i];

    // compute the stitched image
    Mat stitch1(pyr_img1[i].size(), CV_16SC3),
        stitch2(pyr_img1[i].size(), CV_16SC3);
    pyr_output[i] = Mat(pyr_img1[i].size(), CV_16SC3);

    multiply(pyr_img1[i], pyr_coeff1[i], stitch1, 1.0f/(1<<12));
    multiply(pyr_img2[i], pyr_coeff2[i], stitch2, 1.0f/(1<<12));
    add(stitch1, stitch2, pyr_output[i]);

    std::cout << "Level " << i << "min/max:" << std::endl;
    double minVal, maxVal;
    minMaxLoc(pyr_img1[i], &minVal, &maxVal);
    std::cout << "pyr_img1: " << minVal << " " << maxVal << std::endl;
    minMaxLoc(pyr_img2[i], &minVal, &maxVal);
    std::cout << "pyr_img2: " << minVal << " " << maxVal << std::endl;
    minMaxLoc(pyr_output[i], &minVal, &maxVal);
    std::cout << "pyr_output: " << minVal << " " << maxVal << std::endl;
    minMaxLoc(pyr_coeff1[i], &minVal, &maxVal);
    std::cout << "pyr_coeff1: " << minVal << " " << maxVal << std::endl;
    minMaxLoc(pyr_coeff2[i], &minVal, &maxVal);
    std::cout << "pyr_coeff2: " << minVal << " " << maxVal << std::endl;

    divide(pyr_output[i], coeff_sum, pyr_output[i], 1<<12);
  }

  std::cout << "merge complete" << std::endl;

  // restore the output image from a laplacian pyramid
  Mat outputs16;
  imageFromLaplacianPyramid(pyr_output, outputs16);

  minMaxLoc(outputs16, &minVal, &maxVal);
  std::cout << "outputs16 min/max value: " << minVal << " " << maxVal <<
    std::endl;

  std::cout << "laplacian reconstruction complete" << std::endl;
  outputs16.convertTo(output, CV_8UC3);

  if(verbose)
  {
    imwrite("stitched.jpg", output);
  }

  std::cout << "stitch complete" << std::endl;
}

void computeStitchParams(const Mat& img1, const Mat& img2,
      Mat& homography, Rect& bounds, bool verbose = false)
{
  Ptr<FastFeatureDetector> detector = FastFeatureDetector::create(50);

  std::vector<KeyPoint> keypoints1, keypoints2;

  detector->detect( img1, keypoints1 );
  detector->detect( img2, keypoints2 );

  if(verbose)
  {
    std::cout << "Found " << (int)keypoints1.size() << " points" << std::endl;
    std::cout << "Found " << (int)keypoints2.size() << " points" << std::endl;
  }

  //-- Step 2: Calculate descriptors (feature vectors)
  Ptr<ORB> extractor = ORB::create();

  Mat descriptors1, descriptors2;

  extractor->compute( img1, keypoints1, descriptors1 );
  extractor->compute( img2, keypoints2, descriptors2 );

  //-- Step 3: Matching descriptor vectors using FLANN matcher
  BFMatcher matcher(NORM_L2);
  std::vector< DMatch > matches;
  matcher.match( descriptors1, descriptors2, matches );

  double max_dist = 0; double min_dist = 100;

  //-- Quick calculation of max and min distances between keypoints
  for( int i = 0; i < descriptors1.rows; i++ )
  { double dist = matches[i].distance;
    if( dist < min_dist ) min_dist = dist;
    if( dist > max_dist ) max_dist = dist;
  }

  if(verbose)
  {
    printf("-- Max dist : %f \n", max_dist);
    printf("-- Min dist : %f \n", min_dist);
  }

  //-- Draw only "good" matches (i.e. whose distance is less than 3*min_dist )
  std::vector< DMatch > good_matches;
  float dist = max(100*min_dist, max_dist*0.1);

  if(verbose)
    std::cout << "dist = " << dist << std::endl;

  for( int i = 0; i < descriptors1.rows; i++ )
  { if( matches[i].distance < dist )
    { good_matches.push_back( matches[i]); }
  }

  if(verbose)
    std::cout << "Found " << (int)good_matches.size() <<
      " good matches" << std::endl;

  //-- Localize the object from img_1 in img_2
  std::vector<Point2f> points1;
  std::vector<Point2f> points2;

  for( size_t i = 0; i < good_matches.size(); i++ )
  {
    //-- Get the keypoints from the good matches
    points1.push_back( keypoints1[ good_matches[i].queryIdx ].pt );
    points2.push_back( keypoints2[ good_matches[i].trainIdx ].pt );
  }

#if CV_MAJOR_VERSION < 4
  Mat H = findHomography( points1, points2, CV_RANSAC );
#else
  Mat H = findHomography( points1, points2, RANSAC );
#endif
  std::vector<Point2f> points1_mapped;
  perspectiveTransform(points1, points1_mapped, H);

  // filter outliers
  const float max_inlier_dist = 5.0f;
  std::vector<Point2f> inliers1, inliers2;
  for(size_t i = 0; i < points1.size(); i++)
  {
    Point2f diff = points1_mapped[i] - points2[i];
    float dist2 = diff.dot(diff);
    if(dist2 < max_inlier_dist*max_inlier_dist)
    {
      // add to inliers
      inliers1.push_back(points1[i]);
      inliers2.push_back(points2[i]);
    }
  }

  Mat img_matches;
  drawMatches2(img1, inliers1, img2, inliers2, img_matches);

  //-- Get the corners from the image_1 ( the object to be "detected" )
  std::vector<Point2f> corners1(4);
  corners1[0] = Point(0,0); corners1[1] = Point( img1.cols, 0 );
  corners1[2] = Point( img1.cols, img1.rows ); corners1[3] = Point( 0, img1.rows );
  std::vector<Point2f> corners2(4);

  perspectiveTransform(corners1, corners2, H);

  int xmin = 0, xmax = -1, ymin = 0, ymax = -1;
  for(int i = 0; i < 4; i++)
  {
    xmin = min(xmin, min(int(corners1[i].x), int(corners2[i].x)));
    xmax = max(xmax, max(int(corners1[i].x), int(corners2[i].x)));
    ymin = min(ymin, min(int(corners1[i].y), int(corners2[i].y)));
    ymax = max(ymax, max(int(corners1[i].y), int(corners2[i].y)));
  }

  bounds = Rect(xmin, ymin, xmax - xmin, ymax - ymin);

  if(verbose)
  {
    std::cout << "scene_inliers " << inliers2[0] << std::endl;
    std::cout << "xmin = " << xmin << ", ymin = " << ymin << std::endl;
  }

  Mat offsetH = Mat::eye(3, 3, CV_64F);
  offsetH.at<double>(0, 2) = -bounds.x;
  offsetH.at<double>(1, 2) = -bounds.y;

  homography = offsetH*H;

  if(verbose)
  {
    std::cout << "offsetH = " << offsetH << std::endl;
    std::cout << "final homography " << homography << std::endl;
    //-- Show detected matches
    imshow( "Good Matches & Object detection", img_matches );
  }
}

void computeHomographyRemap(const Size& source_size,
  const Size& dest_size, const Mat& homography, Mat& map)
{
  map = Mat(dest_size, CV_32FC2);

  Mat homography_inv = homography.inv();

  std::vector<Point2f> dest_points;
  for(int y = 0; y < dest_size.height; y++)
  {
    for(int x = 0; x < dest_size.width; x++)
    {
      dest_points.push_back(Point2f((float)x, (float)y));
    }
  }

  std::vector<Point2f> source_points;
  perspectiveTransform(dest_points, source_points, homography_inv);

  for(size_t i = 0; i < source_points.size(); i++)
  {
    int x = (int)dest_points[i].x;
    int y = (int)dest_points[i].y;
    map.at<Point2f>(y, x) = source_points[i];
  }
}

void computePanRemap(const Size& source_size, const Size& dest_size,
  const Rect& roi, Mat& map)
{
  map = Mat(dest_size, CV_32FC2);
  map = Scalar(-1, -1);

  for(int y = 0; y < roi.height; y++)
  {
    for(int x = 0; x < roi.width; x++)
    {
      map.at<Point2f>(y + roi.y, x + roi.x) =
        Point2f(x + 0.5f, y + 0.5f);
    }
  }
}

/**
 * @function main
 * @brief Main function
 */
int main( int argc, char** argv )
{
  if( argc < 6 || argc > 7)
  {
    readme();
    return -1;
  }

  bool verbose = false;
  if(argc == 6)
  {
    if(strcmp(argv[5], "1") == 0)
    {
      verbose = true;
    }
  }

  const char* output_filename = argv[3];
  const char* stitch_filename = argv[4];

  Mat img1_color = imread( argv[1]);
  Mat img2_color = imread( argv[2]);

  Mat img1, img2;

#if CV_MAJOR_VERSION < 4
  cvtColor(img1_color, img1, CV_RGB2GRAY);
  cvtColor(img2_color, img2, CV_RGB2GRAY);
#else
  cvtColor(img1_color, img1, COLOR_RGB2GRAY);
  cvtColor(img2_color, img2, COLOR_RGB2GRAY);
#endif

  if( !img1.data || !img2.data )
  {
    printf(" --(!) Error reading images \n");
    return -1;
  }

  Mat homography;
  Rect bounds;
  printf("Starting image registration...\n");
  computeStitchParams(img1, img2, homography, bounds, verbose);

  // make bounds dividable by 32
  bounds.width = (bounds.width/32 + 1)*32;
  bounds.height = (bounds.height/32 + 1)*32;

  Mat weights1, weights2;
  generateWeightImage(img1.size(), weights1);
  generateWeightImage(img2.size(), weights2);

  Mat img_stitched(bounds.height, bounds.width, CV_8UC1);

  Mat coeffs1 = Mat::zeros(img_stitched.size(), CV_32FC1);
  Mat coeffs2 = Mat::zeros(img_stitched.size(), CV_32FC1);

  Mat _img1 = Mat::zeros(img_stitched.size(), CV_8UC3),
      _img2 = Mat::zeros(img_stitched.size(), CV_8UC3);

#if 0
  warpPerspective(img_object_color, img1, homography, img1.size());
  Mat img_roi(img2, roi);
  img_scene_color.copyTo(img_roi);

  // make a white image of the size of obj
  warpPerspective(weights_object, coeffs_object, homography, img_stitched.size(), INTER_NEAREST);
  Mat scene_roi(coeffs_scene, roi);
  weights_scene.copyTo(scene_roi);
#else
  Rect roi = Rect(-bounds.x, -bounds.y, img2.cols, img2.rows);
  Mat remap_homography, remap_pan;
  computeHomographyRemap(img1.size(), img_stitched.size(),
      homography, remap_homography);
  computePanRemap(img2.size(), img_stitched.size(), roi,
      remap_pan);

  remap(img1_color, _img1, remap_homography, Mat(),
    INTER_LINEAR, BORDER_TRANSPARENT);
  remap(img2_color, _img2, remap_pan, Mat(),
    INTER_LINEAR, BORDER_TRANSPARENT);
  remap(weights1, coeffs1, remap_homography, Mat(),
    INTER_LINEAR, BORDER_TRANSPARENT);
  remap(weights2, coeffs2, remap_pan, Mat(),
    INTER_LINEAR, BORDER_TRANSPARENT);

  // convert the blending coefficients to binary
  Mat mask = coeffs1 > coeffs2;
  Mat maskf;
  mask.convertTo(maskf, CV_32FC1, 1.0/256);
  std::cout << "Running bitwise and" << std::endl;
  std::cout << "mask size " << mask.size() << std::endl;
  std::cout << "coeffs1 size " << coeffs1.size() << std::endl;
  bitwise_and(coeffs1, maskf, coeffs1);
  bitwise_not(maskf, maskf);
  bitwise_and(coeffs2, maskf, coeffs2);
  std::cout << "Running bitwise and 2" << std::endl;

  // normalize to account for overlapping image areas
  Mat coeff_total = coeffs1 + coeffs2;
  coeff_total = coeff_total + FLT_MIN; // avoid overflow
  divide(coeffs1, coeff_total, coeffs1);
  divide(coeffs2, coeff_total, coeffs2);

  FileStorage fs(stitch_filename, FileStorage::WRITE);
  fs << "remap1" << remap_homography;
  fs << "remap1_src_width" << img1_color.cols;
  fs << "remap1_src_height" << img1_color.rows;
  fs << "remap1_dst_width" << _img1.cols;
  fs << "remap1_dst_height" << _img1.rows;

  fs << "remap2" << remap_pan;
  fs << "remap2_src_width" << img2.cols;
  fs << "remap2_src_height" << img2.rows;
  fs << "remap2_dst_width" << _img2.cols;
  fs << "remap2_dst_height" << _img2.rows;

  Mat coeffs1_s16, coeffs2_s16;
  coeffs1.convertTo(coeffs1_s16, CV_16SC1, (float)(1<<12));
  fs << "coeffs1" << coeffs1_s16;

  coeffs2.convertTo(coeffs2_s16, CV_16SC1, (float)(1<<12));
  fs << "coeffs2" << coeffs2_s16;
#endif

  const int pyr_levels = 4;
  stitch(_img1, coeffs1_s16, _img2, coeffs2_s16, pyr_levels, img_stitched, verbose);

  imwrite(output_filename, img_stitched);

  if(verbose)
  {
    imshow_ex("obj", coeffs1, 640);
    imshow_ex("scene", coeffs2, 640);
    imshow_ex("Stitched image", img_stitched, 640);
    waitKey(0);
  }


  return 0;
}

/**
 * @function readme
 */
void readme()
{
  printf(" Usage: ./homography <img1> <img2> <output> \
    <stitch.xml> [verbose]\n");
  printf("   verbose can be 0 or 1\n");
}
