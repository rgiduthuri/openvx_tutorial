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
 * \file    homography-opencv.cpp
 * \example homography-opencv
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

void stitch(const Mat& img1, const Mat& coeff1,
            const Mat& img2, const Mat& coeff2,
            float blend_width, Mat& output, bool verbose = false)
{
  Mat coeff_sum = coeff1 + coeff2;

  if(verbose)
  {
    Mat coeff_sum8u;
    coeff_sum.convertTo(coeff_sum8u, CV_8UC1, 255);
    imwrite("coeff_sum.jpg", coeff_sum8u);
  }

  // convert to color
  Mat coeff1c3(coeff1.size(), CV_32FC3), coeff2c3(coeff2.size(), CV_32FC3);

#if CV_MAJOR_VERSION < 4
  cvtColor(coeff1, coeff1c3, CV_GRAY2RGB);
  cvtColor(coeff2, coeff2c3, CV_GRAY2RGB);
#else
cvtColor(coeff1, coeff1c3, COLOR_GRAY2RGB);
cvtColor(coeff2, coeff2c3, COLOR_GRAY2RGB);
#endif

  // convert input images to 32f for blending
  Mat img1f32(img1.size(), CV_32FC3), img2f32(img2.size(), CV_32FC3);
  img1.convertTo(img1f32, CV_32FC3);
  img2.convertTo(img2f32, CV_32FC3);

  // compute the stitched image
  Mat stitch1(img1.size(), CV_32FC3),
      stitch2(img1.size(), CV_32FC3),
      outputf32(img1.size(), CV_32FC3);

  multiply(img1f32, coeff1c3, stitch1);
  multiply(img2f32, coeff2c3, stitch2);
  add(stitch1, stitch2, outputf32);
  outputf32.convertTo(output, CV_8UC3);

  if(verbose)
  {
    Mat stitch18u, stitch28u;
    stitch1.convertTo(stitch18u, CV_8UC3, 1);
    stitch2.convertTo(stitch28u, CV_8UC3, 1);
    imwrite("stitch1.jpg", stitch18u);
    imwrite("stitch2.jpg", stitch28u);
    imwrite("stitched.jpg", output);
  }

  if(verbose)
  {
    std::cout << "(528,528) img1: " << img1f32.at<Point3f>(528, 528) <<
      ", coeff1 = " << coeff1.at<Point3f>(528, 528) << std::endl;
    std::cout << "(528,528) img2: " << img2f32.at<Point3f>(528, 528) <<
      ", coeff1 = " << coeff2.at<Point3f>(528, 528) << std::endl;
    std::cout << "stitch1: " << stitch1.at<Point3f>(528, 528) << std::endl;
    std::cout << "stitch2: " << stitch2.at<Point3f>(528, 528) << std::endl;
  }
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

  // normalize to account for overlapping image areas
  Mat coeff_total = max(coeffs1 + coeffs2, FLT_MIN);
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
  stitch(_img1, coeffs1, _img2, coeffs2, 100, img_stitched, verbose);

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
