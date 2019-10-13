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
 * \brief   This OpenVX sample finds road lanes in an input image using Hough
 * Transform, and then detects their cross point (the vanishing point). It uses
 * the position of the vanishing point and the lanes to find the homography
 * transformation that computes a bird's eye view from the input image.
 * \author  Victor Erukhimov <relrotciv@gmail.com>
 */

#include <math.h>
#include <float.h>
#include <VX/vx.h>
#include <vxa/vxa.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include "readImage.h"
#include "writeImage.h"

#define ERROR_CHECK_STATUS( status ) { \
        vx_status status_ = (status); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }

#define ERROR_CHECK_OBJECT( obj ) { \
        vx_status status_ = vxGetStatus((vx_reference)(obj)); \
        if(status_ != VX_SUCCESS) { \
            printf("ERROR: failed with status = (%d) at " __FILE__ "#%d\n", status_, __LINE__); \
            exit(1); \
        } \
    }

// LU decomoposition of a general matrix
void sgetrf_(int* M, int *N, float* A, int* lda, int* IPIV, int* INFO);

// generate inverse of a matrix given its LU decomposition
void sgetri_(int* N, float* A, int* lda, int* IPIV, float* WORK, int* lwork, int* INFO);

//GEMM
void sgemm_(char* TRANSA, char* TRANSB, const int* M, const int* N, const int* K,
            float* alpha, const float* A, const int* LDA, const float* B, const int* LDB,
            const float* beta, float* C, const int* LDC);

// multiply matrix by vector
void sgemv_(const char* trans, const int* m, const int* n, const float* alpha,
  const float* A, const int* lda, const float* B, const int* incx,
  const float* beta, float* C, const int* incy);


const vx_size max_num_lines = 2000;
vx_uint32 widthr, heightr;
vx_image test;
vx_matrix test_matrix;

const float scale_factor = 4.0f;


enum user_library_e
{
    USER_LIBRARY_EXAMPLE        = 1,
};
enum user_kernel_e
{
    USER_KERNEL_FILTER_LINES     = VX_KERNEL_BASE( VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE ) + 0x001,
    USER_KERNEL_VANISHING_POINTS     = VX_KERNEL_BASE( VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE ) + 0x002,
    USER_KERNEL_BIRDSEYE_TRANSFORM = VX_KERNEL_BASE( VX_ID_DEFAULT, USER_LIBRARY_EXAMPLE ) + 0x003,
};

vx_node userFilterLinesNode(vx_graph graph,
                           vx_array input,
                           vx_array output)
{
    vx_context context = vxGetContext( ( vx_reference ) graph );
    vx_kernel kernel = vxGetKernelByEnum( context, USER_KERNEL_FILTER_LINES);
    ERROR_CHECK_OBJECT( kernel );
    vx_node node       = vxCreateGenericNode( graph, kernel );
    ERROR_CHECK_OBJECT( node );

    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 0, ( vx_reference ) input ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 1, ( vx_reference ) output ) );

    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );

    return node;
}

vx_node userFindVanishingPoint(vx_graph graph,
                           vx_array input,
                           vx_array output)
{
    vx_context context = vxGetContext( ( vx_reference ) graph );
    vx_kernel kernel = vxGetKernelByEnum( context, USER_KERNEL_VANISHING_POINTS);
    ERROR_CHECK_OBJECT( kernel );
    vx_node node       = vxCreateGenericNode( graph, kernel );
    ERROR_CHECK_OBJECT( node );

    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 0, ( vx_reference ) input ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 1, ( vx_reference ) output ) );

    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );

    return node;
}

vx_node userComputeBirdsEyeTransform(vx_graph graph,
                           vx_array input,
                           vx_image image,
                           vx_matrix perspective)
{
    vx_context context = vxGetContext( ( vx_reference ) graph );
    vx_kernel kernel = vxGetKernelByEnum( context, USER_KERNEL_BIRDSEYE_TRANSFORM);
    ERROR_CHECK_OBJECT( kernel );
    vx_node node       = vxCreateGenericNode( graph, kernel );
    ERROR_CHECK_OBJECT( node );

    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 0, ( vx_reference ) input ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 1, ( vx_reference ) image ) );
    ERROR_CHECK_STATUS( vxSetParameterByIndex( node, 2, ( vx_reference ) perspective ) );

    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );

    return node;
}

vx_status VX_CALLBACK filter_lines_validator( vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[] )
{
    // parameter #0 -- check array type
    vx_enum param_type;
    ERROR_CHECK_STATUS( vxQueryArray( ( vx_array )parameters[0], VX_ARRAY_ITEMTYPE, &param_type, sizeof( param_type ) ) );
    if(param_type != VX_TYPE_LINE_2D) // check that the array contains lines
    {
        return VX_ERROR_INVALID_TYPE;
    }

    // parameter #1 -- check array type
    ERROR_CHECK_STATUS( vxQueryArray( ( vx_array )parameters[1], VX_ARRAY_ITEMTYPE, &param_type, sizeof( param_type ) ) );
    if(param_type != VX_TYPE_LINE_2D)
    {
        return VX_ERROR_INVALID_TYPE;
    }

    // set output metadata
    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( metas[1], VX_ARRAY_ITEMTYPE, &param_type, sizeof( param_type ) ) );

    return VX_SUCCESS;
}

vx_status VX_CALLBACK vanishing_point_validator( vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[] )
{
    // parameter #0 -- check array type
    vx_enum param_type;
    ERROR_CHECK_STATUS( vxQueryArray( ( vx_array )parameters[0], VX_ARRAY_ITEMTYPE, &param_type, sizeof( param_type ) ) );
    if(param_type != VX_TYPE_LINE_2D) // check that the array contains lines
    {
        return VX_ERROR_INVALID_TYPE;
    }

    // parameter #1 -- check that the scalar is coordinates2d
    ERROR_CHECK_STATUS( vxQueryArray( ( vx_array )parameters[1], VX_ARRAY_ITEMTYPE, &param_type, sizeof( param_type ) ) );
    if(param_type != VX_TYPE_COORDINATES2D)
    {
        return VX_ERROR_INVALID_TYPE;
    }

    // set output metadata
    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( metas[1], VX_ARRAY_ITEMTYPE, &param_type, sizeof( param_type ) ) );

    return VX_SUCCESS;
}

vx_status VX_CALLBACK birdseye_transform_validator( vx_node node, const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[] )
{
    // parameter #0 -- check array type
    vx_enum param_type;
    ERROR_CHECK_STATUS( vxQueryArray( ( vx_array )parameters[0], VX_ARRAY_ITEMTYPE, &param_type, sizeof( param_type ) ) );
    if(param_type != VX_TYPE_COORDINATES2D) // check that the array contains lines
    {
        return VX_ERROR_INVALID_TYPE;
    }

    vx_uint32 width;
    ERROR_CHECK_STATUS(vxQueryImage((vx_image)parameters[1], VX_IMAGE_WIDTH, &width, sizeof(width)));
    if(width <= 0)
    {
        return VX_ERROR_INVALID_DIMENSION;
    }

    // parameter #2 -- check that this is a floating point 3x3 matrices
    ERROR_CHECK_STATUS( vxQueryMatrix( ( vx_matrix )parameters[2], VX_MATRIX_TYPE, &param_type, sizeof( param_type ) ) );
    if(param_type != VX_TYPE_FLOAT32)
    {
        return VX_ERROR_INVALID_TYPE;
    }

    vx_size rows, columns;
    ERROR_CHECK_STATUS( vxQueryMatrix( ( vx_matrix )parameters[2], VX_MATRIX_ROWS, &rows, sizeof( rows ) ) );
    if(rows != 3)
    {
        return VX_ERROR_INVALID_DIMENSION;
    }

    ERROR_CHECK_STATUS( vxQueryMatrix( ( vx_matrix )parameters[2], VX_MATRIX_COLUMNS, &columns, sizeof( columns ) ) );
    if(columns != 3)
    {
        return VX_ERROR_INVALID_DIMENSION;
    }

    // set output metadata
    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( metas[2], VX_MATRIX_TYPE, &param_type, sizeof( param_type ) ) );
    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( metas[2], VX_MATRIX_ROWS, &rows, sizeof( rows ) ) );
    ERROR_CHECK_STATUS( vxSetMetaFormatAttribute( metas[2], VX_MATRIX_COLUMNS, &columns, sizeof( columns ) ) );

    return VX_SUCCESS;
}

vx_status VX_CALLBACK filter_lines_calc_function( vx_node node, const vx_reference * refs, vx_uint32 num )
{
  vx_array lines = (vx_array) refs[0];
  vx_array lines_output = (vx_array) refs[1];

  vx_size num_lines = -1;
  ERROR_CHECK_STATUS(vxQueryArray(lines, VX_ARRAY_NUMITEMS, &num_lines, sizeof(num_lines)));

  char* __lines = NULL;
  vx_map_id map_id;
  vx_size stride = sizeof(vx_line2d_t);
  vxMapArrayRange(lines, 0, num_lines, &map_id, &stride, (void**)&__lines,
    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

  vx_line2d_t _lines_filtered[max_num_lines];
  vx_size _num_lines_filtered = 0;
  const float max_ratio = 0.1;
  for(int i = 0; i < num_lines; i++, __lines += stride)
  {
    vx_line2d_t* _line = (vx_line2d_t*)__lines;
    int dx = _line->end_x - _line->start_x;
    int dy = _line->end_y - _line->start_y;

    if(_line->start_y < heightr/2 || _line->end_y < heightr/2)
    {
      continue;
    }
    if(abs(dy) < max_ratio*abs(dx))
    {
      continue;
    }

    memcpy(&_lines_filtered[_num_lines_filtered++], _line,
      sizeof(vx_line2d_t));
  }

  vxUnmapArrayRange(lines, map_id);

  vxAddArrayItems(lines_output, _num_lines_filtered, _lines_filtered,
    sizeof(vx_line2d_t));

  return(VX_SUCCESS);
}

void find_cross_point(const float* line1, const float* line2,
  float* cross_point)
{
  cross_point[0] = line1[1]*line2[2] - line1[2]*line2[1];
  cross_point[1] = line1[2]*line2[0] - line1[0]*line2[2];
  cross_point[2] = line1[0]*line2[1] - line1[1]*line2[0];
}

vx_status VX_CALLBACK vanishing_point_calc_function( vx_node node, const vx_reference * refs, vx_uint32 num )
{
  vx_array lines = (vx_array)refs[0];
  vx_array vanishing_points = (vx_array)refs[1];

  vx_size num_lines = -1;
  ERROR_CHECK_STATUS(vxQueryArray(lines, VX_ARRAY_NUMITEMS, &num_lines, sizeof(num_lines)));

  char* __lines = 0;
  vx_size stride = sizeof(vx_line2d_t);
  vx_map_id map_id;

  vxMapArrayRange(lines, 0, num_lines, &map_id, &stride, (void**)&__lines,
    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

  float lines_uniform[max_num_lines][3];
  for(int i = 0; i < num_lines; i++, __lines += stride)
  {
    vx_line2d_t* _line = (vx_line2d_t*)__lines;
    float x0 = _line->start_x;
    float y0 = _line->start_y;
    float dx = _line->end_x - _line->start_x;
    float dy = _line->end_y - _line->start_y;

    lines_uniform[i][0] = dy;
    lines_uniform[i][1] = -dx;
    lines_uniform[i][2] = -x0*dy + y0*dx;
  }

  vxUnmapArrayRange(lines, map_id);

  vx_coordinates2d_t avg_cross_point = {0.0, 0.0};
  int count = 0;
  for(int i = 0; i < num_lines; i++)
  {
    for(int j = 0; j < num_lines; j++)
    {
      float cross_point[3];
      find_cross_point(lines_uniform[i], lines_uniform[j], cross_point);
      if(fabs(cross_point[2]) < FLT_MIN)
      {
        // filter this vanishing point
        continue;
      }
      float cx = cross_point[0]/cross_point[2];
      float cy = cross_point[1]/cross_point[2];

      if(cx < 0 || cy < 0 || cx > widthr || cy > heightr)
      {
        // we know the cross point lies inside an image, so this is an outlier
        continue;
      }

      avg_cross_point.x += (int)cx;
      avg_cross_point.y += (int)cy;
      count++;
    }
  }

  avg_cross_point.x /= count;
  avg_cross_point.y /= count;

  vxAddArrayItems(vanishing_points, 1, &avg_cross_point, sizeof(avg_cross_point));

  return(VX_SUCCESS);
}

#if 0
void calc_homography(float* h, float* p0, float* p1)
{
  float x = p0[0];
  float y = p0[1];

  float x1 = h[0]*x + h[3]*y + h[6];
  float y1 = h[1]*x + h[4]*y + h[7];
  float z1 = h[2]*x + h[5]*y + h[8];

  p1[0] = x1/z1;
  p1[1] = y1/z1;
}
#else
void calc_homography(float* h, float* p0, float* p1)
{
  float x = p0[0];
  float y = p0[1];

  float x1 = h[0]*x + h[1]*y + h[2];
  float y1 = h[3]*x + h[4]*y + h[5];
  float z1 = h[6]*x + h[7]*y + h[8];

  p1[0] = x1/z1;
  p1[1] = y1/z1;
}
#endif

void transpose(float* C)
{
#define exchange(a, b) {float _c;_c=(a);(a)=(b);(b)=_c;}

  exchange(C[1], C[3]);
  exchange(C[2], C[6]);
  exchange(C[5], C[7]);
}

void mult_3x3matrices(const float* A, const float* B, float* C)
{
  char transa = 't', transb = 't';
  int m = 3, n = 3, k = 3;
  float alpha = 1.0f, beta = 0.0f;

  // initialize output with zeros
  for(int i = 0; i < 9; i++)
    C[i] = 0.0f;
  sgemm_(&transa, &transb, &m, &n, &k, &alpha, A, &m, B, &k, &beta, C, &m);
  transpose(C);
}

/*
// multiply matrix by vector
void mult_3x3mv(const float* A, const float* B, float* C)
{
  char trans = 'n';
  int m = 3, n = 3, incx = 1, incy = 1;
  float alpha = 1.0f, beta = 0.0f;

  // initialize output with zeros
  for(int i = 0; i < 3; i++)
    C[i] = 0.0f;

  sgemv_(&trans, &m, &n, &alpha, A, &m, B, &incx, &beta, C, &incy);
}
*/

void calc_inverse_3x3matrix(const float* m, float* invm)
{
  int n = 3;
  int indices[4];
  int lwork = 9;
  float work[lwork];
  int info;

  memcpy(invm, m, 9*sizeof(float));

  sgetrf_(&n, &n, invm, &n, indices, &info);
  sgetri_(&n, invm, &n, indices, work, &lwork, &info);
}


vx_status VX_CALLBACK birdseye_transform_calc_function( vx_node node, const vx_reference * refs, vx_uint32 num )
{
  vx_array points = (vx_array)refs[0];
  vx_image image = (vx_image)refs[1];
  vx_matrix perspective = (vx_matrix)refs[2];

  // get image height
  vx_uint32 image_width, image_height;
  ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_WIDTH, &image_width, sizeof(image_width)));
  ERROR_CHECK_STATUS(vxQueryImage(image, VX_IMAGE_HEIGHT, &image_height, sizeof(image_height)));

  // intrinsic parameters
  float _K[9] = {8.4026236186715255e+02*4, 0., 3.7724917600845038e+02*4,
                              0., 8.3752885759166338e+02*4, 4.6712164335800873e+02*4,
                              0., 0., 1.};

  // calculate intrinsics inverse
  float _Kinv[9];
  calc_inverse_3x3matrix(_K, _Kinv);

  // obtain the vanishing point
  const int num_points = 1;
  vx_coordinates2d_t* _points = 0;
  vx_size stride = sizeof(vx_coordinates2d_t);
  vx_map_id map_id;
  vxMapArrayRange(points, 0, num_points, &map_id, &stride, (void**)&_points,
    VX_READ_ONLY, VX_MEMORY_TYPE_HOST, 0);

  // generate the vanishing point in uniform coordinates
  float pv[] = {_points[0].x*scale_factor, _points[0].y*scale_factor};
  float pvu[2];
  calc_homography(_Kinv, pv, pvu);
  float yv = pvu[1];

  // generate a homography that sends the vanishing point to infinity
  float phi = atan(1/yv);
  float _rotate[9] = {1.0f, 0.0f, 0.0f,
                          0.0f, -cos(phi), -sin(phi),
                          0.0f, sin(phi), -cos(phi)};

  // generate birds eye view homography
  float _temp[9], _perspective[9];
  mult_3x3matrices(_K, _rotate, _temp);
  mult_3x3matrices(_temp, _Kinv, _perspective);

  // now map two control points using the perspective matrix,
  // to adjust scale and translation
  float upper_boundary_factor = 1.2f;
  float control1[2] = {pv[0], pv[1]*upper_boundary_factor};
  float control2[2] = {pv[0], image_height};

  float control1_mapped[2], control2_mapped[2];
  calc_homography(_perspective, control1, control1_mapped);
  calc_homography(_perspective, control2, control2_mapped);

  // find y coordinates of the mapped points from the uniform coordinates
  float y1 = control1_mapped[1];
  float y2 = control2_mapped[1];

  // now define additional translation and scale to have the control points
  // mapped to the upper and lower boundary of the output image
  float scale = ((float)y2 - y1)/image_height;
  float _panzoom[] = {1.0f, 0.0f, image_width*scale/2 - pv[0],
                      0.0f, 1.0f, -y1,
                      0.0f, 0.0f, scale};

  // now create the final perspective transformation by multiplying
  // _perspective by _panzoom from the left
  float _perspective_final[9];
  mult_3x3matrices(_panzoom, _perspective, _perspective_final);

  // now we need to invert and transpose the homography for OpenVX
  float _perspective_final_inv[9];
  calc_inverse_3x3matrix(_perspective_final, _perspective_final_inv);
  transpose(_perspective_final_inv);

  vxCopyMatrix(perspective, _perspective_final_inv, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

  return VX_SUCCESS;
}

vx_status registerUserFilterLinesKernel( vx_context context )
{
    vx_kernel kernel = vxAddUserKernel( context,
                                    "app.userkernels.filter_lines",
                                    USER_KERNEL_FILTER_LINES,
                                    filter_lines_calc_function,
                                    2,   // numParams
                                    filter_lines_validator,
                                    NULL,
                                    NULL );
    ERROR_CHECK_OBJECT( kernel );

    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 0, VX_INPUT,  VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED ) ); // input
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 1, VX_OUTPUT, VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED ) ); // output
    ERROR_CHECK_STATUS( vxFinalizeKernel( kernel ) );
    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );

    vxAddLogEntry( ( vx_reference ) context, VX_SUCCESS, "OK: registered user kernel app.userkernels.filter_lines\n" );
    return VX_SUCCESS;
}

vx_status registerUserVanishingPointKernel( vx_context context )
{
    vx_kernel kernel = vxAddUserKernel( context,
                                    "app.userkernels.vanishing_point",
                                    USER_KERNEL_VANISHING_POINTS,
                                    vanishing_point_calc_function,
                                    2,   // numParams
                                    vanishing_point_validator,
                                    NULL,
                                    NULL );
    ERROR_CHECK_OBJECT( kernel );

    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 0, VX_INPUT,  VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED ) ); // input
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 1, VX_OUTPUT, VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED ) ); // output
    ERROR_CHECK_STATUS( vxFinalizeKernel( kernel ) );
    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );

    vxAddLogEntry( ( vx_reference ) context, VX_SUCCESS, "OK: registered user kernel app.userkernels.vanishing_point\n" );
    return VX_SUCCESS;
}

vx_status registerUserBirdsEyeTransformKernel( vx_context context )
{
    vx_kernel kernel = vxAddUserKernel( context,
                                    "app.userkernels.birdseye_transform",
                                    USER_KERNEL_BIRDSEYE_TRANSFORM,
                                    birdseye_transform_calc_function,
                                    3,   // numParams
                                    birdseye_transform_validator,
                                    NULL,
                                    NULL );
    ERROR_CHECK_OBJECT( kernel );

    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 0, VX_INPUT,  VX_TYPE_ARRAY,  VX_PARAMETER_STATE_REQUIRED ) ); // input
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 1, VX_INPUT, VX_TYPE_IMAGE,  VX_PARAMETER_STATE_REQUIRED ) ); // input
    ERROR_CHECK_STATUS( vxAddParameterToKernel( kernel, 2, VX_OUTPUT, VX_TYPE_MATRIX,  VX_PARAMETER_STATE_REQUIRED ) ); // output

    ERROR_CHECK_STATUS( vxFinalizeKernel( kernel ) );
    ERROR_CHECK_STATUS( vxReleaseKernel( &kernel ) );

    vxAddLogEntry( ( vx_reference ) context, VX_SUCCESS, "OK: registered user kernel app.userkernels.birdseye_transform\n" );
    return VX_SUCCESS;
}

void log_callback(vx_context context, vx_reference ref,
  vx_status status, const char* string)
{
    printf("Log message: status %d, text: %s\n", (int)status, string);
}

vx_graph makeBirdsEyeViewGraph(vx_context context, vx_image input,
  vx_image* binary, vx_array lines, vx_array vanishing_points, vx_matrix perspective,
  vx_image birds_eye)
{

    vx_graph graph = vxCreateGraph(context);

    /* creates a graph with one input image and one output image.
    You supply the input and output images, it is assumed that the input and output images are RGB.
    */

    vx_uint32 width, height;
    vxQueryImage(input, VX_IMAGE_WIDTH, &width, sizeof(vx_uint32));
    vxQueryImage(input, VX_IMAGE_HEIGHT, &height, sizeof(vx_uint32));

    /* create virtual images */
    const int numu8 = 6;
    vx_image virt_u8[numu8];

    for(int i = 0; i < numu8; i++)
    {
      virt_u8[i] = vxCreateVirtualImage(graph, width, height, VX_DF_IMAGE_U8);
    }

    widthr = width/scale_factor;
    heightr = height/scale_factor;

    vx_image virt_nv12 = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_NV12);
    vx_image virt_y = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
    vx_image virt_yr = vxCreateVirtualImage(graph, widthr, heightr,
      VX_DF_IMAGE_U8);
    vx_image binary_thresh = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);

    const int nums16 = 3;
    vx_image virt_s16[nums16];

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

    vx_array _lines = vxCreateVirtualArray(graph, VX_TYPE_LINE_2D, max_num_lines);

    vx_scalar num_lines = vxCreateScalar(context, VX_TYPE_SIZE, NULL);

    /* run hough transform */
    vx_hough_lines_p_t hough_params;
    hough_params.rho = 1.0f;
    hough_params.theta = 3.14f/180;
    hough_params.threshold = 100;
    hough_params.line_length = 100;
    hough_params.line_gap = 10;
    hough_params.theta_max = 3.14;
    hough_params.theta_min = 0.0;

    vxHoughLinesPNode(graph, *binary, &hough_params, _lines, num_lines);

    userFilterLinesNode(graph, _lines, lines);
    userFindVanishingPoint(graph, lines, vanishing_points);

    userComputeBirdsEyeTransform(graph, vanishing_points, input, perspective);

    /* Create the same processing subgraph for each channel */
    enum vx_channel_e channels[] = {VX_CHANNEL_R, VX_CHANNEL_G, VX_CHANNEL_B};

    for(int i = 0; i < 3; i++)
    {
      /* First, extract input and logo R, G, and B channels to individual
      virtual images */
      vxChannelExtractNode(graph, input, channels[i], virt_u8[i]);

      vx_node warp_node = vxWarpPerspectiveNode(graph, virt_u8[i], perspective, VX_INTERPOLATION_BILINEAR, virt_u8[i + 3]);
      ERROR_CHECK_OBJECT(warp_node);

      // set the border mode to constant with zero value
      vx_border_t border_mode;
      border_mode.mode = VX_BORDER_CONSTANT;
      border_mode.constant_value.U8 = 0;
      vxSetNodeAttribute(warp_node, VX_NODE_BORDER, &border_mode, sizeof(border_mode));
    }
    vxChannelCombineNode(graph, virt_u8[3], virt_u8[4], virt_u8[5], NULL, birds_eye);

    return graph;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
      printf("Find straight lines in an image\n"
             "%s <input> <output>\n", (char *)argv[0]);
        exit(0);
    }

    const char* input_filename = argv[1];
    const char* output_filename = argv[2];

    vx_context context = vxCreateContext();

    vx_image image;
    vxa_read_image((const char *)input_filename, context, &image);

    vx_uint32 width, height;
    vxQueryImage(image, VX_IMAGE_WIDTH, &width, sizeof(vx_uint32));
    vxQueryImage(image, VX_IMAGE_HEIGHT, &height, sizeof(vx_uint32));

    vx_image output = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    test = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    test_matrix = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 3);

    float vals[3][3] = {{1, 0, 0},{0, 1, 0}, {0, 0, 1}};
    vxCopyMatrix(test_matrix, vals, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);


    ERROR_CHECK_STATUS(registerUserFilterLinesKernel(context));
    ERROR_CHECK_STATUS(registerUserVanishingPointKernel(context));
    ERROR_CHECK_STATUS(registerUserBirdsEyeTransformKernel(context));

    vx_image binary;
    vx_array lines, vanishing_points;

    /* create an array for storing hough lines output */
    ERROR_CHECK_OBJECT(lines = vxCreateArray(context, VX_TYPE_LINE_2D, max_num_lines));

    /* create an array for storing vanishing point candidates */
    ERROR_CHECK_OBJECT(vanishing_points = vxCreateArray(context, VX_TYPE_COORDINATES2D, max_num_lines));

    vx_matrix perspective = vxCreateMatrix(context, VX_TYPE_FLOAT32, 3, 3);
    vx_graph graph = makeBirdsEyeViewGraph(context, image, &binary, lines, vanishing_points,
      perspective, output);

    vxRegisterLogCallback(context, log_callback, vx_true_e);

    vxProcessGraph(graph);

    vxa_write_image(output, output_filename);

    vxReleaseContext(&context);
    return(0);
}
