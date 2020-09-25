/*
copyImage.c
Read and write an image.
*/
#include <VX/vx.h>
#include <stdio.h>
#include <stdlib.h>
#include "readImage.h"
#include "writeImage.h"

vx_graph makeFilterGraph(vx_context context, vx_image input,
  vx_rectangle_t* rect, vx_image output)
{
    /* creates a graph with one input image and one output image.
    You supply the input and output images, it is assumed that the input and output images are RGB.
    Replace the default processing with what you like!
    */
    #define numv8 (6)             /* Number of virtual U8 images we need */
    vx_graph graph = vxCreateGraph(context);
    vx_image virtu8[numv8];
    int i, j;

    for (i = 0; i < numv8; ++i)
        virtu8[i] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);

    /* Set convolution coefficients to a 3x3 box filter */
    vx_int16 scharr_coeffs[3][3] = {
      {3, 0, -3},
      {10, 0, -10},
      {3, 0, -3}
    };
    /* Set the scale for convolution */
    vx_uint32 scale = 2;

    /* Create convolution object, set convolution coefficients and scale*/
    vx_convolution scharr = vxCreateConvolution(context, 3, 3);
    vxCopyConvolutionCoefficients(scharr, (vx_int16*)scharr_coeffs, VX_WRITE_ONLY,
      VX_MEMORY_TYPE_HOST);
    vxSetConvolutionAttribute(scharr, VX_CONVOLUTION_SCALE, &scale, sizeof(scale));

    /* create ROI */
    vx_image roi = vxCreateImageFromROI(input, rect);

    /* Do scharr filtering on the input image */
    /* First, extract R, G, and B channels to individual virtual images */
    vxChannelExtractNode(graph, roi, VX_CHANNEL_R, virtu8[0]);
    vxChannelExtractNode(graph, roi, VX_CHANNEL_G, virtu8[1]);
    vxChannelExtractNode(graph, roi, VX_CHANNEL_B, virtu8[2]);

    /* Now, run box filter on each of gray scale images */
    for(i = 0; i < 3; i++)
      vxConvolveNode(graph, virtu8[i], scharr, virtu8[i + 3]);

    /* Now combine images together in the ouptut color image */
    vxChannelCombineNode(graph, virtu8[3], virtu8[4], virtu8[5], NULL, output);

    for (i =0; i < numv8; ++i)
        vxReleaseImage(&virtu8[i]);
    return graph;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Filter an image\n"
               "%s <input> <output>\n", (char *)argv[0]);
        return(1);
    }

    struct read_image_attributes attr;
    vx_context context = vxCreateContext();
    vx_image image = createImageFromFile(context, (const char*) argv[1], &attr);

    vx_rectangle_t rect;
    rect.start_x = 204;
    rect.start_y = 179;
    int width = 178;
    int height = 190;
    rect.end_x = rect.start_x + width;
    rect.end_y = rect.start_y + height;

    vx_image output = vxCreateImage(context, width, height, VX_DF_IMAGE_RGB);
    vx_graph graph = makeFilterGraph(context, image, &rect, output);
    if (vxGetStatus((vx_reference)image))
        printf("Could not create input image\n");
    else if (vxProcessGraph(graph))
        printf("Error processing graph\n");
    else if (writeImage(output, (const char*)argv[2]))
        printf("Problem writing the output image\n");
    vxReleaseContext(&context);

    return(0);
}
