/*
changeImage.c
Read an image, change it, write it out.
*/
#include <VX/vx.h>
#include <stdio.h>
#include <stdlib.h>
#include "readImage.h"
#include "writeImage.h"

vx_graph makeTestGraph(vx_context context, vx_image image, vx_image output)
{
    /* creates a graph with one input image and one output image.
    You supply the input and output images, it is assumed that the input and output images are RGB.
    Replace the default processing with what you like!
    */
    #define numv8 (6)             /* Number of virtual U8 images we need */
    #define numvyuv (2)           /* Number of virtual YUV4 images we need */
    vx_graph graph = vxCreateGraph(context);
    vx_image virts8[numv8], virtsyuv[numvyuv];
    vx_threshold hyst = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vx_pixel_value_t lower_value = {.U8 = 220};
    vx_pixel_value_t upper_value = {.U8 = 230};

    vxCopyThresholdRange(hyst, &lower_value, &upper_value, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    int i;

    for (i = 0; i < numv8; ++i)
        virts8[i] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
    for (i = 0; i < numvyuv; ++i)
        virtsyuv[i] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_IYUV);

    /* Do some arbitrary processing on the imput image */
    /* First, make a true greyscale image. We do this by converting to YUV
    and extracting the Y. */
    vxColorConvertNode(graph, image, virtsyuv[0]);
    vxChannelExtractNode(graph, virtsyuv[0], VX_CHANNEL_Y, virts8[0]);

    /* Use a Canny detector on the greyscale image to find edges */
    vxCannyEdgeDetectorNode(graph, virts8[0], hyst, 5, VX_NORM_L1, virts8[1]);

    /* Make the edges black and AND the edges back with the Y value so as to super-impose a black background */
    vxNotNode(graph, virts8[1], virts8[2]);
    vxAndNode(graph, virts8[0], virts8[2], virts8[3]);

    /* Get the U and V channels as well.. */
    vxChannelExtractNode(graph, virtsyuv[0], VX_CHANNEL_U, virts8[4]);
    vxChannelExtractNode(graph, virtsyuv[0], VX_CHANNEL_V, virts8[5]);

    /* Combine the colour channels to give a YUV output image */
    vxChannelCombineNode(graph, virts8[3], virts8[4], virts8[5], NULL, virtsyuv[1]);

    /* Convert the YUV to RGB output */
    vxColorConvertNode(graph, virtsyuv[1], output);

    for (i =0; i < numv8; ++i)
        vxReleaseImage(&virts8[i]);
    for (i =0; i < numvyuv; ++i)
        vxReleaseImage(&virtsyuv[i]);
    return graph;
}

int main(int argc, char **argv)
{
    if (argc != 3)
    {
        printf("Change an image\n"
               "%s <input> <output>\n", (char *)argv[0]);
    }
    else
    {
        struct read_image_attributes attr;
        vx_context context = vxCreateContext();
        vx_image image = createImageFromFile(context, (const char *)argv[1], &attr);
        vx_image output = vxCreateImage(context, attr.width, attr.height, attr.format);
        vx_graph graph = makeTestGraph(context, image, output);
        if (vxGetStatus((vx_reference)image))
            printf("Could not create input image\n");
        else if (vxProcessGraph(graph))
            printf("Error processing graph\n");
        else if (writeImage(output, (const char *)argv[2]))
            printf("Problem writing the output image\n");
        vxReleaseContext(&context);
    }
}
