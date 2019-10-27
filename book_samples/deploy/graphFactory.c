/*
graphFactory.c
Create a test graph in the context
*/
#include <VX/vx.h>
#include <stdio.h>
#include <stdlib.h>

void releaseNode(vx_node node)
{
    vxReleaseNode(&node);
}

vx_graph makeTestGraph(vx_context context, vx_image image, vx_image output)
{
    /* creates a graph with one input image and one output image.
    The input and output images can be provided through the mechanism of graph paramters,
    it is assumed that the input and output images are RGB.
    Replace the default processing with what you like!
    */
    enum {
        numvyuv = 2,     /* Number of virtual YUV images we need */
        numv16  = 3,     /* Number of virtual S16 images we need */
        numv8   = 8      /* Number of virtual U8 images we need */
    };
    vx_graph graph = vxCreateGraph(context);
    vx_image virtsyuv[numvyuv], virts8[numv8], virts16[numv16];
    
    int i;

    for (i = 0; i < numvyuv; ++i)
        virtsyuv[i] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_NV12);
    for (i = 0; i < numv8; ++i)
        virts8[i] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_U8);
    for (i = 0; i < numv16; ++i)
        virts16[i] = vxCreateVirtualImage(graph, 0, 0, VX_DF_IMAGE_S16);
    
    /* Do some arbitrary processing on the imput image */
    /* First, make a true greyscale image. We do this by converting to YUV
    and extracting the Y. */
    vx_node node = vxColorConvertNode(graph, image, virtsyuv[0]);
    
    /* Get the parameter that will be the input and add it to the graph */
    vx_parameter parameter = vxGetParameterByIndex(node, 0);
    vxReleaseNode(&node);
    vxAddParameterToGraph(graph, parameter);
    vxReleaseParameter(&parameter);

    /* Extract the Y */
    releaseNode(vxChannelExtractNode(graph, virtsyuv[0], VX_CHANNEL_Y, virts8[0]));
    
    /* Use Sobel plus magnitude to find edges on the greyscale image */
    releaseNode(vxSobel3x3Node(graph, virts8[0], virts16[0], virts16[1]));
    /* Note that we have to use specifically U8 and S16 images to satisfy the convert depth node */
    releaseNode(vxMagnitudeNode(graph, virts16[0], virts16[1], virts16[2]));
    vx_int32 shift = 1;
    vx_scalar shift_scalar = vxCreateScalar(context, VX_TYPE_INT32, &shift);
    releaseNode(vxConvertDepthNode(graph, virts16[2], virts8[1], VX_CONVERT_POLICY_SATURATE, shift_scalar));
    vxReleaseScalar(&shift_scalar);
    
    /* Make the edges wider, then black and AND the edges back with the Y value so as to super-impose a black background */
    releaseNode(vxDilate3x3Node(graph, virts8[1], virts8[2]));
    releaseNode(vxDilate3x3Node(graph, virts8[2], virts8[3]));
    releaseNode(vxNotNode(graph, virts8[3], virts8[4]));
    releaseNode(vxAndNode(graph, virts8[0], virts8[4], virts8[5]));

    /* Get the U and V channels as well.. */
    releaseNode(vxChannelExtractNode(graph, virtsyuv[0], VX_CHANNEL_U, virts8[6]));
    releaseNode(vxChannelExtractNode(graph, virtsyuv[0], VX_CHANNEL_V, virts8[7]));
    
    /* Combine the colour channels to give a YUV output image */
    releaseNode(vxChannelCombineNode(graph, virts8[5], virts8[6], virts8[7], NULL, virtsyuv[1]));
    
    /* Convert the YUV to RGB output */
    node = vxColorConvertNode(graph, virtsyuv[1], output);
    
    /* Now get the parameter that will be the output and add it to the graph */
    parameter = vxGetParameterByIndex(node, 1);
    vxReleaseNode(&node);
    vxAddParameterToGraph(graph, parameter);
    vxReleaseParameter(&parameter);
    
    /* Give the graph a name */
    vxSetReferenceName((vx_reference)graph, "Test Graph");

    for (i =0; i < numv16; ++i)
        vxReleaseImage(&virts16[i]);
    for (i =0; i < numvyuv; ++i)
        vxReleaseImage(&virtsyuv[i]);
    for (i =0; i < numv8; ++i)
        vxReleaseImage(&virts8[i]);
    return graph;
}
