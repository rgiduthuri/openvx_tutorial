/*
copyImage.c
Read and write an image.
*/
#include <VX/vx.h>
#include <VX/vxu.h>
#include <stdio.h>
#include <stdlib.h>
#include "readImage.h"
#include "writeImage.h"

void main(int argc, char **argv)
{
    printf("Started\n");
    struct read_image_attributes attr;
    vx_context context = vxCreateContext();

    vx_border_t border_mode;
    border_mode.mode = VX_BORDER_REPLICATE;
    vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &border_mode, sizeof(border_mode));

    vx_image input = createImageFromFile(context, "cup.ppm", &attr);
    printf("input image created\n");
    if(vxGetStatus(input) != VX_SUCCESS)
    {
        printf("Status error: %d", vxGetStatus(input));
    }

    vx_rectangle_t rect;
    rect.start_x = 48;
    rect.start_y = 98;
    rect.end_x = 258;
    rect.end_y = 202;
    vx_image roi = vxCreateImageFromROI(input, &rect);

    int width = rect.end_x - rect.start_x;
    int height = rect.end_y - rect.start_y;

    /* create temporary images for working with edge images */
    vx_image copy_channel[3], roi_channel[3], edges, edges_inv;
    edges = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
    edges_inv = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);

    /* set the threshold value */
    vx_pixel_value_t lower, higher;
    lower.U32 = 50;
    higher.U32 = 100;

    /* create a threshold object */
    vx_threshold threshold = vxCreateThresholdForImage(context,
      VX_THRESHOLD_TYPE_RANGE, VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    if(vxGetStatus(threshold) != VX_SUCCESS)
    {
      printf("Threshold creation failed\n");
    }

    /* set threshold values */
    vxCopyThresholdRange(threshold, &lower, &higher, VX_WRITE_ONLY,
      VX_MEMORY_TYPE_HOST);

    enum vx_channel_e channels[] = {VX_CHANNEL_R, VX_CHANNEL_G, VX_CHANNEL_B};
    for(int i = 0; i < 3; i++)
    {
        roi_channel[i] = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
        copy_channel[i] = vxCreateImage(context, width, height, VX_DF_IMAGE_U8);
        vxuChannelExtract(context, roi, channels[i], roi_channel[i]);

        vxuCannyEdgeDetector(context, roi_channel[i], threshold, 3, VX_NORM_L2, edges);

        vxuNot(context, edges, edges_inv);
        vxuAnd(context, roi_channel[i], edges_inv, copy_channel[i]);
    }

    vxuChannelCombine(context, copy_channel[0], copy_channel[1], copy_channel[2], NULL, roi);
    for(int i = 0; i < 3; i++)
    {
      vxReleaseImage(&copy_channel[i]);
      vxReleaseImage(&roi_channel[i]);
    }
    vxReleaseImage(&edges);
    vxReleaseImage(&edges_inv);
    vxReleaseThreshold(&threshold);

    if(writeImage(input, "cup_roi.ppm"))
        printf("Problem writing the output image\n");

    vxReleaseImage(&roi);
    vxReleaseImage(&input);
    vxReleaseContext(&context);
}
