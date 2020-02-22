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

    vx_border_t border_mode;
    border_mode.mode = VX_BORDER_REPLICATE;
    vxSetContextAttribute(context, VX_CONTEXT_IMMEDIATE_BORDER, &border_mode, sizeof(border_mode));

    vx_image input = createImageFromFile(context, (const char*)argv[1], &attr);
    printf("input image created\n");
    if(vxGetStatus((vx_reference)input) != VX_SUCCESS)
    {
        printf("Status error: %d", vxGetStatus((vx_reference)input));
    }

    vx_rectangle_t rect;
    rect.start_x = 204;
    rect.start_y = 179;
    int rect_width = 178;
    int rect_height = 190;
    rect.end_x = rect.start_x + rect_width;
    rect.end_y = rect.start_y + rect_height;
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
    if(vxGetStatus((vx_reference)threshold) != VX_SUCCESS)
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

    if(writeImage(input, (const char*)argv[2]))
        printf("Problem writing the output image\n");

    vxReleaseImage(&roi);
    vxReleaseImage(&input);
    vxReleaseContext(&context);

    return(0);
}
