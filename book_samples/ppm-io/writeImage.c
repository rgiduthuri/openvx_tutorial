/*
writeImage.c
Write an image out to a .ppm or .pgm file.
Supported image formats:
    VX_DF_IMAGE_U8: creates a portable greyscale map (P5) with a maximum value of 255
    VX_DF_IMAGE_U16:creates a portable greyscale map (P5) with a maximum value of 65535
    VX_DF_IMAGE_RGB: creates a portable pixel map (P6) with a maximum value of 255
    VX_DF_IMAGE_RGBX: creates a portable pixel map (P6) with a maximum value of 255, the fourth channel is not output.
Return values:
VX_SUCCESS : all is OK!
VX_FAILURE: Could not open file or write error
VX_ERROR_NOT_SUPPORTED: Image is not of a supported format
Another error may be reported if there is a problem with the image itself.
*/
#include <VX/vx.h>
#include <stdio.h>
#include <stdlib.h>
#include "writeImage.h"

vx_status writeImage(vx_image image, const char *filename)
{
    vx_uint32 image_width;
    vx_uint32 image_height;
    vx_df_image image_format;
    vx_status status = vxGetStatus((vx_reference)image) || 
                       vxQueryImage(image, VX_IMAGE_WIDTH, &image_width, sizeof(image_width)) ||
                       vxQueryImage(image, VX_IMAGE_HEIGHT, &image_height, sizeof(image_height)) ||
                       vxQueryImage(image, VX_IMAGE_FORMAT, &image_format, sizeof(image_format));
    vx_rectangle_t rect = {.start_x = 0, .start_y = 0, .end_x = image_width, .end_y = image_height};
    vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
    void * imgp;
    vx_map_id map_id;
    int maxval, psz;
    char fmt;
    switch (image_format)
    {
        case VX_DF_IMAGE_U8:
            fmt = '5';
            psz = 1;
            maxval = 255;
            break;
        case VX_DF_IMAGE_U16:
            fmt = '5';
            psz = 2;
            maxval = 65535;
            break;
        case VX_DF_IMAGE_RGB:
        case VX_DF_IMAGE_RGBX:
            fmt = '6';
            psz = 3;
            maxval = 255;
            break;
        default:
            status = VX_ERROR_NOT_SUPPORTED;
    }
    status = status || vxMapImagePatch(image, &rect, 0, &map_id, &addr, &imgp, VX_READ_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X);
    if (VX_SUCCESS == status)
    {
        FILE *fp = fopen(filename, "wb");
        if (fp)
        {
            fprintf(fp, "P%c\n%d %d\n%d\n", fmt, image_width, image_height, maxval);
            int x, y;
            for (y = 0; y < image_height; ++y)
                for (x = 0; x < image_width; ++x)
                {
                    if (fwrite(vxFormatImagePatchAddress2d(imgp, x, y, &addr), 1, psz, fp) != psz)
                        status = VX_FAILURE;
                }
            if (fclose(fp))
                status = VX_FAILURE;
        }
        else
            status = VX_FAILURE;
        vxUnmapImagePatch(image, map_id);
    }
    return status;
}