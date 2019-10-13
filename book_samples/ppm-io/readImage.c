/*
readImage(image, filename, crop, place, fill)
Utility to read an openvx image from a file (.ppm, .pgm)
image is a single plane U8, U16, RGB or RGBA vx_image that has already been created, filename is the
name of a file to be opened that will be in one of two recognised formats:
portable pixmap, binary (P6) with a maximum value of 255
portbale greyscale map, binary (P5) with a maximum value of 255 or 65535

crop, place, and fill are flags controlling how the data is treated, as described below.
if the file exists and is of a recognised format, then data will be read from the file into the image.
If either the width or the height of the image in the file is greater than the width or the height of
the vx_image, then the image is cropped according to the value of the 'crop' flag, which may be:
READ_IMAGE_USE_NONE : Return an error if there is too much data in either direction
READ_IMAGE_USE_TOP_LEFT : use top left of image
READ_IMAGE_USE_TOP_RIGHT : use top right of image
READ_IMAGE_USE_BOTTOM_LEFT : use bottom left of image
READ_IMAGE_USE_BOTTOM_RIGHT : use bottom right of image
READ_IMAGE_USE_CENTRE : use centre of the image

If on the other hand the file contains less data than is required to fill the image then the image is placed
according to the value of the 'place' flag, which may be:
READ_IMAGE_PLACE_NONE : return an error if there is too little data in either direction
READ_IMAGE_PLACE_TOP_LEFT : place at top left
READ_IMAGE_PLACE_TOP_RIGHT : place at top right
READ_IMAGE_PLACE_BOTTOM_LEFT : place at bottom left
READ_IMAGE_PLACE_BOTTOM_RIGHT : place at bottom right
READ_IMAGE PLACE_CENTRE : place in the centre

Pixels in the vx_image that have not been written will be filled according to the value of the 'fill' flag:
READ_IMAGE_FILL_NONE : Leave the unwritten locations unchanged (uesful to place an image on a background)
READ_IMAGE_FILL_ZERO : Fill unwritten locations with zero
READ_IMAGE_FILL_ONES : Fill unwritten locations with maximum value

Return values:
VX_SUCCESS : all is OK!
VX_FAILURE: Could not open file
VX_ERROR_NOT_SUPPORTED: File is not of a recognised format
VX_ERROR_INVALID_DIMENSION: Image dimensions not compatible
VX_ERROR_INVALID_FORMAT: Image formats not compatible
VX_ERROR_INVALID_PARAMETERS: Supplied parameters don't make sense in context

Other return values of different meanings are possible when returned from the vx apis if image is not valid.
*/
#include <VX/vx.h>
#include <stdio.h>
#include <stdlib.h>
#include "readImage.h"
/*
fillPixel - set a given pixel to a value based upon a given integer value, the image format, and conversion flag.
*/

static void fillPixel(void *imgp, const vx_imagepatch_addressing_t * addr, vx_uint32 x,
                      vx_uint32 y, vx_pixel_value_t fillValue, vx_df_image image_format)
{
    vx_pixel_value_t *pixptr = (vx_pixel_value_t *) vxFormatImagePatchAddress2d(imgp, x, y, addr);
    switch (image_format)
    {
        case VX_DF_IMAGE_U8:
            pixptr -> U8 = fillValue.U8;
            break;
        case VX_DF_IMAGE_U16:
            pixptr -> U16 = fillValue.U16;
            break;
        case VX_DF_IMAGE_RGBX:
            pixptr -> RGBX[3] = 0;       /* Fall through to RGB */
        case VX_DF_IMAGE_RGB:
            pixptr -> RGBX[0] = fillValue.RGBX[0];
            pixptr -> RGBX[1] = fillValue.RGBX[1];
            pixptr -> RGBX[2] = fillValue.RGBX[2];
            break;
        default: break;
    }
}

static void copyPixel(void *filerow, void *imgp, const vx_imagepatch_addressing_t *addr, vx_uint32 src_x,
                           vx_uint32 dst_x, vx_uint32 dst_y, int psz, vx_df_image image_format)
{
    fillPixel(imgp, addr, dst_x, dst_y, *(vx_pixel_value_t *)(filerow + src_x * psz), image_format);
}

static vx_status readHeader(FILE *fp, int *psz, vx_uint32 *width, vx_uint32 *height)
{
    vx_status status = VX_SUCCESS;
    char *buffer = 0;
    char line[1024];
    int fmt = 0;    /* Format of ppm file (allowed 5 or 6) */
    int maxval = 0; /* maximum value of each datum in the file */
    *width = 0;     /* width of image in the file */
    *height = 0;    /* height of image in the file */
    *psz = 0;       /* size of pixel in file */

    /* Notice that because we are using fgets, the separator before the data *must* be a newline */
    fgets(line, 1024, fp);
    int n = sscanf(line, "P%d %d %d %d", &fmt, width, height, &maxval);

    if (n > 0)
    {
        if (5 == fmt)
        {
            *psz = 1;
        }
        else if (6 == fmt)
        {
            *psz = 3;
        }
        while (n < 4 && fgets(line, 1024, fp))
        {
            if ('#' != line[0])
            {
                if (1 == n)
                    n += sscanf(line, "%d %d %d", width, height, &maxval);
                else if (2 == n)
                    n += sscanf(line, "%d %d", height, &maxval);
                else if (3 == n)
                    n += sscanf(line, "%d", &maxval);
            }
        }
        if (n < 4 || 0 == fmt || (maxval > 255 && *psz == 3))
        {
            /* Report an error because the file is not of a recognised format */
            status = VX_ERROR_NOT_SUPPORTED;
        }
        else if (maxval > 255)
        {
            /* Adjust size of data for maxval */
            *psz <<= 1;
        }
    }
    else
        status = VX_ERROR_NOT_SUPPORTED;
    return status;
}

vx_status readImage(vx_image image, const char * filename, enum read_image_crop crop,
                    enum read_image_place place, enum read_image_fill fill)
{
    /* Check the image and get properties*/
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
    status = status || vxMapImagePatch(image, &rect, 0, &map_id, &addr, &imgp, VX_READ_AND_WRITE, VX_MEMORY_TYPE_HOST, VX_NOGAP_X);
    if (VX_SUCCESS == status)
    {
        FILE *fp = fopen(filename, "rb");
        if (fp)
        {
            int psz = 0;       /* Size of a single pixel in the file */
            vx_uint32 width = 0;     /* width of image in the file */
            vx_uint32 height = 0;    /* height of image in the file */

            status = readHeader(fp, &psz, &width, &height);

            if (VX_SUCCESS == status)
            {
                /* Now check formats and sizes */
                if ((READ_IMAGE_USE_NONE == crop && (width > image_width || height > image_height)) ||
                    (READ_IMAGE_PLACE_NONE == place && (width < image_width || height < image_height)))
                {
                    /* Report an error because the dimension of the file image does not match the dimensions of the
                       vx_imge and caller has specified no cropping or positioning */
                    status = VX_ERROR_INVALID_DIMENSION;
                }
                else if (image_format != VX_DF_IMAGE_U8 && image_format != VX_DF_IMAGE_U16 &&
                         image_format != VX_DF_IMAGE_RGB && image_format != VX_DF_IMAGE_RGBX)
                {
                    /* Report an error becuase the vx_image is in a format that is not supported */
                    status = VX_ERROR_NOT_SUPPORTED;
                }
                else if ((image_format == VX_DF_IMAGE_U8 && psz != 1) ||
                         (image_format == VX_DF_IMAGE_U16 && psz != 2) ||
                         (image_format == VX_DF_IMAGE_RGB && psz != 3) ||
                         (image_format == VX_DF_IMAGE_RGBX && psz != 3))
                {
                    /* Report an error because the image file format does not match the vx_image */
                    status = VX_ERROR_INVALID_FORMAT;
                }
                else /* Now read the data and insert into the vx_image */
                {
                    int x, y;
                    int src_x_offset = 0, src_y_offset = 0, dst_x_offset = 0, dst_y_offset = 0;
                    vx_uint32 copy_width = image_width, copy_height = image_height;
                    vx_pixel_value_t fillValue = { .U32 = READ_IMAGE_FILL_ONES == fill ? 0xFFFFFF : 0 };
                    void *filerow = calloc(width, psz);
                    /* first calculate offsets */
                    if (width > image_width)
                    {
                        if (READ_IMAGE_USE_TOP_RIGHT == crop || READ_IMAGE_USE_BOTTOM_RIGHT == crop)
                          src_x_offset = width - image_width;
                        else if (READ_IMAGE_USE_CENTRE == crop)
                          src_x_offset = (width - image_width) / 2;
                    }
                    else if (image_width > width)
                    {
                        if (READ_IMAGE_PLACE_TOP_RIGHT == place || READ_IMAGE_PLACE_BOTTOM_RIGHT == place)
                          dst_x_offset = image_width - width;
                        else if (READ_IMAGE_PLACE_CENTRE == place)
                          dst_x_offset = (image_width - width) / 2;
                        copy_width = dst_x_offset + width;
                    }
                    if (height > image_height)
                    {
                        if (READ_IMAGE_USE_BOTTOM_LEFT == crop || READ_IMAGE_USE_BOTTOM_RIGHT == crop)
                          src_y_offset = height - image_height;
                        else if (READ_IMAGE_USE_CENTRE == crop)
                          src_y_offset = (height - image_height) / 2;
                    }
                    else if (image_height > height)
                    {
                        if (READ_IMAGE_PLACE_BOTTOM_LEFT == place || READ_IMAGE_PLACE_BOTTOM_RIGHT == place)
                          dst_y_offset = image_height - height;
                        else if (READ_IMAGE_PLACE_CENTRE == place)
                          dst_y_offset = (image_height - height) / 2;
                        copy_height = dst_y_offset + height;
                    }

                    /* Skip initial rows in the file */
                    for (y = 0; y < src_y_offset; ++y)
                        fread(filerow, psz, width, fp);

                    /* Now insert the pixels, performing any necessary conversion */
                    for (y = 0; y < dst_y_offset; ++y)
                    {
                        /* Fill initial rows */
                        if (READ_IMAGE_FILL_NONE != fill)
                            for (x = 0; x < image_width; ++x)
                                fillPixel(imgp, &addr, x, y, fillValue, image_format);
                    }
                    for (; y < copy_height; ++y)
                    {
                        /* Copy rows */
                        for (x = 0; x < dst_x_offset; ++x)
                            if (READ_IMAGE_FILL_NONE != fill)
                            {
                                /* Fill initial pixels on row */
                                fillPixel(imgp, &addr, x, y, fillValue, image_format);
                            }
                        fread(filerow, psz, width, fp);
                        for (; x < copy_width; ++x)
                        {
                            /* Copy pixels */
                            copyPixel(filerow, imgp, &addr, x + src_x_offset, x + dst_x_offset,
                                               y + dst_y_offset, psz, image_format);
                        }
                        if (READ_IMAGE_FILL_NONE != fill)
                            for (; x < image_width; ++x)
                            {
                                /* Fill final pixels on row */
                                fillPixel(imgp, &addr, x, y, fillValue, image_format);
                            }
                    }
                    if (READ_IMAGE_FILL_NONE != fill)
                        for (; y < image_height; ++y)
                        {
                            /* Fill final rows */
                            for (x = 0; x < image_width; ++x)
                                fillPixel(imgp, &addr, x, y, fillValue, image_format);
                        } /* for */
                    free(filerow);
                }   /* else */
            } /* if (VX_SUCCESS == status) */
            fclose(fp);
        } /* if (fp) */
        else
        {
            /* Report an error because file could not be opened */
            status = VX_FAILURE;
        }
        vxUnmapImagePatch(image, map_id);
    } /*   if (VX_SUCCESS == vx_status) */
    return status;
}

vx_image createImageFromFile(vx_context context, const char * filename, struct read_image_attributes *attr)
{
    FILE *fp = fopen(filename, "rb");
    vx_image image = NULL;
    if (fp)
    {
        int psz = 0;             /* Size of a single pixel in the file */
        vx_uint32 width = 0;     /* width of image in the file */
        vx_uint32 height = 0;    /* height of image in the file */
        vx_df_image format;

        if (VX_SUCCESS == readHeader(fp, &psz, &width, &height))
        {
            switch (psz)
            {
                case 1:
                    format = VX_DF_IMAGE_U8;
                    break;
                case 2:
                    format = VX_DF_IMAGE_U16;
                    break;
                case 3:
                    format = VX_DF_IMAGE_RGB;
                    break;
                default:
                    format = VX_DF_IMAGE_VIRT;
                    printf("Invalid value for psz: %d\n", psz);
                    break;
            }
            image = vxCreateImage(context, width, height, format);
            vx_rectangle_t rect = {.start_x = 0, .start_y = 0, .end_x = width, .end_y = height};
            vx_imagepatch_addressing_t addr = VX_IMAGEPATCH_ADDR_INIT;
            void * imgp;
            vx_map_id map_id;
            if (VX_SUCCESS == (vxGetStatus((vx_reference)image) || 
                               vxMapImagePatch(image, &rect, 0, &map_id, &addr, &imgp,
                                               VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST, VX_NOGAP_X)))
            {
                int x, y;
                void *filerow = calloc(width, psz);
                if (!filerow)
                    printf("? Could not allocate memory for line\n");
                for (y = 0; y < height; ++y)
                {
                    /* Copy rows */
                    if (fread(filerow, psz, width, fp) != width)
                        printf("Error reading file\n");
                    for (x = 0; x < width; ++x)
                    {
                        /* Copy pixels */
                        copyPixel(filerow, imgp, &addr, x, x, y, psz, format);
                    }
                }
                free(filerow);
                vxUnmapImagePatch(image, map_id);
                if (NULL != attr)
                {
                    attr->width = width;
                    attr->height = height;
                    attr->format = format;
                }
            }
        }
        fclose(fp);
    }
    return image;
}
