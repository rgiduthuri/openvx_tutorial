/*
writeImage.h
Write an image out to a .ppm or .pgm file.
Supported image formats:
    VX_DF_IMAGE_U8: creates a portable greyscale map (P5) with a maximum value of 255
    VX_DF_IMAGE_U16:creates a portable greyscale map (P5) with a maximum value of 65535
    VX_DF_IMAGE_RGB: creates a portable pixel map (P6) with a maximum value of 255
    VX_DF_IMAGE_RGBX: creates a portable pixel map (P6) with a maximum value of 255, the fourth channel is not output.
*/
#ifndef _writeImage_h_included_
#define _writeImage_h_included_
#ifdef  __cplusplus
extern "C" {
#endif
    vx_status writeImage(vx_image image, const char *filename);
#ifdef  __cplusplus
}
#endif
#endif
