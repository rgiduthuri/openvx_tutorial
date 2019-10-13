/*
readImage.h
Read an image file into a vx_image.
*/
#ifndef _readImage_h_included_
#define _readImage_h_included_
#ifdef  __cplusplus
extern "C" {
#endif
enum read_image_crop {
  READ_IMAGE_USE_NONE,           /* Error if file image too large in either dimension */
  READ_IMAGE_USE_TOP_LEFT,       /* Use left of rows and top of columns */
  READ_IMAGE_USE_TOP_RIGHT,      /* use right of rows and top of columns */
  READ_IMAGE_USE_BOTTOM_LEFT,    /* use left of rows and bottom of columns */
  READ_IMAGE_USE_BOTTOM_RIGHT,   /* use right of rows and bottom of columns */
  READ_IMAGE_USE_CENTRE          /* use centre of image. Indices are truncated when
                                    there are an odd number of pixels left */
};

enum read_image_place {
  READ_IMAGE_PLACE_NONE,         /* Error is file image too small in either direction */
  READ_IMAGE_PLACE_TOP_LEFT,     /* Spare pixels in the vx_image are at the right and bottom */
  READ_IMAGE_PLACE_TOP_RIGHT,    /* Spare pixels in the vx_image are at the left and bottom */
  READ_IMAGE_PLACE_BOTTOM_LEFT,  /* Spare pixels in the vx_image are at the top and right */
  READ_IMAGE_PLACE_BOTTOM_RIGHT, /* Spare pixels in the vx_image are at the top and left */
  READ_IMAGE_PLACE_CENTRE        /* Spare pixels inthe vx_image are distributed evenly. Indices
                                    are truncated when there is an odd number of spare pixels */
};

enum read_image_fill {
  READ_IMAGE_FILL_NONE,          /* Leave spare locations in the target image unchanged */
  READ_IMAGE_FILL_ZERO,          /* Fill with zeros (except A in RGBA will always be max) */
  READ_IMAGE_FILL_ONES           /* Fill with maximum value */
};

struct read_image_attributes {  /* Struct to report back the attributes of the image created from file */
    vx_uint32 width, height;
    vx_df_image format;
};

vx_status readImage(vx_image image, const char *filename, enum read_image_crop crop,
                    enum read_image_place place, enum read_image_fill fill);

vx_image createImageFromFile(vx_context context, const char *filename, struct read_image_attributes *attr);
#ifdef  __cplusplus
}
#endif
#endif