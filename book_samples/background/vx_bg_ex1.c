#include <VX/vx.h>
#include <VX/vxu.h>

#include <VX/vx_lib_debug.h>
#include <VX/vx_lib_extras.h>
#include <VX/vx_helper.h>
#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <stdarg.h>
#include <assert.h>

#define PATH_MAX 4096

char *viddir = "/mnt/c/Users/Frank/Documents/piper-video";
char *basefname = "piper01";
char filename[PATH_MAX];

int myCaptureImage(vx_context context, vx_image image, int framenum) {
  sprintf(filename, "%s/%s/pgm/%s %04d.pgm", viddir, basefname, basefname, framenum);
  if (framenum == 1)
    printf("Beginning processing %s/%s\n", viddir, basefname);
  return vxuFReadImage(context, filename, image);
}

int myDisplayImage(vx_context context, vx_image image, char *suffix, int framenum) {
  sprintf(filename, "%s/%s/out/%s_%s %04d.pgm", viddir, basefname, basefname, suffix, framenum);
  vxuFWriteImage(context, image, filename);
}

int main(int argc, char *argv[])
{
  vx_uint32 w_in = 1080, h_in = 1920;  // index and input image size
  int scale = 4;  // image size scale
  int w = w_in/scale;  // scaled image width
  int h = h_in/scale;  // scaled image height

  vx_uint8 threshval = 10;    // basic difference threshold value
  if (argc > 1) threshval = atoi(argv[1]);
  printf("Threshold value is %d\n", threshval);
  
  vx_context context = vxCreateContext();

  vxLoadKernels(context, "openvx-debug");  // For reading and writing images

  vx_graph graph = vxCreateGraph(context);

  vx_image input_image = vxCreateImage(context, w_in, h_in, VX_DF_IMAGE_U8);
  vx_image curr_image = vxCreateImage(context, w, h, VX_DF_IMAGE_U8);
  vx_image diff_image = vxCreateVirtualImage(graph, w, h, VX_DF_IMAGE_U8);
  vx_image bg_image = vxCreateImage(context, w, h, VX_DF_IMAGE_U8);
  vx_image fg_image = vxCreateImage(context, w, h, VX_DF_IMAGE_U8);
      
  vx_threshold threshold = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY,
						     VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
  vxCopyThresholdValue(threshold, (vx_pixel_value_t*)&threshval, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

  vx_node scale_node = vxScaleImageNode(graph, input_image, curr_image, VX_INTERPOLATION_AREA);
  vx_node absdiff_node = vxAbsDiffNode(graph, bg_image, curr_image, diff_image);
  vx_node thresh_node = vxThresholdNode(graph, diff_image, threshold, fg_image);

  vxVerifyGraph(graph);

  int framenum = 1;
  while (myCaptureImage(context, input_image, framenum) == VX_SUCCESS) {

    // Initialize the background model
    if (framenum == 1) {
      vxuScaleImage(context, input_image, bg_image, VX_INTERPOLATION_AREA);
    }
    printf("Frame %d%c[1000D", framenum, 0x1b);
    fflush(stdout);
	      
    vxProcessGraph(graph);

    myDisplayImage(context, fg_image, "fg", framenum);
    framenum++;
  }
  printf("Finished after %d frames\n", framenum-1);

  vxReleaseNode(&scale_node);
  vxReleaseNode(&absdiff_node);
  vxReleaseNode(&thresh_node);
  vxReleaseGraph(&graph);
  vxUnloadKernels(context, "openvx-debug");
  vxReleaseContext(&context);

}
