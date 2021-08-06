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

#define CHECK_ALL_ITEMS(array, iter, status, label) { \
    status = VX_SUCCESS; \
    for ((iter) = 0; (iter) < dimof(array); (iter)++) { \
        if ((array)[(iter)] == 0) { \
            printf("Item %u in "#array" is null!\n", (iter)); \
            assert((array)[(iter)] != 0); \
            status = VX_ERROR_NOT_SUFFICIENT; \
        } \
    } \
    if (status != VX_SUCCESS) { \
        goto label; \
    } \
}

/* A local definition to point to a specific unit test */
typedef vx_status (*vx_unittest_f)(int argc, char *argv[]);

/* The structure which correlates each unit test with a result and a name. */
typedef struct _vx_unittest_t {
    vx_status status;
    vx_char name[VX_MAX_KERNEL_NAME];
    vx_unittest_f unittest;
} vx_unittest;

static void vx_print_log(vx_reference ref)
{
    char message[VX_MAX_LOG_MESSAGE_LEN];
    vx_uint32 errnum = 1;
    vx_status status = VX_SUCCESS;
    do {
        status = vxGetLogEntry(ref, message);
        if (status != VX_SUCCESS)
            printf("[%05u] error=%d %s", errnum++, status, message);
    } while (status != VX_SUCCESS);
}

#define PATH_MAX 4096
#define INTERP VX_INTERPOLATION_AREA

vx_status vx_test_graph_accum(int argc, char *argv[]) {
  vx_status status = VX_FAILURE;
  vx_context context = vxCreateContext();
  (void)argc;
  (void)argv;

  char *viddir = "/mnt/c/Users/Frank/Documents/piper-video";
  char *basename = "piper06";
  char filename[PATH_MAX];
  
  if (vxGetStatus((vx_reference)context) == VX_SUCCESS) {

    vx_uint32 i = 0, w_in = 1080, h_in = 1920;  // index and input image size
    int scale = 4;  // image size scale
    
    vx_uint8 constval = 4; // scale weighted absdiff by this so we can see it
    vx_float32 multscale = 1.0f;  // scalar needed by multiply node
    vx_float32 alpha = 0.03f;   // blend factor on weighted images
    vx_uint32 shift = 11;       // shift in squared images (NOT USED)
    vx_uint8 threshval = 15;    // exceeds variance threshold value
    vx_uint8 threshval2 = 25;    // basic difference threshold value

    int w = w_in/scale;  // scaled image width
    int h = h_in/scale;  // scaled image height
    vx_image images[] = {
      vxCreateImage(context, w_in, h_in, VX_DF_IMAGE_U8),  // 0. input
      vxCreateImage(context, w, h, VX_DF_IMAGE_S16),       // 1. accum
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 2. accum weighted
      vxCreateImage(context, w, h, VX_DF_IMAGE_S16),       // 3. accum squared
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 4. scaled input
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 5. absdiff
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 6. absdiff mul weighted

      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 7. thresh absdiff
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 8. absdiff mul
      vxCreateUniformImage(context, w, h, VX_DF_IMAGE_U8, (vx_pixel_value_t*)&constval), // 9. const
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 10. diff minus var

      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 11. median
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 12. dilate
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 13. erode
            
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 14. thresh basic
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 15. median2
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 16. absdiff mul weighted smeared

      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 17. median2
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 18. dilate2
      vxCreateImage(context, w, h, VX_DF_IMAGE_U8),        // 19. erode2
    };
    CHECK_ALL_ITEMS(images, i, status, exit);
    
    vx_scalar scalars[] = {
      vxCreateScalar(context, VX_TYPE_FLOAT32, &alpha),
      vxCreateScalar(context, VX_TYPE_UINT32, &shift),
      vxCreateScalar(context, VX_TYPE_FLOAT32, &multscale),
    };
    CHECK_ALL_ITEMS(scalars, i, status, exit);

    // Create the absdiff threshold object and set its value
    vx_threshold thresh = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY,
						    VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vxCopyThresholdValue(thresh, (vx_pixel_value_t*)&threshval, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    // Create the basic absdiff threshold object and set its value
    vx_threshold thresh2 = vxCreateThresholdForImage(context, VX_THRESHOLD_TYPE_BINARY,
						     VX_DF_IMAGE_U8, VX_DF_IMAGE_U8);
    vxCopyThresholdValue(thresh2, (vx_pixel_value_t*)&threshval2, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST);

    status |= vxLoadKernels(context, "openvx-debug");
    if (status == VX_SUCCESS) {
      // Create the graph
      vx_graph graph = vxCreateGraph(context);
      if (vxGetStatus((vx_reference)graph) == VX_SUCCESS) {
	vx_node nodes[] = {
	  vxScaleImageNode(graph, images[0], images[4], INTERP),
	  //	  vxAccumulateImageNode(graph, images[4], images[1]),
	  vxAccumulateWeightedImageNode(graph, images[4], scalars[0], images[2]),
	  //	  vxAccumulateSquareImageNode(graph, images[4], scalars[1], images[3]),
	  vxAbsDiffNode(graph, images[2], images[4], images[5]),
	  vxThresholdNode(graph, images[5], thresh2, images[14]),
	  //	  vxMedian3x3Node(graph, images[14], images[15]),

	  //	  vxMultiplyNode(graph, images[5], images[9], scalars[2], VX_CONVERT_POLICY_SATURATE,
	  //			 VX_ROUND_POLICY_TO_NEAREST_EVEN, images[8]),
	  //	  vxAccumulateWeightedImageNode(graph, images[8], scalars[0], images[6]),
	  //	  vxBox3x3Node(graph, images[6], images[16]),
	  //	  vxSubtractNode(graph, images[5], images[6], VX_CONVERT_POLICY_SATURATE, images[10]),
	  //	  vxThresholdNode(graph, images[10], thresh, images[7]),
	  
	  /* vxMedian3x3Node(graph, images[7], images[11]), */
	  /* vxDilate3x3Node(graph, images[11], images[12]), */
	  /* vxErode3x3Node(graph, images[12], images[13]), */

	  vxMedian3x3Node(graph, images[14], images[17]),
	  vxDilate3x3Node(graph, images[17], images[18]),
	  vxErode3x3Node(graph, images[18], images[19]),
	};
	CHECK_ALL_ITEMS(nodes, i, status, exit);
	if (status == VX_SUCCESS) {
	  // Verify the graph
	  status = vxVerifyGraph(graph);
	  if (status == VX_SUCCESS) {
	    // Start the processing loop
	    int framenum = 1;
	    while (status == VX_SUCCESS) {

	      // Read the input image
	      sprintf(filename, "%s/%s/pgm/%s %04d.pgm", viddir, basename, basename, framenum);
	      if (vxuFReadImage(context, filename, images[0]) != VX_SUCCESS) {
		printf("Finished after %d frames\n", framenum-1);
		break;
	      }
	      // Initialize the weighted accumulator (background model)
	      if (framenum == 1) {
		vxuScaleImage(context, images[0], images[2], INTERP);
		printf("Beginning processing %s/%s\n", viddir, basename);
	      }
	      printf("Frame %d%c[1000D", framenum, 0x1b);
	      fflush(stdout);
	      
	      /* if (framenum == 150) { */
	      /* 	alpha = 0.00; */
	      /* 	vxCopyScalar(scalars[0], &alpha, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST); */
	      /* } */
	      // GO!
	      status = vxProcessGraph(graph);

	      // Write the output results
	      //	      sprintf(filename, "%s/%s/out/o%saccu_16b %04d.pgm", viddir, basename, basename, framenum);
	      //	      vxuFWriteImage(context, images[1], filename);  // accum
	      //	      sprintf(filename, "%s/%s/out/o%saccw_8b %04d.pgm", viddir, basename, basename, framenum);
	      //	      vxuFWriteImage(context, images[2], filename);  // weighted
	      //	      sprintf(filename, "%s/%s/out/o%saccq_16b %04d.pgm", viddir, basename, basename, framenum);
	      //	      vxuFWriteImage(context, images[3], filename);  // squared
	      //	      sprintf(filename, "%s/%s/out/o%sdiff_8b %04d.pgm", viddir, basename, basename, framenum);
	      //	      vxuFWriteImage(context, images[5], filename);  // absdiff
	      //	      sprintf(filename, "%s/%s/out/o%sdif2_8b %04d.pgm", viddir, basename, basename, framenum);
	      //	      vxuFWriteImage(context, images[8], filename);  // absdiff mul
	      //	      sprintf(filename, "%s/%s/out/o%sdifw_8b %04d.pgm", viddir, basename, basename, framenum);
	      //	      vxuFWriteImage(context, images[6], filename);  // absdiff mul weighted
	      //	      sprintf(filename, "%s/%s/out/o%sthrs_8b %04d.pgm", viddir, basename, basename, framenum);
	      //	      vxuFWriteImage(context, images[13], filename);  // median
	      sprintf(filename, "%s/%s/out/o%sthr2_8b %04d.pgm", viddir, basename, basename, framenum);
	      vxuFWriteImage(context, images[14], filename);  // diff image
	      sprintf(filename, "%s/%s/out/o%smorp_8b %04d.pgm", viddir, basename, basename, framenum);
	      vxuFWriteImage(context, images[18], filename);  // after morphology
	      framenum++;
	    }
	  }
	  else {
	    printf("Can't verify graph!!!\n");
	    vx_print_log((vx_reference)context);
	  }
	  for (i = 0; i < dimof(nodes); i++) {
	    vxReleaseNode(&nodes[i]);
	  }
	}
	else {
	  printf("Can't make nodes!!!\n");
	  vx_print_log((vx_reference)context);
	}
	vxReleaseGraph(&graph);
      }
      status |= vxUnloadKernels(context, "openvx-debug");
    }
    for (i = 0; i < dimof(images); i++) vxReleaseImage(&images[i]);
    for (i = 0; i < dimof(scalars); i++) vxReleaseScalar(&scalars[i]);

  exit:
    vxReleaseContext(&context);
  }
  return status;
}

/*! The array of supported unit tests */
vx_unittest unittests[] = {
    {VX_FAILURE, "Graph: Accumulates",          &vx_test_graph_accum},
};

/*! \brief The main unit test.
 * \param argc The number of arguments.
 * \param argv The array of arguments.
 * \return vx_status
 * \retval 0 Success.
 * \retval !0 Failure of some sort.
 */
int main(int argc, char *argv[])
{
  vx_uint32 i;
  vx_uint32 passed = 0;
  vx_bool stopOnErrors = vx_false_e;

  if (argc == 2 && ((strncmp(argv[1], "-?", 2) == 0) ||
		    (strncmp(argv[1], "--list", 6) == 0) ||
		    (strncmp(argv[1], "-l", 2) == 0) ||
		    (strncmp(argv[1], "/?", 2) == 0))) {
    /* we just want to know which graph is which */
    vx_uint32 t = 0;
    for (t = 0; t < dimof(unittests); t++) 
      printf("%u: %s\n", t, unittests[t].name);
    return 0;
  }
  else if (argc == 3 && strncmp(argv[1],"-t",2) == 0) {
    int c = atoi(argv[2]);
    if (c < (int)dimof(unittests)) {
      unittests[c].status = unittests[c].unittest(argc, argv);
      printf("[%u][%s] %s, error = %d\n", c,
	     (unittests[c].status == VX_SUCCESS?"PASSED":"FAILED"),
	     unittests[c].name, unittests[c].status);
      return unittests[c].status;
    }
    else
      return -1;
  }
  else if (argc == 2 && strncmp(argv[1],"-s",2) == 0) {
    stopOnErrors = vx_true_e;
  }

  for (i = 0; i < dimof(unittests); i++) {

    unittests[i].status = unittests[i].unittest(argc, argv);

    if (unittests[i].status == VX_SUCCESS) {
      printf("[PASSED][%02u] %s\n", i, unittests[i].name);
      passed++;
    }
    else {
      printf("[FAILED][%02u] %s, error = %d\n", i, unittests[i].name, unittests[i].status);
      if (stopOnErrors == vx_true_e) break;
    }
  }
  printf("Passed %u out of "VX_FMT_SIZE"\n", passed, dimof(unittests));
  if (passed == dimof(unittests)) return 0;
  else return -1;
}

