/*
graphFactoryTest.c
Read an image, change it, write it out.
*/
#include <VX/vx.h>
#include <stdio.h>
#include <stdlib.h>
#include "readImage.h"
#include "writeImage.h"

extern vx_graph makeTestGraph(vx_context context, vx_image image, vx_image output);

void main(int argc, void **argv)
{
    if (argc != 3) {
        printf("Change an image\n"
               "%s <input> <output>\n", (char *)argv[0]);
    } else {
        struct read_image_attributes attr;
        vx_context context = vxCreateContext();
        vx_image image = createImageFromFile(context, (const char *)argv[1], &attr);
        vx_image output = vxCreateImage(context, attr.width, attr.height, attr.format);
        vx_graph graph = makeTestGraph(context, image, output);
        if (vxGetStatus((vx_reference)image)) {
            printf("Could not create input image\n");
        } else if (vxProcessGraph(graph)) {
            printf("Error processing graph\n");
        } else if (writeImage(output, (const char *)argv[2])) {
            printf("Problem writing the output image\n");
        }
        vxReleaseContext(&context);
    }
}