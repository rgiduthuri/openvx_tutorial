/*
processGraph.cpp
Read an image, change it using a saved graph, write it out.
Using a C++ API
*/
#include "openvx_deploy.hpp"
#include <stdio.h>
#include <stdlib.h>
#include <string>
#include <vector>
#include <string.h>
#include <iostream>
#include <sys/types.h>
#include <sys/stat.h>

#include "readImage.h"
#include "writeImage.h"
using namespace openvx;
using namespace deployment;

VxImport loadObjectsFromFile(VxContext context, VxRefArray & refs, const char * fname)
{
    struct stat statbuf;
    auto statres { stat(fname, &statbuf) };
    auto fp { fopen(fname, "rb") };
    vx_uint8 blob[statbuf.st_size];
    if (!fp || statres || (fread(blob, statbuf.st_size, 1, fp) != 1) )
        std::cout << "Failed to read the file '" << fname << "'\n";
    fclose(fp);
    return context.importObjectsFromMemory(refs, blob, statbuf.st_size);
}

int main(int argc, char **argv)
{
    if (argc != 4) {
        std::cout << "Change an image using a saved graph\n" << argv[0] <<
               " <exported graph> <input image> <output image>\n";
    } else {
        struct read_image_attributes attr;
        VxContext context;
        VxImage input(createImageFromFile(context, argv[2], &attr));
        auto output { context.createImage(attr.width, attr.height, attr.format) };
        auto final_image { context.createImage(attr.width, attr.height, attr.format) };
        std::cout << "Image Width = " << attr.width << ", height = " << attr.height << "\n";
        VxRefArray refs(3);
        refs.put(1, input, VX_IX_USE_APPLICATION_CREATE);
        refs.put(2, output, VX_IX_USE_APPLICATION_CREATE);
        auto graph { loadObjectsFromFile(context, refs, argv[1]).getReferenceByName<VxGraph>("Test Graph") };
        if (input.getStatus() || output.getStatus() || final_image.getStatus()) {
            std::cout << "Could not create input or output images\n";
        } else if (graph.getStatus()) {
            std::cout << "Problem with status of imported graph\n";
        } else if (graph.processGraph()) {
            std::cout << "Error processing graph\n";
        } else {
            std::cout << "Graph was processed OK, about to set parameters and process again\n";
            if (VX_SUCCESS == graph.setGraphParameterByIndex(0, output) &&
                VX_SUCCESS == graph.setGraphParameterByIndex(1, final_image) &&
                VX_SUCCESS == graph.processGraph() ) {
                std::cout << "Once again, successful, writing output image\n";
                if (writeImage(final_image, argv[3])) {
                    std::cout << "Problem writing the output image\n";
                }
            } else {
                std::cout << "Error setting parameters or processing graph\n";
            }
        }
    }
    return 0;
}
