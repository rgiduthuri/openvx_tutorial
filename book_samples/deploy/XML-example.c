#include <VX/vx.h>
#include <stdio.h>
#include <stdlib.h>
#include "readImage.h"
#include "writeImage.h"
#include <VX/vx_khr_xml.h>

vx_char xmlFilename[] = "ExampleXMLGraph.xml";

extern vx_graph makeTestGraph(vx_context context, vx_image image, vx_image output);

vx_status createXMLGraph(vx_uint32 width, vx_uint32 height, vx_char xmlfile[])
{
    vx_context context = vxCreateContext();
    return vxGetStatus((vx_reference)makeTestGraph(context, 
                                                   vxCreateImage(context, width, height, VX_DF_IMAGE_RGB), 
                                                   vxCreateImage(context, width, height, VX_DF_IMAGE_RGB))) ||
           vxExportToXML(context, xmlfile) ||
           vxReleaseContext(&context);
}

void main(int argc, char **argv)
{
    if (argc != 3)
        printf("Change an image\n%s <input> <output>\n", argv[0]);
        /* We create the XML graph here but in practice it will be done by a different application */
        /* Note also our example must specifiy the width and height up front, if the images are a different size
        then the graph will fail to verify later */
    else if (createXMLGraph(640, 480, xmlFilename))
        printf("Failed to export the context\n");        
    else
    {
        struct read_image_attributes attr;
        vx_context context = vxCreateContext();
        vx_image image = createImageFromFile(context, argv[1], &attr);
        vx_image output = vxCreateImage(context, attr.width, attr.height, attr.format);
        vx_import import = vxImportFromXML(context, xmlFilename);
        if (vxGetStatus((vx_reference)import))
            printf("Failed to import the XML\n");        
        else
        {
            vx_graph graph = (vx_graph)vxGetImportReferenceByName(import, "Test Graph");
            if (vxGetStatus((vx_reference)graph))
                printf("Failed to find the test graph\n");
            else if (vxSetGraphParameterByIndex(graph, 0, (vx_reference)image) ||
                     vxSetGraphParameterByIndex(graph, 1, (vx_reference)output))
                printf("Error setting the graph parameters\n");
            else if (vxProcessGraph(graph))
                printf("Error processing the graph\n");
            else if (writeImage(output, argv[2]))
                printf("Problem writing the output image\n");
        }
        vxReleaseContext(&context);
    }
}

