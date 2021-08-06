/*
processGraph.c
Read an image, change it using a saved graph, write it out.
*/
#include <VX/vx.h>
#include <VX/vx_khr_ix.h>
#include <stdio.h>
#include <stdlib.h>
#include <sys/types.h>
#include <sys/stat.h>
#include "readImage.h"
#include "writeImage.h"

vx_import loadObjectsFromFile(vx_context context, vx_size num_refs, vx_reference *refs, vx_enum *uses, const char * fname)
{
    struct stat statbuf;
    int statres = stat(fname, &statbuf);
    FILE *fp = fopen(fname, "rb");
    vx_uint8 *blob = (vx_uint8 *)malloc(statbuf.st_size);
    vx_import import = NULL;
    if (fp && 0 == statres && blob) {
        if (fread(blob, statbuf.st_size, 1, fp) == 1) {
            printf("Read %zu bytes ok\n", (size_t)statbuf.st_size);
            import = vxImportObjectsFromMemory(context, num_refs, refs, uses, blob, statbuf.st_size);
        } else {
            printf("Failed to read the file '%s'\n", fname);
        }
    } else {
        printf("Problem opening '%s' for reading, or allocating %zu bytes of memory\n",
               fname, (size_t)statbuf.st_size);
    }
    fclose(fp);
    if (blob) { 
        free(blob); 
    }
    return import;
}

void main(int argc, void **argv)
{
    if (argc != 4) {
        printf("Change an image using a saved graph\n"
               "%s <exported graph> <input image> <output image>\n", (char *)argv[0]);
    } else {
        struct read_image_attributes attr;
        vx_context context = vxCreateContext();
        vx_image input = createImageFromFile(context, (const char *)argv[2], &attr);
        vx_image output = vxCreateImage(context, attr.width, attr.height, attr.format);
        vx_image final =  vxCreateImage(context, attr.width, attr.height, attr.format);
        printf("Image Width = %d, height = %d\n", attr.width, attr.height);
        enum {num_refs = 3};
        vx_reference refs[num_refs] = { 
            NULL, 
            (vx_reference)input, 
            (vx_reference)output
        };
        vx_enum uses[num_refs] = {
            VX_IX_USE_EXPORT_VALUES,
            VX_IX_USE_APPLICATION_CREATE,
            VX_IX_USE_APPLICATION_CREATE
        };
        vx_import import = loadObjectsFromFile(context, num_refs, refs, uses, (const char *)argv[1]);
        if (vxGetStatus((vx_reference)input) || vxGetStatus((vx_reference)output) || vxGetStatus((vx_reference)final)) {
            printf("Could not create input or output images\n");
        } else if (vxGetStatus(refs[0])) {
            printf("Problem with status of imported graph\n");
        } else {
            vx_graph graph =  (vx_graph)refs[0];
            if (VX_SUCCESS != vxProcessGraph(graph)) {
                printf("Error processing graph\n");
            } else {
                printf("Graph was processed OK, about to set parameters and process again\n");
                if (VX_SUCCESS == vxSetGraphParameterByIndex(graph, 0, (vx_reference)output) &&
                    VX_SUCCESS == vxSetGraphParameterByIndex(graph, 1, (vx_reference)final) &&
                    VX_SUCCESS == vxProcessGraph(graph) ) {
                    printf("Once again, successful, writing output image\n");
                    if (writeImage(final, (const char *)argv[3])) {
                        printf("Problem writing the output image\n");
                    }
                } else {
                    printf("Error setting parameters or processing graph\n");
                }
            }
        }
        vxReleaseImport(&import);
        vxReleaseContext(&context);
    }
}
