/*
export_graph.c
Create a graph and export it using the export and import extension
The memory "blob" is written to a file so it may be later read and imported
*/
#include <VX/vx.h>
#include <stdio.h>
#include <stdlib.h>
#include <VX/vx_khr_ix.h>

extern vx_graph makeTestGraph(vx_context context, vx_image image, vx_image output);

void main(int argc, char **argv)
{
    vx_context context = vxCreateContext();
    vx_image input = vxCreateImage(context, 640, 480, VX_DF_IMAGE_RGB);
    vx_image output = vxCreateImage(context, 640, 480, VX_DF_IMAGE_RGB);
    vx_graph graph = makeTestGraph(context, input, output);
    vx_reference refs[3] = {
        (vx_reference)graph,
        (vx_reference)input,
        (vx_reference)output
    };
    vx_enum uses[3] = {
        VX_IX_USE_EXPORT_VALUES,
        VX_IX_USE_APPLICATION_CREATE,
        VX_IX_USE_APPLICATION_CREATE
    };
    const vx_uint8 *blob = NULL;
    vx_size length;
    if (vxExportObjectsToMemory(context, 3, refs, uses, &blob, &length)) {
        /* There was an error creating the export, report to the user... */
        printf("Got an error when exporting the graph. No file was written.\n");
    } else if (argc != 2) {
        printf("Expected a valid filename: %s <file>\n", argv[0]);
    } else {
        /* We have a valid export of length bytes at address blob. Do something with it like writing it
        to a file... */
        FILE *fp = fopen(argv[1], "wb");
        if (fp && (fwrite(blob, length, 1, fp) == 1) && (fclose(fp) == 0)) {
            printf("Wrote the exported graph to file '%s', total %zu bytes\n", argv[1], length);
        } else {
            fclose(fp);
            printf("Error opening, writing or closing the file '%s'\n", argv[1]);
        }
    }
    /* now release the export blob memory, now we have copied it somewhere */
    vxReleaseExportedMemory(context, &blob);
    /* Release the context and all other resources */
    vxReleaseContext(&context);
}

