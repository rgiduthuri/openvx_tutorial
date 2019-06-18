/*
* Copyright (c) 2019 <copyright holders>
*
* Permission is hereby granted, free of charge, to any person obtaining a copy
* of this software and associated documentation files (the "Software"), to deal
* in the Software without restriction, including without limitation the rights
* to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
* copies of the Software, and to permit persons to whom the Software is
* furnished to do so, subject to the following conditions:
*
* The above copyright notice and this permission notice shall be included in all
* copies or substantial portions of the Software.
*
* THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
* IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
* FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
* AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
* LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
* OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
* SOFTWARE.
*/

#include "onnx2openvx.hpp"
#include "common.hpp"
#include "nn_ext2.h"

////////
// log_callback() function implements a mechanism to print log messages
// from OpenVX framework onto console. The log_callback function can be
// activated by calling vxRegisterLogCallback() in STEP 02.
void VX_CALLBACK log_callback(vx_context context, vx_reference ref,
                              vx_status status, const vx_char msg[])
{
    std::cout << "LOG: [ status = " << status << " ] " << msg << std::endl;
}

int main(int argc, const char * argv[])
{
    // get command-line arguments
    //   onnxFileName: onnx model
    if(argc < 2) {
        std::cout << "Usage: onnx2openvx <model.onnx> [input_i16.raw [output_i16.raw]]" << std::endl;
        exit(1);
    }
    const char * onnxFileName = argv[1];
    const char * inputFileName = (argc > 2) ? argv[2] : nullptr;
    const char * outputFileName = (argc > 3) ? argv[3] : nullptr;
    std::cout << "INFO: command-line: onnx2openvx " << onnxFileName
              << " " << (inputFileName ? inputFileName : "(no-input)")
              << " " << (outputFileName ? outputFileName : "(no-output)")
              << std::endl;

    // set input and output tensor dimensions
    const size_t num_input_dims = 4;
    const size_t num_output_dims = 2;
    vx_size input_dims[num_input_dims] = { 224, 224, 3, 1 };
    vx_size output_dims[num_output_dims] = { 1000, 1 };

    // create OpenVX context and register for log messages
    vx_status status;
    vx_context context = vxCreateContext();
    ERROR_CHECK_OBJECT(context);
    vxRegisterLogCallback(context, log_callback, vx_true_e);

    // load nn_ext2 module
    ERROR_CHECK_STATUS(vxLoadKernelsNnExt2(context));

    // create input and output tensor objects
    vx_tensor input = vxCreateTensor(context, num_input_dims, input_dims, VX_TYPE_INT16, 8);
    vx_tensor output = vxCreateTensor(context, num_output_dims, output_dims, VX_TYPE_INT16, 8);
    ERROR_CHECK_OBJECT(input);
    ERROR_CHECK_OBJECT(output);
    ERROR_CHECK_STATUS(vxSetReferenceName((vx_reference)input, "onnx:input"));
    ERROR_CHECK_STATUS(vxSetReferenceName((vx_reference)output, "onnx:output"));

    // import ONNX model as OpenVX graph
    vx_kernel kernel = vxImportKernelFromURL(context, "ONNX", onnxFileName);
    ERROR_CHECK_OBJECT(kernel);

    // build OpenVX graph
    vx_graph graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(graph);
    vx_node node = vxCreateGenericNode(graph, kernel);
    ERROR_CHECK_OBJECT(node);
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 0, (vx_reference)input));
    ERROR_CHECK_STATUS(vxSetParameterByIndex(node, 1, (vx_reference)output));
    ERROR_CHECK_STATUS(vxReleaseNode(&node));
    ERROR_CHECK_STATUS(vxVerifyGraph(graph));

    // calculate input & output buffer sizes, strides, and begin dims (zeros)
    size_t begin_input_dims[num_input_dims] = { 0 };
    size_t begin_output_dims[num_input_dims] = { 0 };
    size_t input_strides[num_input_dims], input_buffer_size = sizeof(int16_t);
    size_t output_strides[num_output_dims], output_buffer_size = sizeof(int16_t);
    for(size_t i = 0; i < num_input_dims; i++) {
        input_strides[i] = input_buffer_size;
        input_buffer_size *= input_dims[i];
    }
    for(size_t i = 0; i < num_output_dims; i++) {
        output_strides[i] = output_buffer_size;
        output_buffer_size *= output_dims[i];
    }

    // allocate input & output buffers
    int16_t * input_buf = new int16_t[input_buffer_size/sizeof(int16_t)]();
    int16_t * output_buf = new int16_t[output_buffer_size/sizeof(int16_t)]();

    // file input & output
    FILE * fi = NULL, * fo = NULL;
    if(inputFileName) {
        if((fi = fopen(inputFileName, "rb")) == NULL) {
            std::cout << "ERROR: unable to open input: " << inputFileName << std::endl;
            exit(1);
        }
    }
    if(outputFileName) {
        if((fo = fopen(outputFileName, "wb")) == NULL) {
            std::cout << "ERROR: unable to create output: " << outputFileName << std::endl;
            exit(1);
        }
    }

    size_t input_count = 0;
    do {
        // read input if available, close if EOF reached
        if(fi) {
            if(fread(input_buf, 1, input_buffer_size, fi) < input_buffer_size) {
                fclose(fi);
                break;
            }
        }
        std::cout << "INFO: processing input #" << ++input_count << std::endl;

        // initialize input
        ERROR_CHECK_STATUS(vxCopyTensorPatch(input, num_input_dims, begin_input_dims, input_dims, input_strides,
                input_buf, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));

        // process graph
        ERROR_CHECK_STATUS(vxProcessGraph(graph));

        // read output
        ERROR_CHECK_STATUS(vxCopyTensorPatch(output, num_output_dims, begin_output_dims, output_dims, output_strides,
                output_buf, VX_READ_ONLY, VX_MEMORY_TYPE_HOST));

        // write output if created, close if input reached EOF
        if(fo) {
            fwrite(output_buf, 1, output_buffer_size, fo);
        }
    } while(fi);
    if(fo) {
        fclose(fo);
    }

    // release input & output buffers
    delete[] input_buf;
    delete[] output_buf;

    // release all resources
    //ERROR_CHECK_STATUS(vxReleaseContext(&context));
}
