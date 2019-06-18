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
#include "onnx/onnx_pb.h"
#include "common.hpp"
#include <fstream>
#include <vector>
#include <cmath>
#include <map>
#include <set>
#include <VX/vx.h>
#include <VX/vx_khr_nn.h>
#include "nn_ext2.h"

////////
// import node information
//
struct onnx_import_node_info {
    vx_graph graph;
};

////////
// onnx model url
//
static vx_char g_onnx_model_url[8192] = "*onnx-model-not-set*";

// utility functions
template<typename T>
T getAttrValue(const onnx::NodeProto& node, const std::string& name)
{
    T value;
    for(int i = 0; i < node.attribute_size(); i++) {
        auto& attr = node.attribute(i);
        if(attr.name() == name) {
            if(std::is_same<T,int>::value) {
                value = attr.i();
            }
            else if(std::is_same<T,float>::value) {
                value = attr.f();
            }
            break;
        }
    }
    return value;
}
std::string getAttrValue(const onnx::NodeProto& node, const std::string& name)
{
    std::string value;
    for(int i = 0; i < node.attribute_size(); i++) {
        auto& attr = node.attribute(i);
        if(attr.name() == name) {
            value = attr.s();
            break;
        }
    }
    return value;
}

template<typename T>
std::vector<T> getAttrVector(const onnx::NodeProto& node, const std::string& name)
{
    std::vector<T> attrVector;
    for(int i = 0; i < node.attribute_size(); i++) {
        auto& attr = node.attribute(i);
        if(attr.name() == name) {
            attrVector.clear();
            if(std::is_same<T,int>::value) {
                for(size_t j = 0; j < attr.ints().size(); j++) {
                    attrVector.push_back(attr.ints(j));
                }
            }
            else if(std::is_same<T,float>::value) {
                for(size_t j = 0; j < attr.floats().size(); j++) {
                    attrVector.push_back(attr.floats(j));
                }
            }
            break;
        }
    }
    return attrVector;
}

////////
// Simple model compiler to build OpenVX graph from an ONNX model
//
// NOTE: This implementation supports very limited features enough
//       to support this tutorial
//
vx_graph create_openvx_graph_from_onnx_model(
    vx_context context,
    const char * onnx_model_url,
    vx_tensor input_tensor,
    vx_tensor output_tensor)
{
    // load ONNX model from URL
    onnx::ModelProto model;
    std::ifstream ifonnx(onnx_model_url, std::ios_base::binary);
    if(!model.ParseFromIstream(&ifonnx)) {
        std::cout << "ERROR: failed for parse: " << onnx_model_url << std::endl;
        exit(1);
    }
    ifonnx.close();

    // parse onnx model
    auto graph = model.graph();

    // create OpenVX graph
    vx_status status;
    vx_graph openvx_graph = vxCreateGraph(context);
    ERROR_CHECK_OBJECT(openvx_graph);

    // onnx model info
    std::map<std::string,onnx::TensorProto> tensor_info;
    std::map<std::string,size_t> activation_used;
    std::map<std::string,std::vector<int>> tensor_shape;
    std::map<std::string,::google::protobuf::int32> tensor_elem_type;
    std::set<std::string> tensor_names;
    std::vector<std::string> tensor_inputs, tensor_outputs;
    std::map<std::string,vx_tensor> tensor2vx;
    std::string graph_input_name, graph_output_name;

    // utility functions
    auto dims2str = [](std::vector<int> dims) -> std::string
    {
        std::string s;
        for (size_t i = 0; i < dims.size(); i++) {
            s += (i == 0) ? "{" : ",";
            s += std::to_string(dims[i]);
        }
        s += "}";
        return s;
    };
    auto tensor2dims = [](const onnx::TensorProto& tensor) -> std::vector<int>
    {
        std::vector<int> dims;
        for(size_t i = 0; i < tensor.dims_size(); i++) {
            dims.push_back(tensor.dims(i));
        }
        return dims;
    };
    auto shape2dims = [](const onnx::TensorShapeProto& shape) -> std::vector<int>
    {
        std::vector<int> dims;
        for(size_t i = 0; i < shape.dim_size(); i++) {
            dims.push_back(shape.dim(i).dim_value());
        }
        return dims;
    };
    auto create_vx_tensor = [=](const std::string& name, ::google::protobuf::int32 type,
                           const std::vector<int>& dims, bool persistent,
                           std::map<std::string,vx_tensor>& tensor2vx,
                           const float * initial_values = nullptr
            ) -> vx_tensor
    {
        vx_tensor openvx_tensor = nullptr;
        auto it = tensor2vx.find(name);
        if(it != tensor2vx.end()) {
            openvx_tensor = it->second;
        }
        else {
            vx_enum data_type = VX_TYPE_INT16;
            if(type != onnx::TensorProto::FLOAT && type != onnx::TensorProto::INT64) {
                std::cout << "ERROR: tensor data_type " << type << " is not supported for " << name << std::endl;
                exit(1);
            }
            size_t dims_data[8] = { 0 }, strides[8] = { 0 }, element_count = 1;
            for(size_t i = 0; i < dims.size(); i++) {
                dims_data[i] = dims.data()[dims.size()-1-i];
                strides[i] = element_count * sizeof(int16_t);
                element_count *= dims_data[i];
            }
            if(persistent) {
                openvx_tensor = vxCreateTensor(context, dims.size(), dims_data, data_type, 8);
                ERROR_CHECK_OBJECT(openvx_tensor);
                if (initial_values) {
                    int16_t * int16_buf = new int16_t[element_count];
                    for(size_t i = 0; i < element_count; i++) {
                        int16_buf[i] = std::max(std::min((int16_t)std::round(initial_values[i] * 256.0f), (int16_t)INT16_MAX), (int16_t)INT16_MIN);
                    }
                    size_t zeros[8] = { 0 };
                    ERROR_CHECK_STATUS(vxCopyTensorPatch(openvx_tensor, dims.size(), zeros, dims_data, strides, int16_buf, VX_WRITE_ONLY, VX_MEMORY_TYPE_HOST));
                    delete[] int16_buf;
                }
            }
            else {
                openvx_tensor = vxCreateVirtualTensor(openvx_graph, dims.size(), dims_data, data_type, 8);
                ERROR_CHECK_OBJECT(openvx_tensor);
            }
            ERROR_CHECK_STATUS(vxSetReferenceName((vx_reference)openvx_tensor, name.c_str()));
            tensor2vx[name] = openvx_tensor;
            std::cout << "INFO: " << (persistent ? "vxCreateTensor" : "vxCreateVirtualTensor") << ": " << data_type << ":data_type " << dims2str(dims) << ":dims for " << name << std::endl;
        }
        return openvx_tensor;
    };

    // get weights
    for (size_t i = 0; i < graph.initializer_size(); i++) {
        const onnx::TensorProto& initializer = graph.initializer(i);
        tensor_info[initializer.name()] = initializer;
        tensor_elem_type[initializer.name()] = initializer.data_type();
        const float * initial_values = initializer.float_data().data();
        create_vx_tensor(initializer.name(), initializer.data_type(), tensor2dims(initializer), true, tensor2vx, initial_values);
    }

    // get activations and their dimensions
    for(size_t i = 0; i < graph.node_size(); i++) {
        auto& node = graph.node(i);
        for(size_t j = 0; j < node.input_size(); j++) {
            auto& tensor = node.input(j);
            if(tensor_info.find(tensor) == tensor_info.end()) {
                if(activation_used.find(tensor) == activation_used.end()) {
                    activation_used[tensor] = 1;
                    tensor_inputs.push_back(tensor);
                    //std::cout << "INFO: input: " << tensor << std::endl;
                    graph_input_name = tensor;
                }
                else {
                    activation_used[tensor] += 1;
                }
            }
        }
        for(size_t j = 0; j < node.output_size(); j++) {
            auto& tensor = node.output(j);
            activation_used[tensor] = 0;
        }
    }
    for(auto it = activation_used.begin(); it != activation_used.end(); it++) {
        if(it->second == 0) {
            tensor_outputs.push_back(it->first);
            //std::cout << "INFO: output: " << it->first << std::endl;
            graph_output_name = it->first;
        }
    }
    for(size_t i = 0; i < graph.input_size(); i++) {
        auto& value_info = graph.input(i);
        auto& name = value_info.name();
        auto& tensor = value_info.type().tensor_type();
        tensor_shape[name] = shape2dims(tensor.shape());
        tensor_elem_type[name] = tensor.elem_type();
        if(tensor2vx.find(name) == tensor2vx.end()) {
            tensor2vx[name] = input_tensor;
            std::cout << "INFO: " << "input_tensor: INT16:data_type " << dims2str(tensor_shape[name]) << ":dims for " << name << std::endl;
        }
    }

    // go though all nodes and build OpenVX graph
    for(size_t ilayer = 0; ilayer < graph.node_size(); ilayer++) {
        bool is_last_layer = (ilayer == (graph.node_size() - 1)) ? true : false;

        auto& node = graph.node(ilayer);
        auto& op = node.op_type();
        auto name = node.name();
        if(name.empty()) {
            name = node.output(0);
        }
        std::vector<std::string> inames, onames;
        for(size_t i = 0; i < node.input_size(); i++) {
            inames.push_back(node.input(i));
        }
        for(size_t i = 0; i < node.output_size(); i++) {
            onames.push_back(node.output(i));
        }

        auto create_output_tensor = [=](bool is_last_layer,
                const std::string& name, ::google::protobuf::int32 type,
                const std::vector<int>& dims, bool persistent,
                std::map<std::string,vx_tensor>& tensor2vx,
                vx_tensor output_tensor)
        {
            if(is_last_layer) {
                tensor2vx[name] = output_tensor;
            }
            else {
                create_vx_tensor(name, type, dims, persistent, tensor2vx);
            }
        };

        auto& output = node.output(0);
        auto& input = node.input(0);
        auto& dims = tensor_shape[input];
        auto elem_type = tensor_elem_type[input];
        auto odims = dims;
        bool persistent = (std::find(tensor_outputs.begin(), tensor_outputs.end(), output) != tensor_outputs.end());
        vx_node openvx_node = NULL;
        if(op == "Conv" || op == "MaxPool" || op == "AveragePool") {
            std::vector<int> kernel_shape = getAttrVector<int>(node, "kernel_shape");
            std::vector<int> strides = getAttrVector<int>(node, "strides");
            std::vector<int> pads = getAttrVector<int>(node, "pads");
            std::string auto_pad = getAttrValue<std::string>(node, "auto_pad");
            int K = dims[1];
            std::vector<int> dilations = { 1, 1 };
            int rounding = 0;
            if(op == "Conv") {
                int group = getAttrValue<int>(node, "group");
                if(group != 1) {
                    std::cout << "ERROR: unsupported layer: " << op << " with group=" << group << std::endl;
                    exit(1);
                }
                auto& filter = node.input(1);
                auto& filter_dims = tensor_shape[filter];
                K = filter_dims[0];
                dilations = getAttrVector<int>(node, "dilations");
                rounding = 1;
            }
            else if(op == "MaxPool") {
                rounding = 1;
            }
            else if(op == "AveragePool") {
                rounding = 0;
            }
            int dim_p = 1 + (dims[2] + pads[0] + pads[2] - ((kernel_shape[0] - 1) * dilations[0] + 1) + rounding*(strides[0] - 1)) / strides[0];
            int dim_q = 1 + (dims[3] + pads[1] + pads[3] - ((kernel_shape[1] - 1) * dilations[1] + 1) + rounding*(strides[1] - 1)) / strides[1];
            odims = { dims[0], K, dim_p, dim_q };
            // create output tensor
            tensor_elem_type[output] = elem_type;
            tensor_shape[output] = odims;
            create_output_tensor(is_last_layer, output, elem_type, tensor_shape[output], persistent, tensor2vx, output_tensor);
            // create node for the operation
            if(op == "Conv") {
                vx_tensor weights_tensor = tensor2vx[node.input(1)];
                vx_tensor biases_tensor = NULL;
                if(node.input_size() > 2) {
                    biases_tensor = tensor2vx[node.input(2)];
                }
                vx_nn_convolution_params_t convolution_params = {
                    (vx_size)pads[1],
                    (vx_size)pads[0],
                    VX_CONVERT_POLICY_SATURATE,
                    VX_ROUND_POLICY_TO_NEAREST_EVEN,
                    VX_NN_DS_SIZE_ROUNDING_CEILING,
                    (vx_size)dilations[1] - 1,
                    (vx_size)dilations[0] - 1
                };
                openvx_node = vxConvolutionLayer(openvx_graph,
                    tensor2vx[input],
                    weights_tensor, biases_tensor,
                    &convolution_params, sizeof(convolution_params),
                    tensor2vx[output]);
                ERROR_CHECK_OBJECT(openvx_node);
            }
            else {
                openvx_node = vxPoolingLayer(openvx_graph,
                    tensor2vx[input],
                    (op == "MaxPool") ? VX_NN_POOLING_MAX : VX_NN_POOLING_AVG,
                    (vx_size)kernel_shape[1],
                    (vx_size)kernel_shape[0],
                    (vx_size)pads[1],
                    (vx_size)pads[0],
                    VX_NN_DS_SIZE_ROUNDING_CEILING,
                    tensor2vx[output]);
                ERROR_CHECK_OBJECT(openvx_node);
            }
        }
        else if (op == "Relu") {
            // create output tensor
            tensor_elem_type[output] = elem_type;
            tensor_shape[output] = odims;
            create_output_tensor(is_last_layer, output, elem_type, tensor_shape[output], persistent, tensor2vx, output_tensor);
            // create Relu node
            openvx_node = vxActivationLayer(openvx_graph,
                tensor2vx[input], VX_NN_ACTIVATION_RELU, 0, 0, tensor2vx[output]);
            ERROR_CHECK_OBJECT(openvx_node);
        }
        else if(op == "Dropout") {
            // create output tensor
            tensor_elem_type[output] = elem_type;
            tensor_shape[output] = odims;
            create_output_tensor(is_last_layer, output, elem_type, tensor_shape[output], persistent, tensor2vx, output_tensor);
            // Dropout is NOP, so use copy node which can be optimized by an implementation
            openvx_node = vxCopyNode(openvx_graph,
                (vx_reference)tensor2vx[input], (vx_reference)tensor2vx[output]);
            ERROR_CHECK_OBJECT(openvx_node);
        }
        else if(op == "Reshape") {
            // create output tensor
            auto& shape = node.input(1);
            auto& shape_tensor = tensor_info[shape];
            std::vector<int> odims;
            auto shape_data = shape_tensor.int64_data().data();
            size_t total = 1;
            for(auto v : dims)
                total *= v;
            int minus_1_index = -1;
            size_t jprod = 1;
            for (size_t j = 0; j < shape_tensor.int64_data_size(); j++) {
                auto value = shape_data[j];
                if (value == 0) {
                    value = dims[j];
                }
                else if(value < 0) {
                    minus_1_index = j;
                }
                odims.push_back(value);
                if(value > 0) {
                    jprod *= value;
                }
            }
            if(minus_1_index >= 0) {
                odims[minus_1_index] = total / jprod;
            }
            tensor_elem_type[output] = elem_type;
            tensor_shape[output] = odims;
            create_output_tensor(is_last_layer, output, elem_type, tensor_shape[output], persistent, tensor2vx, output_tensor);
            // create Reshape node
            openvx_node = vxReshapeLayer(openvx_graph, tensor2vx[input], tensor2vx[output]);
            ERROR_CHECK_OBJECT(openvx_node);
        }
        else if(op == "Concat") {
            // create output tensor
            int axis = getAttrValue<int>(node, "axis");
            odims[axis] = 0;
            for(int i = 0; i < node.input_size(); i++) {
                odims[axis] += tensor_shape[node.input(i)][axis];
            }
            tensor_elem_type[output] = elem_type;
            tensor_shape[output] = odims;
            create_output_tensor(is_last_layer, output, elem_type, tensor_shape[output], persistent, tensor2vx, output_tensor);
            // get list input tensors
            vx_tensor * inputs = new vx_tensor[node.input_size()]();
            for(int i = 0; i < node.input_size(); i++) {
                inputs[i] = tensor2vx[node.input(i)];
            }
            // create Reshape node
            int concat_axis = dims.size()-1 - axis;
            openvx_node = vxConcatLayer(openvx_graph, inputs, node.input_size(), concat_axis, tensor2vx[output]);
            ERROR_CHECK_OBJECT(openvx_node);
            delete[] inputs;
        }
        else {
            std::cout << "ERROR: unsupported layer: " << op << std::endl;
            exit(1);
        }

        // if graph parameters are node
        if(vxGetStatus((vx_reference)openvx_node) == VX_SUCCESS) {
            vx_uint32 num_node_params = 0;
            ERROR_CHECK_STATUS(vxQueryNode(openvx_node, VX_NODE_PARAMETERS, &num_node_params, sizeof(num_node_params)));
            if(node.input(0) == graph_input_name) {
                vx_parameter parameter = vxGetParameterByIndex(openvx_node, 0);
                ERROR_CHECK_STATUS(vxGetStatus((vx_reference)parameter));
                ERROR_CHECK_STATUS(vxAddParameterToGraph(openvx_graph, parameter));
            }
            if(node.output(0) == graph_output_name) {
                vx_parameter parameter = vxGetParameterByIndex(openvx_node, num_node_params - 1);
                ERROR_CHECK_STATUS(vxGetStatus((vx_reference)parameter));
                ERROR_CHECK_STATUS(vxAddParameterToGraph(openvx_graph, parameter));
            }
            ERROR_CHECK_STATUS(vxReleaseNode(&openvx_node));
        }
    }

    // compile OpenVX graph
    ERROR_CHECK_STATUS(vxVerifyGraph(openvx_graph));

    return openvx_graph;
}

vx_status VX_CALLBACK onnx_import_validator(vx_node node,
        const vx_reference parameters[], vx_uint32 num, vx_meta_format metas[])
{
    // parameter #1 -- query dimensions and format
    vx_size num_of_dims;
    vx_size dims[4];
    vx_enum data_type;
    vx_uint8 fixed_point_pos;
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DIMS, &dims, num_of_dims * sizeof(vx_size)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type)));
    ERROR_CHECK_STATUS(vxQueryTensor((vx_tensor)parameters[1], VX_TENSOR_FIXED_POINT_POSITION, &fixed_point_pos, sizeof(fixed_point_pos)));

    // parameter #1 -- set required output tensor meta data
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute( metas[1], VX_TENSOR_NUMBER_OF_DIMS, &num_of_dims, sizeof(num_of_dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute( metas[1], VX_TENSOR_DIMS, &dims, sizeof(dims)));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute( metas[1], VX_TENSOR_DATA_TYPE, &data_type, sizeof(data_type )));
    ERROR_CHECK_STATUS(vxSetMetaFormatAttribute( metas[1], VX_TENSOR_FIXED_POINT_POSITION, &fixed_point_pos, sizeof(fixed_point_pos)));

    return VX_SUCCESS;
}

vx_status VX_CALLBACK onnx_import_host_compute(vx_node node, const vx_reference * refs, vx_uint32 num)
{
    onnx_import_node_info * node_info = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &node_info, sizeof(node_info)));

    // execute the graph created by import
    ERROR_CHECK_STATUS(vxProcessGraph(node_info->graph));

    return VX_SUCCESS;
}

vx_status VX_CALLBACK onnx_import_initialize(vx_node node, const vx_reference * refs, vx_uint32 num)
{
    // create onnx_import_node_info and set local buffer to the node
    onnx_import_node_info * node_info = new onnx_import_node_info();
    vx_size buffer_size = sizeof(*node_info);
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_SIZE, &buffer_size, sizeof(buffer_size)));
    ERROR_CHECK_STATUS(vxSetNodeAttribute(node, VX_NODE_LOCAL_DATA_PTR, &node_info, sizeof(node_info)));

    // generate OpenVX graph from ONNX model
    vx_status status;
    node_info->graph = create_openvx_graph_from_onnx_model(vxGetContext((vx_reference)node), g_onnx_model_url, (vx_tensor)refs[0], (vx_tensor)refs[1]);
    ERROR_CHECK_OBJECT(node_info->graph);
    vx_uint32 num_parameters = 0;
    ERROR_CHECK_STATUS(vxQueryGraph(node_info->graph, VX_GRAPH_NUMPARAMETERS, &num_parameters, sizeof(num_parameters)));
    if(num_parameters != 2) {
        std::cout << "ERROR: vxImportKernelFromURL: supports models with 1-input and 1-output only" << std::endl;
        exit(1);
    }

    return VX_SUCCESS;
}

vx_status VX_CALLBACK onnx_import_deinitialize(vx_node node, const vx_reference * refs, vx_uint32 num)
{
    onnx_import_node_info * node_info = NULL;
    ERROR_CHECK_STATUS(vxQueryNode(node, VX_NODE_LOCAL_DATA_PTR, &node_info, sizeof(node_info)));
    if(node_info) {
        if(node_info->graph) {
            ERROR_CHECK_STATUS(vxReleaseGraph(&node_info->graph));
        }
        delete node_info;
    }

    return VX_SUCCESS;
}

////////
// onnx2openvx is implemented as an import kernel extension
//   - type must be "ONNX"
//   - url mist be path to ONNX model
//
vx_kernel vxImportKernelFromURL(
    vx_context      context,
    const vx_char * type,
    const vx_char * url)
{
    // check to make sure ONNX model passed as input
    if(strcmp(type, "ONNX") != 0) {
        std::cout << "ERROR: vxImportKernelFromURL: supports type='ONNX' only" << std::endl;
        exit(1);
    }

    // check to make sure that only one kernel import is used
    if(g_onnx_model_url[0] != '*') {
        std::cout << "ERROR: vxImportKernelFromURL: supports import of only one ONNX model" << std::endl;
        exit(1);
    }
    strncpy(g_onnx_model_url, url, sizeof(g_onnx_model_url)-1);

    // register a custom kernel for import
    vx_enum onnx_import_kernel_id;
    ERROR_CHECK_STATUS(vxAllocateUserKernelLibraryId(context, &onnx_import_kernel_id));
    vx_kernel kernel = vxAddUserKernel(context,
            "openvx_tutorial.kernel_import",
            onnx_import_kernel_id,
            onnx_import_host_compute,
            2,
            onnx_import_validator,
            onnx_import_initialize,
            onnx_import_deinitialize);
    ERROR_CHECK_OBJECT(kernel);
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 0, VX_INPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxAddParameterToKernel(kernel, 1, VX_OUTPUT, VX_TYPE_TENSOR, VX_PARAMETER_STATE_REQUIRED));
    ERROR_CHECK_STATUS(vxFinalizeKernel(kernel));

    return kernel;
}
