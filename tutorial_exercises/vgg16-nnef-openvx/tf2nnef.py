# Copyright (c) 2017 The Khronos Group Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from __future__ import division

import sys
import os
import shutil
import tarfile
import inspect
import traceback
import numpy as np
import tensorflow as tf
if sys.version_info[0] < 3:
    from Queue import Queue
else:
    from queue import Queue
from tensorflow.python.ops import gen_math_ops as tf_ops
from tensorflow.contrib.layers.python.layers import layers as tf_layers
from tensorflow.python.layers import utils


RedText = "\x1b[31m"
ResetStyle = "\x1b[0m"


def tf_version_greater(major, minor):
    str = tf.__version__
    i = str.index('.')
    j = str.index('.', i+1)
    return (int(str[:i]), int(str[i+1:j])) >= (major, minor)


def print_error(msg, stack):
    sys.stderr.write("%sError: %s%s\n" % (RedText, msg, ResetStyle))
    if stack is not None:
        traceback.print_list(stack)


def print_warning(msg):
    sys.stderr.write("%sWarning: %s%s\n" % (RedText, msg, ResetStyle))


def undecorate(decorated, orig_name=None):
    if orig_name is None:
        orig_name = decorated.__name__

    if not hasattr(decorated, "__closure__") or not decorated.__closure__:
        return decorated

    for obj in (c.cell_contents for c in decorated.__closure__):
        if hasattr(obj, "__name__") and obj.__name__ == orig_name:
            return obj
        if hasattr(obj, "__closure__") and obj.__closure__:
            found = undecorate(obj, orig_name)
            if found:
                return found
    return None


class InvocationTrace:

    def __init__(self, functions, handler):
        self.frame = None
        self.invocations = list()
        self.handler = handler

        self.func_names = set()
        self.qualified = {}
        for func in functions:
            undecorated = undecorate(func)
            self.func_names.add(undecorated.__name__)
            self.qualified[undecorated.__module__ + '.' + undecorated.__name__] = func

    def __call__(self, frame, event, result):
        func_name = frame.f_code.co_name
        if func_name == '__init__':
            result = frame.f_locals.get('self')
            if result is not None:
                func_name = result.__class__.__name__

        if func_name not in self.func_names:
            return

        mod = inspect.getmodule(frame)
        if mod is None:
            return

        func_name = mod.__name__ + '.' + func_name

        if event == 'call':
            if self.frame is None:
                func = self.qualified.get(func_name)
                if func is not None:
                    arg_values = inspect.getargvalues(frame)
                    self.frame = frame
                    self.func = func
                    self.args = {key: value for (key,value) in arg_values.locals.items() if key in arg_values.args}
        elif event == 'return':
            if self.frame == frame and result is not None:
                if isinstance(result, (list, dict)):
                    result = result.copy()
                results = result if isinstance(result, tuple) else (result, )
                stack = traceback.extract_stack(frame.f_back)
                self.invocations.append((self.func, self.args, results, stack))
                if self.handler:
                    self.handler(self.func, self.args, results, stack)
                self.frame = None

        return self


class TF2NNEFConverter:

    def __init__(self, producers, exporters, reader, output_path):
        self.producers = producers
        self.exporters = exporters
        self.reader = reader
        self.consumrs = {}
        self.tensor_names = {}
        self.tensor_counts = {}
        self.activations = []
        self.output_path = output_path
        self.fused = set()

        for invocation in producers.values():
            args = invocation[1]
            for arg in args.values():
                if isinstance(arg, (tf.Tensor, tf.Variable)):
                    self.consumrs.setdefault(arg, []).append(invocation)

    def producer(self, tensor):
        return self.producers.get(tensor)

    def consumers(self, tensor):
        return self.consumrs.get(tensor)

    def consumer(self, tensor):
        consumers = self.consumrs.get(tensor)
        return consumers[0] if consumers is not None and len(consumers) == 1 else None

    def exporter(self, func):
        item = self.exporters.get(func)
        return item[0] if isinstance(item, tuple) else item

    def make_fused(self, tensor):
        self.fused.add(tensor)

    def is_fused(self, tensor):
        return tensor in self.fused

    def make_constant(self, tf_tensor, nnef_value):
        self.tensor_names[tf_tensor] = nnef_value

    def make_tensor(self, tf_tensor, nnef_name, indexed=True):
        name = self.tensor_names.get(tf_tensor)
        if name is not None:
            return name

        if indexed:
            count = self.tensor_counts.get(nnef_name, 1)
            self.tensor_counts[nnef_name] = count + 1
            indexed_name = nnef_name + str(count)
        else:
            indexed_name = nnef_name

        self.tensor_names[tf_tensor.value() if isinstance(tf_tensor, tf.Variable) else tf_tensor] = indexed_name
        self.activations.append((tf_tensor, indexed_name))
        return indexed_name

    def make_passthrough_tensor(self, tf_tensor_in, tf_tensor_out):
        variable = isinstance(tf_tensor_out, tf.Variable)
        self.tensor_names[tf_tensor_out.value() if variable else tf_tensor_out] = self.nnef_tensor(tf_tensor_in)

    def nnef_tensor(self, tf_tensor):
        if isinstance(tf_tensor, (float, int)):
            return str(float(tf_tensor))
        elif isinstance(tf_tensor, tf.Variable):
            return self.tensor_names[tf_tensor.value()]
        else:
            return self.tensor_names[tf_tensor]

    def nnef_op(self, func):
        item = self.exporters.get(func)
        return item[1] if isinstance(item, tuple) else None

    @staticmethod
    def nnef_shape(shape, stack=None, is_filter=False, is_broadcast=False):
        if isinstance(shape, tf.Tensor):
            shape = tf.contrib.util.constant_value(shape)
            if shape is None:
                print_error('cannot handle dynamic tensor shape', stack)
                return []

        if isinstance(shape, tf.TensorShape):
            shape = shape.as_list()

        if not isinstance(shape, list):
            shape = list(shape)

        shape = [s.value if isinstance(s, tf.Dimension) else int(s) for s in shape]

        if len(shape) == 0:
            return []
        elif len(shape) == 1:
            return [1, shape[0]] if is_broadcast else [shape[0]]
        elif len(shape) == 2:
            return shape
        else:
            if is_filter:
                return [shape[-1], shape[-2]] + shape[:-2]
            else:
                return [shape[0], shape[-1]] + shape[1:-1]

    @staticmethod
    def nnef_axis(axis, rank):
        if axis < 0:
            axis = rank + axis

        if rank == 1:
            return 1
        elif rank == 2:
            return axis
        else:
            if axis == 0:
                return 0
            elif axis == rank - 1:
                return 1
            else:
                return axis + 1

    @staticmethod
    def nnef_axes(axis, rank):
        if isinstance(axis, (list, tuple)):
            return [TF2NNEFConverter.nnef_axis(a, rank) for a in axis]
        else:
            return [TF2NNEFConverter.nnef_axis(axis, rank)]

    @staticmethod
    def nnef_bool(value):
        if value is None:
            value = False
        return 'true' if value else 'false'

    @staticmethod
    def nnef_array(value, rank):
        if isinstance(value, list):
            return value
        elif isinstance(value, tuple):
            return list(value)
        else:
            return [value] * rank

    @staticmethod
    def nnef_padding(padding, rank):
        return [] if padding.upper() == 'SAME' else [(0, 0)] * rank

    @staticmethod
    def nnef_padding_ex(padding, input_sizes, filter_sizes, strides):

        def same_padding(input_size, filter_size, stride):
            output_size = int(np.ceil(float(input_size) / float(stride)))
            pad_total = (output_size - 1) * stride + filter_size - input_size
            if pad_total >= 0:
                pad_front = pad_total // 2
                pad_back = pad_total - pad_front
                return (pad_front, pad_back)
            else:
                return (0, pad_total)

        def valid_padding(input_size, filter_size, stride):
            output_size = int(np.ceil(float(input_size - filter_size + 1) / float(stride)))
            pad_total = (output_size - 1) * stride + filter_size - input_size
            return (0, pad_total)

        return [same_padding(input_size, filter_size, stride) if padding.upper() == 'SAME' else valid_padding(input_size, filter_size, stride)
                for (input_size, filter_size, stride) in zip(input_sizes, filter_sizes, strides)]

    @staticmethod
    def nnef_tensor_shuffle_dims(tensor, is_filter, is_broadcast):
        rank = tensor.ndim
        if rank == 0:
            return tensor
        elif rank == 1:
            return np.expand_dims(tensor, axis=0) if is_broadcast else tensor
        elif rank == 2:
            return tensor
        else:
            axes = [rank-1, rank-2] + list(range(rank-2)) if is_filter else [0, rank-1] + list(range(1,rank-1))
            return np.transpose(tensor, axes)

    @staticmethod
    def nnef_ids(ids):
        return "[" + ", ".join(map(str, ids)) + "]"

    @staticmethod
    def dilated_size(size, dilation):
        return [(s - 1) * d + 1 for (s, d) in zip(size, dilation)]

    def propagate_padding(self, input, padding, border, spatial, stack):
        producer = self.producer(input)
        if producer is not None and producer[0] == tf.pad:
            if len(padding) == 0:
                print_error("only 'VALID' padding is accepted after an explicit 'pad' operation", stack)

            args = producer[1]
            border = args['mode'].lower()
            if border == 'symmetric':
                border = 'reflect-even'

            paddings = args['paddings']
            paddings = paddings[1:-1] if spatial else [paddings[0], paddings[-1]] + paddings[1:-1]

            padding = [tuple(p) for p in paddings]

        return padding, border

    def propagate_space_to_batch(self, input, dilation, padding):
        producer = self.producer(input)
        if producer is not None and (producer[0] == tf.space_to_batch_nd or producer[0] == tf.space_to_batch):
            args = producer[1]

            input = args['input']
            dilation = args['block_shape'].tolist()
            padding = 'SAME' if args['paddings'].any() else 'VALID'

        return input, dilation, padding

    def propagate_batch_to_space(self, output):
        consumer = self.consumer(output)
        if consumer is not None and (consumer[0] == tf.batch_to_space_nd or consumer[0] == tf.batch_to_space):
            results = consumer[2]
            return results[0]

        return output

    def is_binary_op(self, func):
        return self.exporter(func) == export_binary

    def is_broadcast(self, tensor):
        shape = tensor.shape
        if isinstance(shape, tf.TensorShape):
            shape = shape.dims

        if len(shape) != 1:
            return False

        consumers = self.consumers(tensor)
        if consumers is None:
            return False

        for invocation in consumers:
            func, args = invocation[:2]

            if self.is_binary_op(func):
                x = args['x']
                other = args['y'] if tensor == x else x
            elif func == tf.nn.bias_add:
                other = args['value']
            elif func == tf.nn.batch_normalization or func == tf.nn.fused_batch_norm:
                other = args['x']
            else:
                return False

            if isinstance(other, tf.Variable):
                other = other.value()

            if not isinstance(other, tf.Tensor):
                return False

            if other.shape[-1] != tensor.shape[0]:
                return False

        return True

    def is_filter(self, tensor):
        consumers = self.consumers(tensor)
        if consumers is not None:
            for invocation in consumers:
                func, args = invocation[:2]
                if func in [tf.nn.conv1d, tf.nn.atrous_conv2d, tf.nn.atrous_conv2d_transpose] \
                        and args['filters'] == tensor:
                    return True
                elif func in [tf.nn.conv2d, tf.nn.conv3d, tf.nn.convolution,
                              tf.nn.conv2d_transpose, tf.nn.conv3d_transpose,
                              tf.nn.depthwise_conv2d, tf.nn.depthwise_conv2d_native,
                              tf.nn.depthwise_conv2d_native_backprop_input] \
                        and args['filter'] == tensor:
                    return True
                elif func == tf.nn.separable_conv2d:
                    if args['depthwise_filter'] == tensor or args['pointwise_filter'] == tensor:
                        return True
        return False

    def is_depthwise(self, tensor):
        consumers = self.consumers(tensor)
        if consumers is not None:
            for invocation in consumers:
                func, args = invocation[:2]
                if func in [tf.nn.depthwise_conv2d, tf.nn.depthwise_conv2d_native,
                            tf.nn.depthwise_conv2d_native_backprop_input] \
                        and args['filter'] == tensor:
                    return True
                elif func == tf.nn.separable_conv2d:
                    if args['depthwise_filter'] == tensor:
                        return True
        return False



def export_skip(func, args, results, stack, converter):
    return None


def export_passthrough(func, args, results, stack, converter):
    arg = converter.exporters.get(func)[1]
    converter.make_passthrough_tensor(args[arg], results[0])
    return None


def export_placeholder(func, args, results, stack, converter):
    result = results[0]
    shape = converter.nnef_shape(args['shape'], stack=stack, is_broadcast=converter.is_broadcast(result))
    name = args['name']
    if name is not None:
        output = converter.make_tensor(result, name, indexed=False)
    else:
        output = converter.make_tensor(result, 'input')
    return "{} = external(shape = {})".format(output, shape)


def export_variable(func, args, results, stack, converter):
    name = args['name']
    if name is None or name == '':
        print_error("non-empty 'name' argument must be provided for {}".
                    format("tf.Variable()" if func == tf.Variable else "tf.get_variable()"), stack)
        return None

    if func == tf.get_variable:
        initializer = args.get('initializer')
        if isinstance(initializer, np.ndarray):
            shape = initializer.shape
        elif isinstance(initializer, (int, float)):
            shape = [1,1]
        else:
            shape = args['shape']
    else:
        shape = tf.convert_to_tensor(args['initial_value']).shape

    result = results[0]

    is_filter = converter.is_filter(result)
    is_depthwise = converter.is_depthwise(result)
    is_broadcast = converter.is_broadcast(result)

    shape = converter.nnef_shape(shape, stack=stack, is_filter=is_filter, is_broadcast=is_broadcast)

    pos = name.rfind('/')
    output = converter.make_tensor(result, name[pos + 1:] if pos != -1 else name)

    if is_filter and is_depthwise:
        shape[0] *= shape[1]
        shape[1] = 1

    if converter.reader:
        key = result.name[:-2]
        if converter.reader.has_tensor(key):
            tensor = converter.reader.get_tensor(key)
            if is_filter and is_depthwise:
                tensor = np.reshape(tensor, newshape = tensor.shape[:-2] + (1, tensor.shape[-2] * tensor.shape[-1]))
            filename = converter.output_path + '/' + name + '.dat'
            write_nnef_tensor(filename, tensor, is_filter=is_filter, is_broadcast=is_broadcast)
        else:
            print_error("variable '{}' not found in checkpoint".format(key), stack)

    return "{} = variable(shape = {}, label = '{}')".format(output, shape, name)


def export_constant(func, args, results, stack, converter):
    shape = args['shape']
    value = args['value']
    result = results[0]

    singular = True
    if shape is not None:
        for s in shape:
            if s != 1:
                singular = False

    if not isinstance(value, (np.ndarray, list, tuple)) and singular:
        converter.make_constant(results[0], float(value))
        return None

    if not isinstance(value, np.ndarray):
        value = np.array(value, dtype=np.float32)

    if value.size == 1 and singular:
        converter.make_constant(results[0], value.flatten()[0])
        return None

    if shape is None:
        shape = list(value.shape)

    is_broadcast = converter.is_broadcast(result)

    shape = converter.nnef_shape(shape, stack=stack, is_broadcast=is_broadcast)
    value = converter.nnef_tensor_shuffle_dims(value, is_filter=False, is_broadcast=is_broadcast).flatten().tolist()

    output = converter.make_tensor(results[0], 'const')

    return '{} = constant(shape = {}, value = {})'.format(output, shape, value)


def export_conv(func, args, results, stack, converter):
    kernel = args['filter']
    if isinstance(kernel, tf.Variable):
        kernel = kernel.value()

    value = args.get('input')
    if value is None:
        value = args['value']

    input = converter.nnef_tensor(value)
    filter = converter.nnef_tensor(kernel)
    size = kernel.shape.as_list()[:-2]
    strides = list(args['strides'])[1:-1]
    rate = args.get('rate', args.get('dilation_rate'))
    rate = list(rate) if rate else [1] * len(size)
    filter_sizes = converter.dilated_size(size, rate)
    padding = args['padding']
    border = 'constant'

    value, rate, padding = converter.propagate_space_to_batch(value, rate, padding)

    result = converter.propagate_batch_to_space(results[0])

    bias = 0.0
    consumers = converter.consumers(result)
    if consumers is not None and len(consumers) == 1:
        invocation = consumers[0]
        _func, _args, _res = invocation[:3]

        if _func == tf.nn.bias_add and _args["value"] == result:
            bias = converter.nnef_tensor(_args["bias"])
            result = _res[0]
            converter.make_fused(result)
        elif _func in [tf.add, tf_ops.add]:
            if _args["x"] == result:
                bias = converter.nnef_tensor(_args["y"])
            elif _args["y"] == result:
                bias = converter.nnef_tensor(_args["x"])
            result = _res[0]
            converter.make_fused(result)

    output_shape = args.get('output_shape')
    if output_shape is not None:
        if isinstance(output_shape, tf.Tensor):
            output_shape = tf.contrib.util.constant_value(output_shape)
            if output_shape is None:
                output_shape = result.get_shape()
                if output_shape is not None:
                    output_shape = output_shape.as_list()
                    if None in output_shape:
                        output_shape = None

            if output_shape is None:
                print_warning("dynamic 'output_shape' cannot be evaluated, reverting to default")

        value_shape = value.shape.as_list()[1:-1]
        input_shape = output_shape[1:-1] if output_shape is not None else \
            [utils.deconv_output_length(value_shape[i], filter_sizes[i], padding.lower(), strides[i]) for i in range(len(value_shape))]

        padding = converter.nnef_padding_ex(padding, input_shape, filter_sizes, strides)
    else:
        padding = converter.nnef_padding(padding, len(size))
        padding, border = converter.propagate_padding(value, padding, border, spatial=True, stack=stack)

    op = converter.nnef_op(func)
    output = converter.make_tensor(result, 'conv' if op == 'planewise_conv' else op)

    return "{} = {}({}, {}, {}, padding = {}, border = '{}', stride = {}, dilation = {})" \
        .format(output, op, input, filter, bias, padding, border, strides, rate)


def export_convolution(func, args, results, stack, converter):
    args['strides'] = [1] + list(args['strides']) + [1]
    return export_conv(func, args, results, stack, converter)


def export_separable_conv(func, args, results, stack, converter):
    value = args['input']
    depth_kernel = args['depthwise_filter']
    point_kernel = args['pointwise_filter']

    if isinstance(depth_kernel, tf.Variable):
        depth_kernel = depth_kernel.value()
    if isinstance(point_kernel, tf.Variable):
        point_kernel = point_kernel.value()

    input = converter.nnef_tensor(value)
    depth_filter = converter.nnef_tensor(depth_kernel)
    point_filter = converter.nnef_tensor(point_kernel)
    size = depth_kernel.shape.as_list()[:-2]
    strides = converter.nnef_array(args['strides'][1:-1], 2)
    rate = converter.nnef_array(args['rate'], 2)
    padding = converter.nnef_padding(args['padding'], len(size))
    border = 'constant'

    padding, border = converter.propagate_padding(value, padding, border, spatial=True, stack=stack)

    output = converter.make_tensor(results[0], 'conv')

    return "{} = separable_conv({}, plane_filter = {}, point_filter = {}, padding = {}, border = '{}', stride = {}, dilation = {})" \
        .format(output, input, depth_filter, point_filter, padding, border, strides, rate)


def export_pool(func, args, results, stack, converter):
    value = args['value']
    input = converter.nnef_tensor(value)
    size = converter.nnef_shape(args['ksize'], stack=stack)
    strides = converter.nnef_shape(args['strides'], stack=stack)
    padding = converter.nnef_padding(args['padding'], len(size))
    border = 'ignore'

    padding, border = converter.propagate_padding(value, padding, border, spatial=False, stack=stack)

    op = converter.nnef_op(func)
    output = converter.make_tensor(results[0], 'pool')

    return "{} = {}({}, size = {}, padding = {}, border = '{}', stride = {})".format(output, op, input, size, padding, border, strides)


def export_activation(func, args, results, stack, converter):
    x = converter.nnef_tensor(args['features'])

    op = converter.nnef_op(func)
    output = converter.make_tensor(results[0], op)

    return '{} = {}({})'.format(output, op, x)


def export_unary(func, args, results, stack, converter):
    x = converter.nnef_tensor(args['x'])

    op = converter.nnef_op(func)
    output = converter.make_tensor(results[0], op)

    return '{} = {}({})'.format(output, op, x)


def export_binary(func, args, results, stack, converter):
    x = converter.nnef_tensor(args['x'])
    y = converter.nnef_tensor(args['y'])

    op = converter.nnef_op(func)
    output = converter.make_tensor(results[0], op)

    return '{} = {}({}, {})'.format(output, op, x, y)


def export_squared_diff(func, args, results, stack, converter):
    x = converter.nnef_tensor(args['x'])
    y = converter.nnef_tensor(args['y'])

    output = converter.make_tensor(results[0], 'diff')

    return '{} = sqr({} - {})'.format(output, x, y)


def export_where(func, args, results, stack, converter):
    c = converter.nnef_tensor(args['condition'])
    x = converter.nnef_tensor(args['x'])
    y = converter.nnef_tensor(args['y'])

    if x is None or y is None:
        print_error("arguments must not be None in tf.where() operation", stack)
        return None

    output = converter.make_tensor(results[0], 'select')

    return '{} = select({}, {}, {})'.format(output, c, x, y)


def export_reduce(func, args, results, stack, converter):
    tensor = args['input_tensor']
    input = converter.nnef_tensor(tensor)
    rank = len(tensor.shape.as_list())
    axis = args['axis']
    axes = sorted(converter.nnef_axes(axis, rank)) if axis else list(range(rank))

    op = converter.nnef_op(func)
    output = converter.make_tensor(results[0], 'reduce')

    return '{} = {}({}, axes = {})'.format(output, op, input, axes)


def export_lrn(func, args, results, stack, converter):
    input = converter.nnef_tensor(args['input'])
    depth_radius = args['depth_radius']
    depth_size = 2 * depth_radius + 1
    bias = float(args['bias'])
    alpha = float(args['alpha'] * depth_size)
    beta = float(args['beta'])
    size = [1, depth_size, 1, 1]

    output = converter.make_tensor(results[0], 'norm')

    return '{} = local_response_normalization({}, size = {}, alpha = {}, beta = {}, bias = {})'\
        .format(output, input, size, alpha, beta, bias)


def export_batch_normalization(func, args, results, stack, converter):
    input = converter.nnef_tensor(args['x'])
    mean = converter.nnef_tensor(args['mean'])
    variance = converter.nnef_tensor(args['variance'])
    offset = converter.nnef_tensor(args['offset']) if args.get('offset') is not None else float(0)
    scale = converter.nnef_tensor(args['scale']) if args.get('scale') is not None else float(1)
    epsilon = float(args.get('variance_epsilon', args.get('epsilon')))

    output = converter.make_tensor(results[0], 'norm')

    return '{} = batch_normalization({}, mean = {}, variance = {}, offset = {}, scale = {}, epsilon = {})'\
        .format(output, input, mean, variance, offset, scale, epsilon)


def export_l2_normalization(func, args, results, stack, converter):
    input = converter.nnef_tensor(args['x'])
    axes = sorted(converter.nnef_axes(args['dim']))
    epsilon = float(args.get('epsilon'))

    output = converter.make_tensor(results[0], 'norm')

    return "{} = l2_normalization({}, axes = {}, bias = {})".format(output, input, axes, epsilon)


def export_matmul(func, args, results, stack, converter):
    A = converter.nnef_tensor(args['a'])
    B = converter.nnef_tensor(args['b'])
    trA = converter.nnef_bool(args['transpose_a'])
    trB = converter.nnef_bool(args['transpose_b'])

    output = converter.make_tensor(results[0], 'matmul')

    return '{} = matmul({}, {}, trA = {}, trB = {})'.format(output, A, B, trA, trB)


def export_assign(func, args, results, stack, converter):
    ref = converter.nnef_tensor(args['ref'])
    value = converter.nnef_tensor(args['value'])

    output = converter.make_tensor(results[0], 'assign')

    return '{} = update({}, {})'.format(output, ref, value)


def export_add_n(func, args, results, stack, converter):
    inputs = args['inputs']
    value = converter.nnef_ids([converter.nnef_tensor(input) for input in inputs])

    output = converter.make_tensor(results[0], 'add')

    return '{} = add_n({})'.format(output, value)


def export_bias_add(func, args, results, stack, converter):
    input = converter.nnef_tensor(args['value'])
    bias = converter.nnef_tensor(args['bias'])

    output = converter.make_tensor(results[0], 'add')

    return '{} = add({}, {})'.format(output, input, bias)


def export_concat(func, args, results, stack, converter):
    values = args['values']
    rank = values[0].shape.ndims
    axis = converter.nnef_axis(args['axis'], rank)

    parts = converter.nnef_ids([converter.nnef_tensor(value) for value in values])

    output = converter.make_tensor(results[0], 'concat')

    return '{} = concat({}, axis = {})'.format(output, parts, axis)


def export_split(func, args, results, stack, converter):
    value = args['value']
    whole = converter.nnef_tensor(value)
    num_or_sizes = args['num_or_size_splits']
    ratios = num_or_sizes if isinstance(num_or_sizes, list) else [1] * num_or_sizes
    rank = value.shape.ndims
    axis = converter.nnef_axis(args['axis'], rank)

    output = converter.nnef_ids([converter.make_tensor(result, 'split') for result in results[0]])

    return '{} = split({}, axis = {}, ratios = {})'.format(output, whole, axis, ratios)


def export_softmax(func, args, results, stack, converter):
    logits = args['logits']
    rank = len(logits.shape.as_list())
    axis = sorted(converter.nnef_axes(args.get('dim', -1), rank))
    parts = converter.nnef_tensor(logits)

    output = converter.make_tensor(results[0], 'softmax')

    return '{} = softmax({}, axes = {})'.format(output, parts, axis)


def export_moments(func, args, results, stack, converter):
    value = args['x']
    input = converter.nnef_tensor(value)
    rank = value.shape.ndims
    axes = sorted(converter.nnef_axes(args['axes'], rank))

    mean = converter.make_tensor(results[0], 'mean')
    variance = converter.make_tensor(results[1], 'variance')

    return "{}, {} = moments({}, axes = {})".format(mean, variance, input, axes)


def export_reshape(func, args, results, stack, converter):
    input = converter.nnef_tensor(args['tensor'])
    shape = converter.nnef_shape(args['shape'], stack=stack)

    output = converter.make_tensor(results[0], 'reshape')

    return '{} = reshape({}, shape = {})'.format(output, input, shape)


def export_flatten(func, args, results, stack, converter):
    value = args['inputs']
    input = converter.nnef_tensor(value)

    output = converter.make_tensor(results[0], 'reshape')

    return '{} = reshape({}, shape = [0, -1])'.format(output, input)


def export_expand_dims(func, args, results, stack, converter):
    value = args['input']
    rank = value.shape.ndims
    input = converter.nnef_tensor(value)
    axis = args['axis']
    if axis is None:
        axis = rank

    shape = value.shape.as_list()
    shape.insert(axis, 1)

    shape = converter.nnef_shape(shape)

    output = converter.make_tensor(results[0], 'reshape')

    return '{} = reshape({}, shape = {})'.format(output, input, shape)


def export_squeeze(func, args, results, stack, converter):
    value = args['input']
    input = converter.nnef_tensor(value)
    axis = args['axis']

    if axis is not None:
        axis = sorted(axis)
    else:
        shape = value.shape.as_list()
        axis = [i for i in range(len(shape)) if shape[i] == 1]

    rank = value.shape.ndims

    if axis == list(range(rank - 1)) or axis == list(range(1,rank - 1)):
        converter.make_passthrough_tensor(value, results[0])
        return None

    axes = converter.nnef_axes(axis, rank)

    shape = ''
    for a in range(0,rank):
        if a not in axes:
            if len(shape) != 0:
                shape += ', '
            shape += 'shape_of({})[{}]'.format(input, a)

    output = converter.make_tensor(results[0], 'reshape')

    return '{} = reshape({}, shape = [{}])'.format(output, input, shape)


def export_transpose(func, args, results, stack, converter):
    value = args['a']
    input = converter.nnef_tensor(value)
    rank = value.shape.ndims
    perm = args['perm']
    if perm is None:
        perm = list(reversed(range(rank)))

    perm = converter.nnef_axes(perm, rank)

    p = list(perm)
    for i in range(len(perm)):
        perm[converter.nnef_axis(i, rank)] = p[i]

    output = converter.make_tensor(results[0], 'trans')

    return '{} = transpose({}, perm = {})'.format(output, input, perm)


def export_resize_images(func, args, results, stack, converter):
    value = args['images']
    input = converter.nnef_tensor(value)
    size = args['size']
    method = args['method']
    aligned = args['align_corners']

    if isinstance(size, tf.Tensor):
        print_error('cannot handle dynamic target size in tf.image.resize()', stack)
        return None

    input_size = [s.value if isinstance(s,tf.Dimension) else int(s) for s in value.shape[1:-1]]
    size = [s.value if isinstance(s, tf.Dimension) else int(s) for s in size]

    if size[0] == input_size[0] and size[1] == input_size[1]:
        converter.make_passthrough_tensor(value, results[0])
        return None

    if (size[0] > input_size[0] and size[1] < input_size[1]) or (size[0] < input_size[0] and size[1] > input_size[1]):
        print_error("resize must be up or down-sampling", stack)
        return None

    if size[0] > input_size[0]:
        if size[0] % input_size[0] or size[1] % input_size[1]:
            print_error('only integer factor resize allowed', stack)
            return None

        factor = [size[0] // input_size[0], size[1] // input_size[1]]

        output = converter.make_tensor(results[0], 'upsample')

        if method == tf.image.ResizeMethod.BILINEAR:
            return "{} = multilinear_upsample({}, factor = {}, method = '{}', border = 'replicate')"\
                    .format(output, input, factor, 'aligned' if aligned else 'asymmetric')
        elif method == tf.image.ResizeMethod.NEAREST_NEIGHBOR:
            return "{} = nearest_upsample({}, factor = {})".format(output, input, factor)
        else:
            print_error("unsupported upsample method '{}'".format(method), stack)
            return None
    else:
        if input_size[0] % size[0] or input_size[1] % size[1]:
            print_error('only integer factor resize allowed', stack)
            return None

        factor = [input_size[0] // size[0], input_size[1] // size[1]]

        output = converter.make_tensor(results[0], 'downsample')

        if method == tf.image.ResizeMethod.AREA:
            return "{} = area_downsample({}, factor = {})".format(output, input, factor)
        elif method == tf.image.ResizeMethod.NEAREST_NEIGHBOR:
            return "{} = nearest_downsample({}, factor = {})".format(output, input, factor)
        else:
            print_error("unsupported downsample method '{}'".format(method), stack)
            return None


def export_resize_bilinear(func, args, results, stack, converter):
    args['method'] = tf.image.ResizeMethod.BILINEAR
    return export_resize_images(func, args, results, stack, converter)


def export_resize_bicubic(func, args, results, stack, converter):
    args['method'] = tf.image.ResizeMethod.BICUBIC
    return export_resize_images(func, args, results, stack, converter)


def export_resize_nearest(func, args, results, stack, converter):
    args['method'] = tf.image.ResizeMethod.NEAREST_NEIGHBOR
    return export_resize_images(func, args, results, stack, converter)


def export_resize_area(func, args, results, stack, converter):
    args['method'] = tf.image.ResizeMethod.AREA
    return export_resize_images(func, args, results, stack, converter)

def export_space_to_batch(func, args, results, stack, converter):
    input = converter.nnef_tensor(args['input'])
    block_shape = args['block_shape']
    paddings = args['paddings']
    output = converter.make_tensor(results[0], 'space2batch')
    return "{} = space2batch({}, block_shape = {}, paddings = {})".format(output, input, block_shape, paddings)

def export_batch_to_space(func, args, results, stack, converter):
    input = converter.nnef_tensor(args['input'])
    block_shape = args['block_shape']
    output = converter.make_tensor(results[0], 'batch2space')
    return "{} = batch2space({}, block_shape = {})".format(output, input, block_shape)


DefaultExporters =\
{
    tf.Variable: (export_variable, 'variable'),
    tf.get_variable: (export_variable, 'variable'),
    tf.placeholder: (export_placeholder, 'external'),
    tf.constant: (export_constant, 'constant'),
    tf.identity: (export_passthrough, 'input'),
    tf.concat: (export_concat, 'concat'),
    tf.split: (export_split, 'split'),
    tf.reshape: (export_reshape, 'reshape'),
    tf.squeeze: (export_squeeze, 'reshape'),
    tf.expand_dims: (export_expand_dims, 'reshape'),
    tf.transpose: (export_transpose, 'transpose'),
    tf.stop_gradient: (export_passthrough, 'input'),
    tf.cast: (export_passthrough, 'x'),
    tf.pad: (export_passthrough, 'tensor'),
    tf.add: (export_binary, 'add'),
    tf.subtract: (export_binary, 'sub'),
    tf.multiply: (export_binary, 'mul'),
    tf.divide: (export_binary, 'div'),
    tf.pow: (export_binary, 'pow'),
    tf.squared_difference: (export_squared_diff, 'sqr'),
    tf.logical_and: (export_binary, 'and'),
    tf.logical_or: (export_binary, 'or'),
    tf.negative: (export_unary, 'neg'),
    tf.logical_not: (export_unary, 'not'),
    tf.abs: (export_unary, 'abs'),
    tf.sign: (export_unary, 'sign'),
    tf.exp: (export_unary, 'exp'),
    tf.log: (export_unary, 'log'),
    tf.sqrt: (export_unary, 'sqrt'),
    tf.rsqrt: (export_unary, 'rsqrt'),
    tf.square: (export_unary, 'sqr'),
    tf.floor: (export_unary, 'floor'),
    tf.ceil: (export_unary, 'ceil'),
    tf.round: (export_unary, 'round'),
    tf.where: (export_where, 'select'),
    tf.greater: (export_binary, 'gt'),
    tf.greater_equal: (export_binary, 'ge'),
    tf.less: (export_binary, 'lt'),
    tf.less_equal: (export_binary, 'le'),
    tf.equal: (export_binary, 'eq'),
    tf.not_equal: (export_binary, 'ne'),
    tf.minimum: (export_binary, 'min'),
    tf.maximum: (export_binary, 'max'),
    tf.assign: (export_assign, 'update'),
    tf_ops.add: (export_binary, 'add'),
    tf_ops.sub: (export_binary, 'sub'),
    tf_ops.mul: (export_binary, 'mul'),
    tf_ops.div: (export_binary, 'div'),
    tf_ops.real_div: (export_binary, 'div'),
    tf_ops._pow: (export_binary, 'pow'),
    tf_ops.logical_and: (export_binary, 'and'),
    tf_ops.logical_or: (export_binary, 'or'),
    tf_ops.neg: (export_unary, 'neg'),
    tf_ops.reciprocal: (export_unary, 'rcp'),
    tf_ops.logical_not: (export_unary, 'not'),
    tf_ops._abs: (export_unary, 'abs'),
    tf_ops.sign: (export_unary, 'sign'),
    tf_ops.exp: (export_unary, 'exp'),
    tf_ops.log: (export_unary, 'log'),
    tf_ops.square: (export_unary, 'sqr'),
    tf_ops.floor: (export_unary, 'floor'),
    tf_ops.ceil: (export_unary, 'ceil'),
    tf_ops.round: (export_unary, 'round'),
    tf_ops.greater: (export_binary, 'gt'),
    tf_ops.greater_equal: (export_binary, 'ge'),
    tf_ops.less: (export_binary, 'lt'),
    tf_ops.less_equal: (export_binary, 'le'),
    tf_ops.equal: (export_binary, 'eq'),
    tf_ops.not_equal: (export_binary, 'ne'),
    tf_ops.sqrt: (export_unary, 'sqrt'),
    tf_ops.rsqrt: (export_unary, 'rsqrt'),
    tf.sigmoid: (export_unary, 'sigmoid'),
    tf.tanh: (export_unary, 'tanh'),
    tf.reduce_sum: (export_reduce, 'sum_reduce'),
    tf.reduce_mean: (export_reduce, 'mean_reduce'),
    tf.reduce_max: (export_reduce, 'max_reduce'),
    tf.matmul: (export_matmul, 'matmul'),
    tf.add_n: (export_add_n, 'add_n'),
    tf.nn.sigmoid: (export_unary, 'sigmoid'),
    tf.nn.tanh: (export_unary, 'tanh'),
    tf.nn.elu: (export_activation, 'elu'),
    tf.nn.relu: (export_activation, 'relu'),
    tf.nn.softsign: (export_activation, 'softsign'),
    tf.nn.softplus: (export_activation, 'softplus'),
    tf.nn.conv1d: (export_conv, 'conv'),
    tf.nn.conv2d: (export_conv, 'conv'),
    tf.nn.conv3d: (export_conv, 'conv'),
    tf.nn.convolution: (export_convolution, 'conv'),
    tf.nn.conv2d_transpose: (export_conv, 'deconv'),
    tf.nn.conv3d_transpose: (export_conv, 'deconv'),
    tf.nn.depthwise_conv2d: (export_conv, 'planewise_conv'),
    tf.nn.depthwise_conv2d_native: (export_conv, 'planewise_conv'),
    tf.nn.separable_conv2d: (export_separable_conv, 'conv'),
    tf.nn.max_pool: (export_pool, 'max_pool'),
    tf.nn.max_pool_with_argmax: (export_pool, 'max_pool_with_indices'),
    tf.nn.avg_pool: (export_pool, 'avg_pool'),
    tf.nn.dropout: (export_passthrough, 'x'),
    tf.nn.bias_add: (export_bias_add, 'add'),
    tf.nn.lrn: (export_lrn, 'local_response_normalization'),
    tf.nn.local_response_normalization: (export_lrn, 'local_response_normalization'),
    tf.nn.batch_normalization: (export_batch_normalization, 'batch_normalization'),
    tf.nn.fused_batch_norm: (export_batch_normalization, 'batch_normalization'),
    tf.nn.l2_normalize: (export_l2_normalization, 'l2_normalization'),
    tf.nn.softmax: (export_softmax, 'softmax'),
    tf.nn.moments: (export_moments, 'moments'),
    tf.image.resize_images: export_resize_images,
    tf.image.resize_bilinear: export_resize_bilinear,
    tf.image.resize_nearest_neighbor: export_resize_nearest,
    tf.image.resize_bicubic: export_resize_bicubic,
    tf.image.resize_area: export_resize_area,
    tf.space_to_batch: (export_passthrough, 'input'),
    tf.space_to_batch_nd: (export_passthrough, 'input'),
    tf.batch_to_space: export_skip,
    tf.batch_to_space_nd: export_skip,
    tf_layers.softmax: (export_softmax, 'softmax'),
    tf_layers.flatten: (export_flatten, 'reshape'),
}


if tf_version_greater(1,3):
    DefaultExporters.update(
    {
        tf.sinh: (export_unary, 'sinh'),
        tf.cosh: (export_unary, 'cosh')
    })


def unrolled_rnn(cell, inputs, sequence_length=None, initial_state=None, dtype=tf.float32, scope=None):
    if sequence_length is None:
        sequence_length = tf.constant(shape=[inputs.shape[0]], value=[float(inputs.shape[1])], dtype=tf.float32)

    split_inputs = tf.split(inputs, axis=1, num_or_size_splits=inputs.shape[1])

    if initial_state is not None:
        c, h = initial_state
    else:
        c = tf.zeros(shape=[inputs.shape[0], inputs.shape[2]], dtype=dtype)
        h = tf.zeros(shape=[inputs.shape[0], inputs.shape[2]], dtype=dtype)

    _c = c
    _h = h

    output_list = []

    with tf.variable_scope(scope or "rnn"):
        for index, input in enumerate(split_inputs):
            output, (c, h) = cell(tf.squeeze(input, axis=[1]), (c, h))
            output_list.append(output)

            condition = tf.equal(sequence_length, index + 1)
            _c = tf.where(condition, c, _c)
            _h = tf.where(condition, h, _h)

    outputs = tf.concat(output_list, axis=1)

    return outputs, (_c, _h)


def trace_invocations(func, functions, handler=None):
    systrace = sys.gettrace()
    trace = InvocationTrace(functions, handler)
    sys.settrace(trace)
    results = func()
    sys.settrace(systrace)

    if isinstance(results, (tf.Tensor, tf.Variable)):
        outputs = { 'output': results }
    elif isinstance(results, (list, tuple)):
        outputs = {}
        for i, result in enumerate(results):
            if isinstance(result, (tf.Tensor, tf.Variable)):
                outputs['output' + str(i+1)] = result
    elif isinstance(results, dict):
        outputs = {}
        for name, result in results.items():
            if isinstance(result, (tf.Tensor, tf.Variable)):
                outputs[name] = result

    return trace.invocations, outputs


def enumerate_dependencies(dependencies, targets, exclusions):
    q = Queue()
    s = set()

    def insert(tensor,func,stack):
        if tensor not in s:
            if isinstance(tensor, tf.Variable):
                tensor = tensor.value()
            q.put((tensor,func,stack))
            s.add(tensor)


    for target in targets:
        insert(target,None,None)

    while not q.empty():
        tensor, func, stack = q.get()

        invocation = dependencies.get(tensor)

        if invocation is None:
            if func:
                op = func.__module__ + '.' + func.__name__
                print_error("tensor '{}' used by operation {} is not the result of any exported operation".format(tensor.name, op), stack)
            else:
                print_error("output tensor '{}' is not the result of any exported operation".format(tensor.name), stack)
            continue

        func, args, results, stack = invocation

        exc = exclusions.get(func)

        for key, arg in args.items():
            if exc is not None and key in exc:
                continue

            if isinstance(arg, (list, tuple)):
                for a in arg:
                    if isinstance(a, (tf.Tensor, tf.Variable)):
                        insert(a, func, stack)
            elif isinstance(arg, (tf.Tensor, tf.Variable)):
                insert(arg, func, stack)

    return s


def write_nnef_version(file, major, minor):
    np.asarray([0x4E, 0xEF], dtype=np.uint8).tofile(file)
    np.asarray([major,minor], dtype=np.uint8).tofile(file)


def write_nnef_hdrlen(file, rank, quantization=''):
    length = 4 + 4 + (rank + 1) * 4 + 4 + len(quantization)
    np.asarray([length], dtype=np.uint32).tofile(file)


def write_nnef_tensor_shape(file, shape):
    np.asarray([len(shape)], dtype=np.uint32).tofile(file)
    np.asarray(shape, dtype=np.uint32).tofile(file)


def write_nnef_tensor_dtype(file, bits, quantization=''):
    quantized = 1 if quantization != '' else 0

    np.asarray([quantized,bits], dtype=np.uint8).tofile(file)
    np.asarray([len(quantization)], dtype=np.uint16).tofile(file)
    if quantization is not None:
        file.write(quantization)


def write_nnef_tensor(filename, tensor, is_filter, is_broadcast):
    directory = os.path.dirname(filename)
    if not os.path.exists(directory):
        os.makedirs(directory)

    with open(filename, "w") as file:
        tensor = TF2NNEFConverter.nnef_tensor_shuffle_dims(tensor, is_filter, is_broadcast)
        write_nnef_version(file, major=1, minor=0)
        write_nnef_hdrlen(file, len(tensor.shape))
        write_nnef_tensor_shape(file, tensor.shape)
        write_nnef_tensor_dtype(file, bits=32)
        tensor.astype(np.float32).tofile(file)


def export_network(net_func, checkpoint=None, custom_exporters={}, custom_fragments='', output_path=None, compress=False, verbose=True, unroll_rnn=True):
    if unroll_rnn:
        dynamic_rnn_func = tf.nn.dynamic_rnn
        tf.nn.dynamic_rnn = unrolled_rnn

    exporters = DefaultExporters
    if custom_exporters:
        exporters.update(custom_exporters)

    if output_path is None and checkpoint is not None:
        output_path = os.path.splitext(checkpoint)[0] + '-nnef'

    if checkpoint is not None and not os.path.exists(output_path):
        os.makedirs(output_path)

    def trace_handler(func, args, results, stack):
        if func == tf.Variable or func == tf.get_variable:
            name = args['name']
            if name is not None:
                scope = tf.get_variable_scope().name
                args['name'] = scope + '/' + name if len(scope) != 0 else name

    if verbose:
        sys.stdout.write("Tracing invocations...")
        sys.stdout.flush()

    invocations, outputs = trace_invocations(net_func, exporters.keys(), trace_handler)

    if verbose:
        sys.stdout.write(" done\n")
        sys.stdout.write("Tracing dependencies...")
        sys.stdout.flush()

    dependencies = {}
    for invocation in invocations:
        results = invocation[2]
        for result in results:
            if isinstance(result, (list, tuple)):
                for tensor in result:
                    tensor = tensor.value() if isinstance(tensor, tf.Variable) else tensor
                    if tensor not in dependencies:
                        dependencies[tensor] = invocation
            else:
                tensor = result.value() if isinstance(result, tf.Variable) else result
                if tensor not in dependencies:
                    dependencies[tensor] = invocation

    exclusions =\
    {
        tf.nn.conv2d_transpose: ['output_shape'],
        tf.nn.conv3d_transpose: ['output_shape']
    }

    accessible = enumerate_dependencies(dependencies, outputs.values(), exclusions)

    if verbose:
        sys.stdout.write(" done\n")

    reader = tf.contrib.framework.load_checkpoint(checkpoint) if checkpoint is not None else None
    converter = TF2NNEFConverter(dependencies, exporters, reader, output_path)

    def has_accessible_result(results):
        for result in results:
            if isinstance(result, (list, tuple)):
                for r in result:
                    if (r.value() if isinstance(r, tf.Variable) else r) in accessible:
                        return True
            elif (result.value() if isinstance(result, tf.Variable) else result) in accessible:
                return True
        return False

    def all_results_fused(results):
        for result in results:
            if isinstance(result, (list, tuple)):
                for r in result:
                    if not converter.is_fused(r.value() if isinstance(r, tf.Variable) else r):
                        return False
            elif not converter.is_fused(result.value() if isinstance(result, tf.Variable) else result):
                return False
        return True

    if verbose:
        sys.stdout.write("Exporting invocations...")
        if checkpoint is None:
            sys.stdout.write('\n')
        sys.stdout.flush()

    params = []
    operations = []
    returns = []

    for name, tensor in sorted(outputs.items()):
        returns.append(converter.make_tensor(tensor, name, indexed=False))

    for invocation in invocations:
        func, args, results, stack = invocation
        if has_accessible_result(results) and not all_results_fused(results):
            item = exporters.get(func)
            if item is not None:
                exporter = item[0] if isinstance(item, tuple) else item
                text = exporter(func, args, results, stack, converter)
                if text is not None:
                    operations.append(text)
                    if func == tf.placeholder:
                        params.append(text[:text.find(' ')])


    file = open(output_path + '/graph.nnef', 'w') if checkpoint is not None else sys.stdout

    file.write('version 1.0\n\n')

    if len(custom_fragments):
        file.write(custom_fragments + '\n')

    file.write('graph ')

    file.write(net_func.__name__)

    file.write('( ')
    for i in range(len(params)):
        if i > 0:
            file.write(', ')
        file.write(params[i])
    file.write(' )')

    file.write(' -> ')
    file.write('( ')
    for i in range(len(returns)):
        if i > 0:
            file.write(', ')
        file.write(returns[i])
    file.write(' )\n')

    file.write('{\n')
    for line in operations:
        file.write('\t' + line + ';\n')
    file.write('}\n')

    if verbose and file != sys.stdout:
        sys.stdout.write(" done\n")

    if compress and file != sys.stdout:
        if verbose and file != sys.stdout:
            sys.stdout.write("Compressing files...")

        filename = output_path + '.tgz'
        tar = tarfile.open(filename, 'w:gz')
        for file in os.listdir(output_path):
            tar.add(output_path + '/' + file, file)
        tar.close()
        shutil.rmtree(output_path)

        if verbose and file != sys.stdout:
            sys.stdout.write(" done")

    if unroll_rnn:
        tf.nn.dynamic_rnn = dynamic_rnn_func

    return converter


def export_activations(converter, checkpoint, feed_dict, output_path=None, evaluate_count_per_iter=25, verbose=True):
    path = output_path if output_path is not None else os.path.splitext(checkpoint)[0] + '-activations'
    if not os.path.exists(path):
        os.makedirs(path)

    if verbose:
        sys.stdout.write('Evaluating activations..\n')
        sys.stdout.flush()

    activations = converter.activations

    with tf.Session() as sess:
        saver = tf.train.Saver()
        graph = tf.get_default_graph()

        total = 0
        for k,v in activations:
            if graph.is_fetchable(k) and isinstance(k, tf.Tensor):
                total += 1

        next = 0
        evaluated = 0
        while next < len(activations):
            tensors = {}
            while next < len(activations) and len(tensors) < evaluate_count_per_iter:
                k,v = activations[next]
                if graph.is_fetchable(k) and isinstance(k, tf.Tensor):
                    tensors[v] = k
                    evaluated += 1
                next += 1

            saver.restore(sess, checkpoint)
            values = sess.run(tensors, feed_dict)

            if verbose:
                sys.stdout.write("Evaluated {}/{}\n".format(evaluated, total))
                sys.stdout.flush()

            for k, v in values.items():
                filename = path + '/' + k + '.dat'
                write_nnef_tensor(filename, v, is_filter=False, is_broadcast=converter.is_broadcast(tensors[k]))
