# Copyright (c) 2019 <copyright holders>
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

import sys
import numpy as np
import onnx
from onnx import numpy_helper

# get command-line arguments
if len(sys.argv) < 3:
    print('Usage: python pb2int16.py file.pb file.raw [scale]')
    exit(1)
file_pb = sys.argv[1]
file_vx = sys.argv[2]
scale_factor = float(sys.argv[3]) if len(sys.argv) > 3 else 1.0

# load a TensorProto
tensor = onnx.TensorProto()
with open(file_pb, 'rb') as f:
    tensor.ParseFromString(f.read())
data_f = numpy_helper.to_array(tensor)
data_s = data_f * 256 * scale_factor
data_i = data_s.astype(np.int16)
data_i.tofile(file_vx)
print('created ' + file_vx)

#print('{}: {}'.format(file_pb, data_i))
