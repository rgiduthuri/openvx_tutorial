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

TARGET = onnx2openvx

ONNX_DIR = ../../../onnx
OPENVX_DIR = ../../../khronos
HFILES = onnx2openvx.hpp common.hpp \
            nn_ext2/nn_ext2.h \
            nn_ext2/nn_ext2_common.hpp \
            nn_ext2/concat_kernel.hpp \
            nn_ext2/reshape_kernel.hpp
CFILES = onnx2openvx.cpp main.cpp \
            nn_ext2/nn_ext2.cpp \
            nn_ext2/nn_ext2_common.cpp \
            nn_ext2/concat_kernel.cpp \
            nn_ext2/reshape_kernel.cpp

CC = g++ -std=c++11
CFLAGS = -DONNX_ML=1
CINC_DIRS = -I $(ONNX_DIR) -I $(ONNX_DIR)/build \
            -I nn_ext2 -I . \
            -I $(OPENVX_DIR)/OpenVX-sample-impl/api-docs/include
CLIB_DIRS = -L $(ONNX_DIR)/build \
            -L $(OPENVX_DIR)/OpenVX-sample-impl/build/Linux/x64/Release/sample/framework \
            -L $(OPENVX_DIR)/OpenVX-sample-impl/build/Linux/x64/Release/kernels/c_model
CLIBS = -lonnx -lonnx_proto -lprotobuf \
        -lopenvx -lopenvx-c_model-lib

$(TARGET): $(CFILES) $(HFILES)
	$(CC) $(CFLAGS) $(CINC_DIRS) $(CFILES) $(CLIB_DIRS) $(CLIBS) -o $(TARGET)

clean:
	$(RM) $(TARGET)
