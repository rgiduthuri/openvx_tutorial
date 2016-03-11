# Copyright (c) 2016 The Khronos Group Inc.
#
# Permission is hereby granted, free of charge, to any person obtaining a
# copy of this software and/or associated documentation files (the
# "Materials"), to deal in the Materials without restriction, including
# without limitation the rights to use, copy, modify, merge, publish,
# distribute, sublicense, and/or sell copies of the Materials, and to
# permit persons to whom the Materials are furnished to do so, subject to
# the following conditions:
#
# The above copyright notice and this permission notice shall be included
# in all copies or substantial portions of the Materials.
#
# THE MATERIALS ARE PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND,
# EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF
# MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT.
# IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY
# CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT,
# TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE
# MATERIALS OR THE USE OR OTHER DEALINGS IN THE MATERIALS.

OPENVX_INSTALL_DIR ?= /home/openvx/openvx_sample/install/Linux/x64/Release
IDIRS ?= /usr/include $(OPENVX_INSTALL_DIR)/include

CPP_FILES = $(wildcard *.cpp)

ODIR = obj
OBJ_FILES = $(CPP_FILES:.cpp=.o)
OBJ = $(patsubst %,$(ODIR)/%,$(OBJ_FILES))

LDIRS  = $(OPENVX_INSTALL_DIR)/bin
LIBS = m openvx vxu opencv_highgui opencv_imgproc opencv_core

TARGET = easy_exercise3

CC = g++
CFLAGS = -g

$(TARGET) : $(OBJ)
	$(CC) -o $@ $^ $(CFLAGS) -Wl,-rpath,$(OPENVX_INSTALL_DIR)/bin \
	$(patsubst %,-L%,$(LDIRS)) $(patsubst %,-l%,$(LIBS))

$(ODIR)/%.o: %.cpp
	@mkdir -p $(ODIR)
	$(CC) -c $(patsubst %,-I%,$(IDIRS)) -o $@ $< $(CFLAGS)

clean :
	rm -rf $(ODIR)
	rm -f $(TARGET)
