# How to convert VGG16 tensorflow model to NNEF and run inference on OpenVX.

## Pre-requisites:
* [Tensorflow](https://www.tensorflow.org/)  installed. 
* Neural Network extension module of [amdovx-modules](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules) installed.

## Steps to run VGG-16 example:

First,clone the following repository:

```
% git clone -b cl/vgg16 https://github.com/lcskrishna/NNEF-Tools.git
% mkdir exercise; cd exercise
```

### Where to find VGG 16 sample model?
A sample VGG-16 model can be found here : [Link](http://download.tensorflow.org/models/vgg_16_2016_08_28.tar.gz)
Save the model into exercise folder and untar the model.

### How to convert tensorflow model to NNEF?
```
% python <Path to NNEF-tools>/converter/tensorflow/src/vgg16_export.py --ckpt-file <VGG-16 ckpt file>
```

This generates a folder named vgg_16-nnef that contains a binary folder and a graph.nnef file

### How to generate OpenVX code from NNEF?

Navigate to the executables that are obtained after installing amdovx-modules, and execute the following command to generate OpenVX inference code.

```
% nnef2openvx vgg_16-nnef openx_vgg16
```

This generates a folder named openvx_vgg16 where OpenVX inference code is generated.

### How to execute the OpenVX inference code?

Firstly, convert an image to a FP32 tensor as shown in the example of [ReadMe](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules/tree/develop/vx_nn). Name it as input.f32
Execute the following commands:

```
% cd openvx_vgg16 ; mkdir build; cd build
% cmake .. ; make
% ./anntest ../weights.bin input.f32 output.f32
```

This generates a output tensor. Now, use this tensor to post-process the data, by adding an argmax layer and identifying the classification that is generated.
You can use argmax.cpp file to get the classification label.

```
% g++ argmax.c++ -o argmax
% ./argmax output.f32 labels.txt
```

This prints the label classified to an image.
