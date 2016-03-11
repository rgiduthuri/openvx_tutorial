# Khronos OpenVX Tutorial Material
[OpenVX](https://www.khronos.org/registry/vx/) is a royalty-free open standard API released by the Khronos Group in 2014. 
OpenVX enables performance and power-optimized computer vision functionality, 
especially important in embedded and real-time use cases. The course covers both 
the function-based API and the graph API that enable OpenVX developers to efficiently 
run computer vision algorithms on heterogeneous computing architectures. A set of example 
algorithms from computational photography and advanced driver assistance mapped to the 
graph API will be discussed. Also covered is the relationship between OpenVX and OpenCV, 
as well as OpenCL. The tutorial includes hands-on practice session that gets the participants 
started on solving real computer vision problems using OpenVX.

## Tutorial Preperations
You should setup your computer before starting the tutorial.
  * Choose a laptop with a recent 64-bit OS.
  * Download and install a recent VirtualBox from https://www.virtualbox.org/wiki/Downloads.
  * Download virtual machine "Ubuntu-64-OpenVX.zip" (2 GB) from https://goo.gl/3HRmdi and extract files into a local folder.
  * Run VirtualBox and add "Ubuntu-64-OpenVX" virtual machine [Machine -> Add] from the local folder. If you can’t install 64-bit VM, 
  even though you have a 64-bit Windows, you need to enable virtualization in the BIOS. In the Security section, 
  enable Virtualization Technology and VT-d Feature. On Windows 8.1, you also need to turn Hyper-V off 
  (search for Turn Windows features on or off).
  * Start the "Ubuntu-64-OpenVX" virtual machine. 
  * Run "Qt Creator" (click Qt icon on left) and open project /home/openvx/openvx_tutorial/tutorial_exercises/CMakeLists.txt.
  * Build and run to make sure a window opens playing a video. Press ESC to stop the app.

The first version of the virtual machine, used at CVPR 2015, was prepared with contributions from 
Colin Tracey (NVIDIA), Elif Albuz (NVIDIA), Kari Pulli (Intel), Radha Giduthuri (AMD), Thierry Lepley (NVIDIA), 
Victor Eruhimov (Itseez), and Vlad Vinogradov (Itseez).

## Tutorial Exercises
All the tutorial exercises are kept in tutorial_exercises folder. The tutorial_exercises/CMakeLists.txt includes
all the exercises as separate projects. It is best to start doing these exercises after going over the tutorial presentations.
  * exercise1: framework basics, import media, run a keypoint detector
  * exercise2: graph concepts, keypoint tracking
  * exercise3: user kernels, build a wrapper kernel to OpenCV function
  * exercise4: user kernels, build a keypoint tracker
  * easy_exercise#: partial solution with lot of hints in the code (use it only if you find exercise1/exercise2/... are hard)
  * solution_exercise#: complete solution of exercises (just for reference)
  
All of the exercise folders contain two important files:
  * opencv_camera_display.h: wrapper for importing media and displaying results using OpenCV library.
  * exercise*.cpp: OpenVX application. Look for TODO keyword in comments to code snipets that you need to write. 
    Walk through the code from top to bottom and follow the instructions in the comments.
