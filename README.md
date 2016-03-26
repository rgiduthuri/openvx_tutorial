# Khronos OpenVX Tutorial Material
[OpenVX](https://www.khronos.org/registry/vx/) 
is a royalty-free open standard API released by the Khronos Group
in 2014. OpenVX enables performance and power-optimized computer vision
functionality, especially important in embedded and real-time use cases.
This course covers both the function-based API and the graph API that
enable OpenVX developers to efficiently run computer vision algorithms
on heterogeneous computing architectures. A set of example algorithms
from computational photography and advanced driver assistance mapped to
the graph API will be discussed. Also covered is the relationship between
OpenVX and OpenCV, as well as OpenCL. The tutorial includes hands-on practice
sessions that get the participants started on solving real computer vision
problems using OpenVX.

## Tutorial Exercises
It is best to start doing these exercises after the tutorial presentations.
The directory ~ in the text below refers to directory containing the openvx_tutorial
sub-directory.

All the tutorial exercises are kept in ~/openvx_tutorial/tutorial_exercises.
The tutorial_exercises/CMakeLists.txt includes all the exercises as separate 
projects. All of the exercise folders contain only one .cpp file with main() 
as the entry point. All the include files are kept in the 
tutorial_exercises/include directory.

The tutorial_exercises sub-directory contains four exercises:
  * exercise1: framework basics, import media, run a keypoint detector
  * exercise2: graph concepts, keypoint tracking
  * exercise3: user kernels, build a wrapper kernel to OpenCV function
  * exercise4: user kernels, build a keypoint tracker

There are additional folders with full solutions:
  * solution_exerciseN: complete solution of exerciseN. Just for reference.

Each exercise requires you to modify exerciseN/exerciseN.cpp file.
Here are few helpful instructions:
  * Look for *TODO* keyword in *exerciseN/exerciseN.cpp* comments for instructions
    for code snippets that you need to create.
  * The steps are numbered, do them in that order.
  * All header files are kept in *tutorial_exercises/include*.
    To open a header file, move the cursor to corresponding #include statement
    in exerciseN/exerciseN.cpp and press F2.
    - The "opencv_camera_display.h" is a wrapper that imports media and
      displays results using OpenCV library.
    - The "VX/vx.h" & "VX/vxu.h" files are part of OpenVX header files
      downloaded from https://www.khronos.org/registry/vx/
  * To view the definition of any OpenVX API or data type, simply move the
    cursor to the name and press F2.
  * You have to download [PETS09-S1-L1-View001.avi](http://ewh.ieee.org/r6/scv/sps/openvx-material/PETS09-S1-L1-View001.avi) and keep it in *~/openvx_tutorial/tutorial_videos* folder. 
    - All the exercises in this tutorial use this video sequence as input.
    - You can also specify you own video sequence on command-line as an argument.

## Custom Build of Tutorials using CMake
In order to custom-build these tutorial exercises using cmake and
any OpenVX implementation, you need the following:
  * Laptop with a recent 64-bit OS (Windows, OS X, or Linux)
  * Download and install [OpenCV 3.1](http://opencv.org/downloads.html)
  * Recommended IDE: Qt Creator (OS X/Linux) or Visual Studio 2013/2015 (Windows)
  * [Khronos OpenVX sample implementation](https://www.khronos.org/registry/vx/) (recommended), or any other OpenVX open-source implementation, or any pre-built 3rd party OpenVX libraries. See [Khronos OpenVX Resources](https://www.khronos.org/openvx/resources) for available implementations.

The ~/openvx_tutorial/tutorial_exercises/CMakeLists.txt has below cmake variables:
  * OpenVX_LIBS:         list of OpenVX libraries
  * OpenVX_LIBS_DIR:     path to OpenVX libraries
  * OpenVX_SOURCE_DIR:   sub-directory of OpenVX open-source implementation
                           inside tutorial_exercises with CMakeLists.txt
  * OpenVX_INCLUDE_DIRS: path to non-khronos OpenVX header files (optional)

The above cmake variables are expected in the following combinations:
  * OpenVX_SOURCE_DIR & OpenVX_LIBS: debug with OpenVX implementation source
  * OpenVX_LIBS_DIR & OpenVX_LIBS: run with any 3rd-party OpenVX library
  * None specified: use pre-installed Khronos OpenVX sample implementation

Here are few cmake build examples for this tutorial:
  * Build exercises using pre-installed khronos sample libraries in
    /home/openvx/openvx_sample/install/Linux/x64:
      * cmake ~/openvx_tutorial/tutorial_exercises
  * Build exercises using OpenVX open-source implementation from GitHub
    (works on any x86 CPU with SSE 4.1):
      * pushd ~/openvx_tutorial/tutorial_exercises
      * git clone https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-core
      * popd
      * cmake -DOpenVX_SOURCE_DIR=amdovx-core/openvx -DCMAKE_DISABLE_FIND_PACKAGE_OpenCL=TRUE ~/openvx_tutorial/tutorial_exercises
  * Build exercises using a 3rd-party OpenVX library
      * cmake -DOpenVX_LIBS_DIR=*path-to-openvx-libraries* -DOpenVX_LIBS=*list-of-openvx-libraries* ~/openvx_tutorial/tutorial_exercises

## Tutorial Preperations using Virtual Box
The current version of VirtualBox VM has been updated to use the exercises in this project. You should setup your computer before starting the tutorial.
  * Choose a laptop with a recent 64-bit OS.
  * Download and install a recent VirtualBox from https://www.virtualbox.org/wiki/Downloads.
  * Download virtual machine "Ubuntu-64-OpenVX.zip" (2 GB) from https://goo.gl/Ia53AB and extract files into a local folder.
  * Run VirtualBox and add "Ubuntu-64-OpenVX" virtual machine [Machine -> Add] from the local folder. 
    - If you cannot install 64-bit VM, even though you have a 64-bit Windows, you need to enable virtualization in the BIOS. 
    - In the Security section, enable Virtualization Technology and VT-d Feature. 
    - On Windows 8.1, you also need to turn Hyper-V off (search for Turn Windows features on or off).
  * Start the "Ubuntu-64-OpenVX" virtual machine ([username: openvx][password: openvx]).
  * Run "Qt Creator" (click Qt icon on left) and open exercises project.
    - Open Project: *CMakeLists.txt* in ~/openvx_tutorial/tutorial_exercises
    - click *"Configure Project"* to open CMake Wizard
    - click *"Run CMake"* and *"Finish"*
  * Select exercise1 as active sub-project.
    - click *"Open Build and Run Kit Selector"* under the *"Build"* menu
    - select Run *"exercise1"* under the Build *"Default"* and press ESCAPE
    - expand *"exercise1"* folder and click *"exercise1.cpp"*
    - you are going to modify this file during the first practice session
  * Build the project and run.
    - click *"Run"* under the *"Build"* menu (or use keyboard shortcut Ctrl+R)
    - you should see video in a window (you can move the window for better view)
    - press ESCAPE or 'q' to exit the app

The first version of the virtual machine, used at CVPR 2015, was prepared with contributions from
Colin Tracey (NVIDIA), Elif Albuz (NVIDIA), Kari Pulli (Intel), Radha Giduthuri (AMD), Thierry Lepley (NVIDIA),
Victor Eruhimov (Itseez), and Vlad Vinogradov (Itseez).
