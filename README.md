# Khronos OpenVX Tutorial Material
Khronos [OpenVX](https://www.khronos.org/registry/vx/) 
is an open, royalty-free standard API for cross platform acceleration of computer
vision applications. OpenVX enables performance and power-optimized computer vision
functionality, especially important in embedded and real-time use cases.
This course covers both the function-based API and the graph API that
enable OpenVX developers to efficiently run computer vision algorithms
on heterogeneous computing architectures. A set of example algorithms
from computational photography and advanced driver assistance mapped to
the graph API will be discussed. Also covered is the relationship between
OpenVX and OpenCV, as well as OpenCL. The tutorial includes hands-on practice
sessions that get the participants started on solving real computer vision
problems using OpenVX.

NOTE: The OpenVX Neural Network acceleration tutorials from Embedded Vision Summit 2017 are now available. Refer to [wiki](https://github.com/rgiduthuri/openvx_tutorial/wiki) page for details.

This tutorial material is based on OpenVX 1.1 and it can be used on a PC with recent 64-bit OS (Windows, OS X, or Linux).
Tutorial exercises and build procedures on various platforms are explained in this document.
An additional section explains tutorial exercises with a VirtualBox VM.
  1. [Tutorial Exercises Overview](#1-tutorial-exercises-overview)
  2. [Build on any PC using VirtualBox software and a pre-built VM](#2-build-on-any-pc-using-virtualbox-software-and-a-pre-built-vm) (recommended)
  3. [Build on Mac or Linux PC](#3-build-on-mac-or-linux-pc)
  4. [Build on Windows PC using Visual Studio](#4-build-on-windows-pc-using-visual-studio)

NOTE: The directory ``~`` in the text below refers to directory containing the ``openvx_tutorial`` sub-directory.

Refer to [Wiki page](https://github.com/rgiduthuri/openvx_tutorial/wiki) for upcomming events and past event archives.

## 1. Tutorial Exercises Overview
It is best to start doing these exercises after going through the [tutorial presentations](https://github.com/rgiduthuri/openvx_tutorial/wiki) that discuss OpenVX ecosystem overview and hands-on programming sessions.

All the tutorial exercises are kept in ``~/openvx_tutorial/tutorial_exercises``.
The ``tutorial_exercises/CMakeLists.txt`` includes all the exercises as separate 
projects. All of the exercise folders contain only one .cpp file with main() 
as the entry point. All the include files are kept in the 
``tutorial_exercises/include`` directory.

The ``tutorial_exercises`` sub-directory contains four exercises:
  * ``exercise1``: framework basics, graph concepts, keypoint tracking
  * ``exercise2``: user kernels, build a wrapper kernel to OpenCV function
  * ``exercise3``: intro to tensor object, build a cosine activation user kernel for tensor objects
  * ``exercise4``: build OpenCL based user kernel for neural network cosine activation and tensor objects

There are additional folders with full solutions:
  * ``solution_exerciseN``: complete solution of ``exerciseN``. Just for reference.

Each exercise requires you to modify ``exerciseN/exerciseN.cpp`` file.
Here are few helpful instructions:
  * Look for ``TODO`` keyword in ``exerciseN/exerciseN.cpp`` comments for instructions
    for code snippets that you need to create.
  * The steps are numbered, do them in that order.
  * All header files are kept in ``tutorial_exercises/include``.
    To open a header file, move the cursor to corresponding ``#include`` statement
    in ``exerciseN/exerciseN.cpp`` and press *F2*.
    - The ``"opencv_camera_display.h"`` is a wrapper that imports media and
      displays results using OpenCV library.
      * ``#define DEFAULT_WAITKEY_DELAY 1`` is used to specify wait time in milliseconds after each frame processing; to slowdown use larger numbers; or use 0 to wait for a key after each frame.
    - The ``"VX/vx.h"`` & ``"VX/vxu.h"`` files are part of OpenVX header files
      downloaded from https://www.khronos.org/registry/vx/
  * To view the definition of any OpenVX API or data type, simply move the
    cursor to the name and press *F2*.
  * The video sequence [PETS09-S1-L1-View001.avi](http://ewh.ieee.org/r6/scv/sps/openvx-material/PETS09-S1-L1-View001.avi) will be in ``~/openvx_tutorial/tutorial_videos`` folder. 
    - All the exercises in this tutorial use this video sequence as input.
    - Optionally, you can also specify you own video sequence on command-line as an argument to exercises.
  * Once you finish all the exercises, try using Release Build to see better performance.

## 2. Build on any PC using VirtualBox software and a pre-built VM
We have prepared a VirtualBox VM with this new course material. Make sure to setup your computer before starting the tutorial.
  * Choose a laptop with a recent 64-bit OS.
  * Download and install a recent VirtualBox from https://www.virtualbox.org/wiki/Downloads.
  * Download virtual machine "Ubuntu-64-OpenVX.zip" (2 GB) from https://goo.gl/YfcTLh and extract files into a local folder (~6 GB extracted).
    - This VM image includes all the necessary tools and packages required to run the tutorial, including the following two OpenVX implementations options:
      * [Open-source OpenVX on GitHub](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules) from AMD (default)
      * Khronos OpenVX sample implementation from [khronos.org/registry/vx](https://www.khronos.org/registry/vx/)
  * Run VirtualBox and add "Ubuntu-64-OpenVX" virtual machine [Machine -> Add] from the local folder. 
    - If you cannot install 64-bit VM, even though you have a 64-bit Windows, you need to enable virtualization in the BIOS. 
    - In the Security section, enable Virtualization Technology and VT-d Feature. 
    - On Windows 8.1, you also need to turn Hyper-V off (search for Turn Windows features on or off).
  * Start the "Ubuntu-64-OpenVX" virtual machine ([username: openvx][password: openvx]).
  * Run "Qt Creator" (click Qt icon on left) and open exercises project.
    - Open Project: ``CMakeLists.txt`` in ``/home/openvx/openvx_tutorial/tutorial_exercises``
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

Please note that the VirtualBox VM might not have the latest version of tutorial exercises. You can copy files from this GitHub project into ``/home/openvx/openvx_tutorial`` directory when new updates are available.

## 3. Build on Mac or Linux PC
In order to build these tutorial exercises, you need the following:
  * Laptop with a recent 64-bit OS (OS X or Linux)
  * Download and install [OpenCV 3.1](http://opencv.org/downloads.html)
  * Download and install [Qt Creator](http://www.qt.io/download-open-source/) (optional)
  * Download and install an OpenVX implementation:
    - [Open-source OpenVX on GitHub](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules) from AMD
      * The source code will be in ``~/openvx_tutorial/tutorial_exercises/amdovx-modules`` directory.
      * *CPU only build will be used for this tutorial*.
    - Khronos OpenVX sample implementation from [khronos.org/registry/vx](https://www.khronos.org/registry/vx/).
      * Follow the instructions in openvx_sample/README to create pre-built OpenVX libraries
    - See [Khronos OpenVX Resources](https://www.khronos.org/openvx/resources) for available commertial implementations.
  * Download [PETS09-S1-L1-View001.avi](http://ewh.ieee.org/r6/scv/sps/openvx-material/PETS09-S1-L1-View001.avi) and keep it in ``~/openvx_tutorial/tutorial_videos`` folder. 

### 3.1 Build using open-source OpenVX
Create an empty folder ``~/openvx_tutorial/build-open-source``. The directory structure should be like:
```
~/openvx_tutorial/
  ├── LICENSE
  ├── README.md
  ├── build-open-source/
  ├── scripts/
  ├── tutorial_exercises/
  │   ├── amdovx-modules/
  │   ├── CMakeLists.txt
  │   ├── exercise1/
  │   ├── exercise2/
  │   ├── exercise3/
  │   ├── exercise4/
  │   ├── include/
  │   ├── solution_exercise1/
  │   ├── solution_exercise2/
  │   ├── solution_exercise3/
  │   ├── solution_exercise4/
  └── tutorial_videos/
      └── PETS09-S1-L1-View001.avi
```
  * To prepare for build: ``% cd ~/openvx_tutorial/build-open-source; cmake ../tutorial_exercises; make``
  * To build and run an example: ``% cd ~/openvx_tutorial/build-open-source/exercise1; make; ./exercise1``
    * You should see video in a window. Press ESCAPE or 'q' to exit the app.
  
### 3.2 Build using open-source OpenVX in Qt Creator
  * Open Project ``~/openvx_tutorial/tutorial_exercises/CMakeLists.txt``.
  * Click *"Configure Project"* to open CMake Wizard
  * CPU only build will be used for this tutorial, unless CMake build flag ENABLE_OPENCL=TRUE is selected.
  * Click *"Run CMake"* and *"Done"*
  * Compile and run the project by clicking the higher of the green triangles at left bottom, or with *CTRL-R*.
  * You should see video in a window. Press ESCAPE or 'q' to exit the app.
  
### 3.3 Build using pre-built OpenVX library in Qt Creator
  * Open Project ``~/openvx_tutorial/tutorial_exercises/CMakeLists.txt``.
  * Click *"Configure Project"* to open CMake Wizard
    - Specify arguments: ``-DOpenVX_LIBS_DIR=<path-to-openvx-libraries> -DOpenVX_LIBS=<list-of-openvx-libraries>``
    - Click *"Run CMake"* and *"Done"*
  * Compile and run the project by clicking the higher of the green triangles at left bottom, or with *CTRL-R*.
  * You should see video in a window. Press ESCAPE or 'q' to exit the app.

## 4. Build on Windows PC using Visual Studio
In order to build these tutorial exercises, you need the following:
  * Laptop with a recent Windows 64-bit OS
  * Download and install [OpenCV 3.1](http://opencv.org/downloads.html) and set ``OpenCV_DIR`` environment variable to ``<installed-folder>\opencv\build`` folder.
  * Download and install latest [CMake](https://cmake.org/download/)
  * Download and install [Visual Studio 2015 Community (Free) or Visual Studio 2013](https://www.visualstudio.com/downloads/download-visual-studio-vs)
  * Download and install an OpenVX implementation:
    - [Open-source OpenVX on GitHub](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-modules) from AMD
      * The source code will be in ``~/openvx_tutorial/tutorial_exercises/amdovx-modules`` directory.
	  * *CPU only build will be used for this tutorial, unless CMake build flag ENABLE_OPENCL=TRUE is selected (see below)*.
    - Khronos OpenVX sample implementation from [khronos.org/registry/vx](https://www.khronos.org/registry/vx/).
      * Follow the instructions in openvx_sample/README to create pre-built OpenVX libraries
    - See [Khronos OpenVX Resources](https://www.khronos.org/openvx/resources) for available commertial implementations.
  * Download [PETS09-S1-L1-View001.avi](http://ewh.ieee.org/r6/scv/sps/openvx-material/PETS09-S1-L1-View001.avi) and keep it in ``~/openvx_tutorial/tutorial_videos`` folder. 

### 4.1 Build using open-source OpenVX in Visual Studio
  * Run CMake (cmake-gui)
    - Click *Browse Source* button and select ``~/openvx_tutorial/tutorial_exercises``
    - Click *Browse Build* button and select ``~/openvx_tutorial/build-open-source``
    - To enable OpenCL code path in amdovx-core, click *Add Entry* button
      * Set *Name* to ``ENABLE_OPENCL``
      * Set *Type* to *BOOL*
      * Select *Value* (i.e., TRUE)
      * Click *OK*
    - Click *Configure* button; you get a window asking for compilers to use: select *"Visual Studio 14 2015 Win64"* or *"Visual Studio 12 2013 Win64"*
    - Click *Generate* button
  * Run Visual Studio and open solution ``~/openvx_tutorial/build-open-source/tutorial_exercises.sln``
  * Set *exercise1* as startup project
  * Build and run the project.
  * You should see video in a window. Press ESCAPE or 'q' to exit the app.
  
### 4.2 Build using pre-built OpenVX library in Visual Studio
  * Run CMake (cmake-gui)
    - Click *Browse Source* button and select ``~/openvx_tutorial/tutorial_exercises``
    - Click *Browse Build* button and select ``~/openvx_tutorial/build-pre-built``
    - Click *Add Entry* button
      * Set *Name* to ``OpenVX_LIBS_DIR``
      * Set *Type* to *STRING*
      * Set *Value* to ``<path-to-openvx-libraries>``
      * Click *OK*
    - Click *Add Entry* button
      * Set *Name* to ``OpenVX_LIBS``
      * Set *Type* to *STRING*
      * Set *Value* to ``<list-of-openvx-libraries>``
      * Click *OK*
    - Click *Configure* button; you get a window asking for compilers to use: select *"Visual Studio 14 2015 Win64"* or *"Visual Studio 12 2013 Win64"*
    - Click *Generate* button
  * Run Visual Studio and open solution ``~/openvx_tutorial/build-open-source/tutorial_exercises.sln``
  * Set *exercise1* as startup project
  * Build and run the project.
  * You should see video in a window. Press ESCAPE or 'q' to exit the app.
