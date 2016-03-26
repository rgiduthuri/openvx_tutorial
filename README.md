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

This tutorial material can be used on a PC with recent 64-bit OS (Windows, OS X, or Linux).
Tutorial exercises and build procedures on various platforms are explained in this document.
An additional section explains tutorial exercises with a VirtualBox VM.
  1. [Tutorial Exercises Overview](#1-tutorial-exercises-overview)
  2. [Build on Mac or Linux PC using Qt Creator](#2-build-on-mac-or-pc-using-qt-creator)
  3. [Build on Windows PC using Visual Studio 2015 (Free)](#3-build-windows-pc-using-visual-studio-2015-free)
  4. [Build on any PC using VirtualBox software and a pre-built VM](#4-build-on-any-pc-using-virtualbox-software-and-a-pre-built-vm)

NOTE: The directory ``~`` in the text below refers to directory containing the ``openvx_tutorial`` sub-directory.

## 1. Tutorial Exercises Overview
It is best to start doing these exercises after the tutorial presentations.

All the tutorial exercises are kept in ``~/openvx_tutorial/tutorial_exercises``.
The ``tutorial_exercises/CMakeLists.txt`` includes all the exercises as separate 
projects. All of the exercise folders contain only one .cpp file with main() 
as the entry point. All the include files are kept in the 
``tutorial_exercises/include`` directory.

The ``tutorial_exercises`` sub-directory contains four exercises:
  * ``exercise1``: framework basics, import media, run a keypoint detector
  * ``exercise2``: graph concepts, keypoint tracking
  * ``exercise3``: user kernels, build a wrapper kernel to OpenCV function
  * ``exercise4``: user kernels, build a keypoint tracker

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
    - The ``"VX/vx.h"`` & ``"VX/vxu.h"`` files are part of OpenVX header files
      downloaded from https://www.khronos.org/registry/vx/
  * To view the definition of any OpenVX API or data type, simply move the
    cursor to the name and press *F2*.
  * You have to download [PETS09-S1-L1-View001.avi](http://ewh.ieee.org/r6/scv/sps/openvx-material/PETS09-S1-L1-View001.avi) and keep it in ``~/openvx_tutorial/tutorial_videos`` folder. 
    - All the exercises in this tutorial use this video sequence as input.
    - You can also specify you own video sequence on command-line as an argument.

## 2. Build on Mac or Linux PC using Qt Creator
In order to build these tutorial exercises, you need the following:
  * Laptop with a recent 64-bit OS (OS X or Linux)
  * Download and install [OpenCV 3.1](http://opencv.org/downloads.html)
  * Download and install [Qt Creator](http://www.qt.io/download-open-source/)
  * Download and install an OpenVX implementation:
    - Khronos OpenVX sample implementation from [khronos.org/registry/vx](https://www.khronos.org/registry/vx/).
    - [Open-source OpenVX on GitHub](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-core) from AMD
      * Copy the source code into ``~/openvx_tutorial/tutorial_exercises/amdovx-core`` directory.
      * *CPU only build is recommended for this tutorial*.
    - See [Khronos OpenVX Resources](https://www.khronos.org/openvx/resources) for available commertial implementations.
  * Download [PETS09-S1-L1-View001.avi](http://ewh.ieee.org/r6/scv/sps/openvx-material/PETS09-S1-L1-View001.avi) and keep it in ``~/openvx_tutorial/tutorial_videos`` folder. 

### 2.1 Build using open-source OpenVX in Qt Creator
  * Open Project ``~/openvx_tutorial/tutorial_exercises/CMakeLists.txt``; you get a window for running CMake.
  * Specify CMake arguments: ``-DOpenVX_SOURCE_DIR=amdovx-core/openvx -DOpenVX_LIBS=openvx -DCMAKE_DISABLE_FIND_PACKAGE_OpenCL=TRUE``
  * Click the *Run CMake* button, then click *Finish* button.
  * Compile and run the project by clicking the higher of the green triangles at left bottom, or with *CTRL-R*.
  * You should see video in a window. Press ESCAPE or 'q' to exit the app.
  
### 2.2 Build using pre-built OpenVX library in Qt Creator
  * Open Project ``~/openvx_tutorial/tutorial_exercises/CMakeLists.txt``; you get a window for running CMake.
  * Specify CMake arguments: ``-DOpenVX_LIBS_DIR=<path-to-openvx-libraries> -DOpenVX_LIBS=<list-of-openvx-libraries>``
  * Click the *Run CMake* button, then click *Finish* button.
  * Compile and run the project by clicking the higher of the green triangles at left bottom, or with *CTRL-R*.
  * You should see video in a window. Press ESCAPE or 'q' to exit the app.
  
## 3. Build on Windows PC using Visual Studio 2015 (Free)
In order to build these tutorial exercises, you need the following:
  * Laptop with a recent Windows 64-bit OS
  * Download and install [OpenCV 3.1](http://opencv.org/downloads.html) and set ``OpenCV_DIR`` environment variable to ``<installed-folder>\opencv\build`` folder.
  * Download and install latest [CMake](https://cmake.org/download/)
  * Download and install [Visual Studio Community (Free)](https://www.visualstudio.com/downloads/download-visual-studio-vs)
  * Download and install an OpenVX implementation:
    - Khronos OpenVX sample implementation from [khronos.org/registry/vx](https://www.khronos.org/registry/vx/).
    - [Open-source OpenVX on GitHub](https://github.com/GPUOpen-ProfessionalCompute-Libraries/amdovx-core) from AMD
      * Copy the source code into ``~/openvx_tutorial/tutorial_exercises/amdovx-core`` directory.
      * *CPU only build is recommended for this tutorial*.
    - See [Khronos OpenVX Resources](https://www.khronos.org/openvx/resources) for available commertial implementations.
  * Download [PETS09-S1-L1-View001.avi](http://ewh.ieee.org/r6/scv/sps/openvx-material/PETS09-S1-L1-View001.avi) and keep it in ``~/openvx_tutorial/tutorial_videos`` folder. 

### 3.1 Build using open-source OpenVX in Visual Studio 2015
  * Run CMake (cmake-gui)
    - Click *Browse Source* button and select ``~/openvx_tutorial/tutorial_exercises``
    - Click *Browse Build* button and select ``~/openvx_tutorial/build-open-source``
    - Click *Add Entry* button
      * Set *Name* to ``OpenVX_SOURCE_DIR``
      * Set *Type* to *STRING*
      * Set *Value* to ``amdovx-core/openvx``
      * Click *OK*
    - Click *Add Entry* button
      * Set *Name* to ``OpenVX_LIBS``
      * Set *Type* to *STRING*
      * Set *Value* to ``openvx``
      * Click *OK*
    - Click *Add Entry* button
      * Set *Name* to ``CMAKE_DISABLE_FIND_PACKAGE_OpenCL``
      * Set *Type* to *BOOL*
      * Select *Value* to indicate *TRUE*
      * Click *OK*
    - Click *Configure* button; you get a window asking for compilers to use: select *"Visual Studio 14 2015 Win64"*
    - Click *Generate* button
  * Run Visual Studio 2015 and open solution ``~/openvx_tutorial/build-open-source/tutorial_exercises.sln``
  * Set *exercise1* as startup project
  * Build and run the project.
  * You should see video in a window. Press ESCAPE or 'q' to exit the app.
  
### 3.2 Build using pre-built OpenVX library in Visual Studio 2015
  * Run CMake (cmake-gui)
    - Click *Browse Source* button and select ``~/openvx_tutorial/tutorial_exercises``
    - Click *Browse Build* button and select ``~/openvx_tutorial/build-open-source``
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
    - Click *Configure* button; you get a window asking for compilers to use: select *"Visual Studio 14 2015 Win64"*
    - Click *Generate* button
  * Run Visual Studio 2015 and open solution ``~/openvx_tutorial/build-open-source/tutorial_exercises.sln``
  * Set *exercise1* as startup project
  * Build and run the project.
  * You should see video in a window. Press ESCAPE or 'q' to exit the app.
  
## 4. Build on any PC using VirtualBox software and a pre-built VM
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
    - Open Project: ``CMakeLists.txt`` in ``/home/openvx/openvx_tutorial/tutorial_exercises``
    - click *"Configure Project"* to open CMake Wizard
    - CMake arguments for open-source OpenVX: ``-DOpenVX_SOURCE_DIR=amdovx-core/openvx -DOpenVX_LIBS=openvx``
      * uses Khronos sample implementation if the above CMake arguments are not specified
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

You can copy files from this GitHub project into ``/home/openvx/openvx_tutorial`` directory when new updates are available.

The first version of the virtual machine, used at CVPR 2015, was prepared with contributions from
Colin Tracey (NVIDIA), Elif Albuz (NVIDIA), Kari Pulli (Intel), Radha Giduthuri (AMD), Thierry Lepley (NVIDIA),
Victor Eruhimov (Itseez), and Vlad Vinogradov (Itseez).
