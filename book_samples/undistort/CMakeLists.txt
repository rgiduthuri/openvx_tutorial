add_executable(undistort undistort-remap.c)
target_link_libraries(undistort ${OpenCV_LIBS} vxa)# ${OPENVX})

add_executable(undistortOpenCV undistortOpenCV.cpp)
target_link_libraries(undistortOpenCV ${OpenCV_LIBS})
