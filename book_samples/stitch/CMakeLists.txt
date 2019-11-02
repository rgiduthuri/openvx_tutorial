add_executable(stitch stitch.c)
target_link_libraries(stitch ${OpenCV_LIBS} vxa ${OPENVX})

add_executable(stitch-debug stitch-debug.c)
target_link_libraries(stitch-debug ${OpenCV_LIBS} vxa ${OPENVX})

add_executable(stitch-multiband stitch-multiband.c)
target_link_libraries(stitch-multiband ${OpenCV_LIBS} vxa ${OPENVX})

add_executable(homography homography-opencv.cpp)
target_link_libraries(homography ${OpenCV_LIBS})

add_executable(homography-multiband homography-multiband-opencv.cpp)
target_link_libraries(homography-multiband ${OpenCV_LIBS})
