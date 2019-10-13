add_executable(hough houghLines.c)
target_link_libraries(hough ${OpenCV_LIBS} vxa ${OPENVX})

add_executable(houghEx houghLinesEx.c)
target_link_libraries(houghEx ${OpenCV_LIBS} vxa ${OPENVX})
