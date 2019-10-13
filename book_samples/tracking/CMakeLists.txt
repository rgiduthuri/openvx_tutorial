add_executable(tracking_example tracking_example.cpp centroid_tracking.c)
target_link_libraries(tracking_example ${OpenCV_LIBS} ${OPENVX})
