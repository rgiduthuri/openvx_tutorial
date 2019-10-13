if(${LAPACK_FOUND})
  add_executable(birdsEyeView birdsEyeView.c)
  target_link_libraries(birdsEyeView ${OpenCV_LIBS} vxa ${OPENVX} m ${LAPACK_LIBRARIES})
else()
  message("LAPACK is required for building birdsEyeView, skipping...")
endif()

add_executable(opencv-birdsEyeView opencv-birdsEyeView.cpp)
target_link_libraries(opencv-birdsEyeView ${OpenCV_LIBS})
