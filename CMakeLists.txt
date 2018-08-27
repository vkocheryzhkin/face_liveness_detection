cmake_minimum_required(VERSION 2.8)
project( liveness_train )

add_subdirectory(dlib)

find_package( OpenCV REQUIRED )
add_executable( liveness_train src/train.cpp )
target_link_libraries( liveness_train ${OpenCV_LIBS} dlib)