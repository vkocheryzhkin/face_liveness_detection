cmake_minimum_required(VERSION 2.8)
project( liveness_train )

add_subdirectory(dlib)

find_package( OpenCV REQUIRED )

add_executable( preprocess src/preprocess.cpp src/common.hpp )
target_link_libraries( preprocess stdc++fs ${OpenCV_LIBS} dlib)

add_executable( train src/train.cpp src/common.hpp)
target_link_libraries( train stdc++fs ${OpenCV_LIBS} dlib)

add_executable( predict src/predict.cpp src/common.hpp)
target_link_libraries( predict stdc++fs ${OpenCV_LIBS} dlib)
