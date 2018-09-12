#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>
#include <opencv2/ml/ml.hpp>

#include <iostream>
#include <experimental/filesystem>
#include "common.hpp"

using namespace std;

int predict(std::string svmFile, std::string left, std::string right) {
    std::vector<float> res = getDistances68(left, right);
    cv::Mat test_mat(1, 68, CV_32FC1);
    for(int i = 0; i < res.size(); i++) {
        test_mat.at<float>(0, i) = res.at(i);
    }

    try {
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(svmFile);
        float t = svm->predict(test_mat);
        return (int)t;
    } catch(cv::Exception& e) {
        cout << e.what() << endl;
        return -1;
    }
}

int main(int argc, char** argv)
{
    if ( argc != 4 )
    {
        printf("usage3: svmFile image1 image2\n");
        return -1;
    }

    std::string svmFile = argv[1];
    std::string left = argv[2];
    std::string right = argv[3];
    int t = predict(svmFile, left, right);
    if (t == 0){
        cout << "live face" << endl; //liveness.svm ~/Work/face_liveness_detection_data/live1/0000.png ~/Work/face_liveness_detection_data/live1/0145.png
    } else if (t == 1){
        cout << "printed face" << endl; //liveness.svm ~/Work/face_liveness_detection_data/fake1/0000.png ~/Work/face_liveness_detection_data/fake1/0220.png
    } else {
        cout << "error" << endl;
    }
    return 0;
}
