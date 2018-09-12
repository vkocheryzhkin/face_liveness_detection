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

namespace fs = std::experimental::filesystem;

using namespace dlib;
using namespace std;

void performSVM(std::string csvFile)
{
    ifstream file(csvFile);
    std::string line;
    int number_of_lines = 0;
    while (std::getline(file, line))
      ++number_of_lines;

    file.close();

    int features = 68 + 1;
    cv::Mat training_mat(number_of_lines, features - 1, CV_32FC1);
    cv::Mat labels(number_of_lines, 1, CV_32SC1);

    ifstream f(csvFile);
    int line_num = 0;

    while (std::getline(f, line))
    {
        std::istringstream iss(line);
        float i;
        std::vector<float> vect;
        while (iss >> i)
        {
            vect.push_back(i);
            if (iss.peek() == ',')
                iss.ignore();
        }
        for (int j = 0; j < vect.size(); j++) {
            if (j == 68) {
                int tt = (int)vect.at(j);
                labels.at<int>(line_num, 0) = tt;
            } else{
                training_mat.at<float>(line_num, j) = vect.at(j);
            }
        }
        line_num++;
    }

    f.close();

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::RBF);

    try {
        svm->trainAuto(training_mat, cv::ml::ROW_SAMPLE, labels);
        svm->save("liveness.svm");
    } catch(cv::Exception& e) {
        cout << e.what() << endl;
    }
    return;
}



int main(int argc, char** argv )
{
    if ( argc != 2 )
    {
        printf("usage2: csvFile>\n");
        return -1;
    }

    std::string csvFile = argv[1];
    performSVM(csvFile);//~/Work/face_liveness_detection_data/combine.csv

    return 0;
}
