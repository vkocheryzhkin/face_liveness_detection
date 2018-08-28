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
namespace fs = std::experimental::filesystem;

using namespace dlib;
using namespace std;


std::vector<float> getDistances68(std::string left, std::string right)
{
    std::vector<float> res;

    std::vector<string> image_files;
    image_files.push_back(left);
    image_files.push_back(right);

    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

//    image_window win, win_faces;
    std::vector<cv::Point2f> obj, scene;
    std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
    for (int i = 0; i < image_files.size(); ++i) //process two images
    {
        //cout << "processing image " << image_files[i] << endl;
        array2d<rgb_pixel> img;
        load_image(img, image_files[i]);

        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces in the image.
        std::vector<rectangle> dets = detector(img);
        //cout << "Number of faces detected: " << dets.size() << endl;

        // Now we will go ask the shape_predictor to tell us the pose of
        // each face we detected.
        std::vector<full_object_detection> shapes;
        for (unsigned long j = 0; j < dets.size(); ++j)
        {
            full_object_detection shape = sp(img, dets[j]);
            if (i == 0) { //obj is left
                for (unsigned long s = 0; s < shape.num_parts(); ++s) {
                    cv::Point2f p = cv::Point2f(shape.part(s).x(), shape.part(s).y());
                    obj.push_back(p);
                    keypoints_object.push_back(cv::KeyPoint(p,1));
                }
            }
            else {
                for (unsigned long s = 0; s < shape.num_parts(); ++s) {
                    cv::Point2f p = cv::Point2f(shape.part(s).x(), shape.part(s).y());
                    scene.push_back(p);
                    keypoints_scene.push_back(cv::KeyPoint(p, 1));
                }
            }
            //cout << "number of parts: " << shape.num_parts() << endl;
            //cout << "pixel position of first part:  " << shape.part(0) << endl;
            //cout << "pixel position of second part: " << shape.part(1) << endl;
            // You get the idea, you can get all the face part locations if
            // you want them.  Here we just store them in shapes so we can
            // put them on the screen.
            shapes.push_back(shape);
        }

        // Now let's view our face poses on the screen.
        //win.clear_overlay();
        //win.set_image(img);
        //win.add_overlay(render_face_detections(shapes));

        // We can also extract copies of each face that are cropped, rotated upright,
        // and scaled to a standard size as shown here:
//        dlib::array<array2d<rgb_pixel> > face_chips;
//        extract_image_chips(img, get_face_chip_details(shapes), face_chips);
//        win_faces.set_image(tile_images(face_chips));
    }

    //--------------------------------------Homography(H) (begin)
    std::vector< cv::DMatch > good_matches;
    for (unsigned long i = 0; i < keypoints_object.size(); ++i) {
        good_matches.push_back(cv::DMatch(i, i, 1));
    }

    cv::Mat H = findHomography(obj, scene, CV_RANSAC);

    cv::Mat img_object = cv::imread(image_files[0], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img_scene = cv::imread(image_files[1], CV_LOAD_IMAGE_GRAYSCALE);

    cv::Mat img_matches;
    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
        good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    //cv::imwrite("res_H.png", img_matches);


    //--------------------------------------Homography(H) (end)

    //--------------------------------------Map left points to right using H (begin)
    int num_points = 68;
    std::vector<cv::Point2f> scene__mapped_points(num_points);
    cv::perspectiveTransform(obj, scene__mapped_points, H);

    for (int i = 0; i < num_points; ++i) {
        cv::Point2f p = scene[i];
        cv::Point2f p1 = scene__mapped_points[i];
        cv::circle(img_scene, p, 1, cv::Scalar(0), 3, 8, 0); //orig black
        cv::circle(img_scene, p1, 1, cv::Scalar(255), 3, 8, 0); //mapped white
        double dist = cv::norm(p - p1);
        //cout << dist << endl;
        res.push_back(dist);
    }

    //cv::imwrite("res_right.png", img_scene);
    //--------------------------------------Map left points to right using H (end)

    return res;
}

std::size_t numbeOfFilesInDirectory(std::experimental::filesystem::path path)
{
    using std::experimental::filesystem::directory_iterator;
    using fp = bool (*)( const std::experimental::filesystem::path&);
    return std::count_if(directory_iterator(path), directory_iterator{},
                         (fp)std::experimental::filesystem::is_regular_file);
}

void processImagesDir(std::string folder)
{
    size_t n = numbeOfFilesInDirectory(folder);
    int r = 2;
//    cout << t << endl;

    std::vector<std::string> fileList;
    for (auto & p : fs::directory_iterator(folder)) {
        std::string name = p.path();
        fileList.push_back(name);
    }
//        std::cout << p << std::endl;


    std::vector<pair<int, int>> pairIds;
    std::vector<bool> v(n);
    std::fill(v.begin(), v.begin() + r, true);

    do {
       std::vector<int> pair;
       for (int i = 0; i < n; ++i) {
           if (v[i]) {
               //std::cout << (i + 1) << " ";
               pair.push_back(i);
           }
       }
       pairIds.push_back(std::make_pair(pair.at(0), pair.at(1)));
    } while (std::prev_permutation(v.begin(), v.end()));

    ofstream file;

    string resFile = "results.csv";
    file.open(resFile);
    for (auto& t : pairIds) {
        std::string left = fileList.at(t.first);
        std::string right = fileList.at(t.second);
        std::vector<float> res = getDistances68(left, right);
        //--------------------------------------MSE (begin)
        float error = 0;
        for (int i = 0; i < res.size(); ++i) {
            float dist = res.at(i);
            error += pow(dist, 2);
        }
        error = error / res.size();
        cout << left << " " << right << " mse: " << error << endl;
        //--------------------------------------MSE (end)
        for (int i = 0; i < res.size(); i++)
        {
            file << res[i] << ",";
        }

        if (folder.find("fake") != std::string::npos) {
            file << "1\n";
        } else {
            file << "0\n";
        }
    }
    file.close();
}

void performSVM(std::string csvFile)
{
    ifstream file(csvFile);
    std::string line;
    int number_of_lines = 0;
    while (std::getline(file, line))
      ++number_of_lines;

    //cout << number_of_lines << endl;
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
                //cout << labels.at<int>(line_num, 0) << endl;
            } else{
                training_mat.at<float>(line_num, j) = vect.at(j);
            }
        }
        line_num++;
    }

    f.close();

    cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::create();

    svm->setType(cv::ml::SVM::C_SVC);
    svm->setKernel(cv::ml::SVM::POLY);
    svm->setGamma(3); //todo: tune
    svm->setDegree(3); //todo: tune

    try {
        svm->train( training_mat , cv::ml::ROW_SAMPLE , labels );
        svm->save("liveness.svm");
    } catch(cv::Exception& e) {
        cout << e.what() << endl;
    }

    return;
}

void predict(std::string svmFile, std::string left, std::string right) {
    std::vector<float> res = getDistances68(left, right);
    cv::Mat test_mat(1, 68, CV_32FC1);
    for(int i = 0; i < res.size(); i++) {
        test_mat.at<float>(0, i) = res.at(i);
    }

    try {
        cv::Ptr<cv::ml::SVM> svm = cv::ml::SVM::load(svmFile);
        float t = svm->predict(test_mat);
        return t;
    } catch(cv::Exception& e) {
        cout << e.what() << endl;
    }
}

int main(int argc, char** argv )
{
    if ( argc == 1 )
    {
        printf("usage1: process imageDir\n");
        printf("usage2: train csvFile>\n");
        printf("usage3: predict image1 image2\n");
        return -1;
    }

    std::string type = argv[1];
    if (type == "process") {
        std::string dir = argv[2];
        processImagesDir(dir);
    } else if (type == "train") {
        std::string csvFile = argv[2];
        performSVM(csvFile);
    } else { //predict
        std::string svmFile = argv[2];
        std::string left = argv[3];
        std::string right = argv[4];
        predict(svmFile, left, right);
    }

    return 0;
}
