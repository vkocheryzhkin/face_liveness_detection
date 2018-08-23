#include <stdio.h>
#include <opencv2/opencv.hpp>
#include <dlib/image_processing/frontal_face_detector.h>
#include <dlib/image_processing/render_face_detections.h>
#include <dlib/image_processing.h>
#include <dlib/gui_widgets.h>
#include <dlib/dnn.h>
#include <dlib/image_io.h>

using namespace dlib;
using namespace std;

int main(int argc, char** argv )
{
    if ( argc != 3 )
    {
        printf("usage: <Image_Path1> <Image_Path2>\n");
        return -1;
    }

    frontal_face_detector detector = get_frontal_face_detector();
    shape_predictor sp;
    deserialize("shape_predictor_68_face_landmarks.dat") >> sp;

    image_window win, win_faces;
    std::vector<cv::Point2f> obj, scene;
    std::vector<cv::KeyPoint> keypoints_object, keypoints_scene;
    for (int i = 1; i < argc; ++i)
    {
        cout << "processing image " << argv[i] << endl;
        array2d<rgb_pixel> img;
        load_image(img, argv[i]);

        // Now tell the face detector to give us a list of bounding boxes
        // around all the faces in the image.
        std::vector<rectangle> dets = detector(img);
        cout << "Number of faces detected: " << dets.size() << endl;

        // Now we will go ask the shape_predictor to tell us the pose of
        // each face we detected.
        std::vector<full_object_detection> shapes;
        for (unsigned long j = 0; j < dets.size(); ++j)
        {
            full_object_detection shape = sp(img, dets[j]);
            if (i == 1) {
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
            cout << "number of parts: " << shape.num_parts() << endl;
            cout << "pixel position of first part:  " << shape.part(0) << endl;
            cout << "pixel position of second part: " << shape.part(1) << endl;
            // You get the idea, you can get all the face part locations if
            // you want them.  Here we just store them in shapes so we can
            // put them on the screen.
            shapes.push_back(shape);
        }

        // Now let's view our face poses on the screen.
        win.clear_overlay();
        win.set_image(img);
        win.add_overlay(render_face_detections(shapes));

        // We can also extract copies of each face that are cropped, rotated upright,
        // and scaled to a standard size as shown here:
        dlib::array<array2d<rgb_pixel> > face_chips;
        extract_image_chips(img, get_face_chip_details(shapes), face_chips);
        win_faces.set_image(tile_images(face_chips));
    }

    std::vector< cv::DMatch > good_matches;
    for (unsigned long i = 0; i < keypoints_object.size(); ++i) {
        good_matches.push_back(cv::DMatch(i, i, 1));
    }
    
    cv::Mat H = findHomography(obj, scene, CV_RANSAC);

    cv::Mat img_object = cv::imread(argv[1], CV_LOAD_IMAGE_GRAYSCALE);
    cv::Mat img_scene = cv::imread(argv[2], CV_LOAD_IMAGE_GRAYSCALE);

    cv::Mat img_matches;
    drawMatches(img_object, keypoints_object, img_scene, keypoints_scene,
        good_matches, img_matches, cv::Scalar::all(-1), cv::Scalar::all(-1),
        std::vector<char>(), cv::DrawMatchesFlags::NOT_DRAW_SINGLE_POINTS);

    cv::imwrite("res.png", img_matches);

    return 0;
}